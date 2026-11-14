"""Shared utilities: data, model definition, training/eval helpers, model registry."""
import gzip, os, platform, random, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.quantization import resnet18 as _q_resnet18

CKPT_DIR, RESULTS_DIR, LOG_DIR = "checkpoints", "results", "logs"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

PATHS = {   # PyTorch models. *.pth = FP32 dict checkpoint, *.ts = TorchScript INT8
    "fp32": f"{CKPT_DIR}/resnet18_fp32.pth",
    "ptq": f"{CKPT_DIR}/resnet18_ptq_int8.ts",
    "qat": f"{CKPT_DIR}/resnet18_qat_int8.ts",
    "unstructured": f"{CKPT_DIR}/resnet18_unstructured_pruned.pth",
    "structured": f"{CKPT_DIR}/resnet18_structured_pruned.pth",
}
ONNX_PATHS = {
    "fp32_onnx": f"{CKPT_DIR}/onnx/resnet18_fp32.onnx",
    "unstructured_onnx": f"{CKPT_DIR}/onnx/resnet18_unstructured.onnx",
    "structured_onnx": f"{CKPT_DIR}/onnx/resnet18_structured.onnx",
    "fp32_onnx_int8": f"{CKPT_DIR}/onnx/resnet18_fp32_ort_int8.onnx",
}
ALL_MODELS = list(PATHS) + list(ONNX_PATHS)


def ensure_dirs():
    for d in (CKPT_DIR, RESULTS_DIR, LOG_DIR, f"{CKPT_DIR}/onnx"):
        os.makedirs(d, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- data
def _tf(train):
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


def _dataset(train, tf, root="./data"):
    if os.environ.get("FAKE_DATA") == "1":  # smoke-test mode (no download)
        return torchvision.datasets.FakeData(size=256 if train else 128, image_size=(3, 32, 32),
                                             num_classes=10, transform=tf,
                                             random_offset=0 if train else 1)
    return torchvision.datasets.CIFAR10(root, train=train, download=True, transform=tf)


def get_loaders(batch_size=128, workers=2, root="./data"):
    pin = torch.cuda.is_available()
    tr = torch.utils.data.DataLoader(_dataset(True, _tf(True), root), batch_size=batch_size,
                                     shuffle=True, num_workers=workers, pin_memory=pin)
    te = torch.utils.data.DataLoader(_dataset(False, _tf(False), root), batch_size=batch_size,
                                     shuffle=False, num_workers=workers, pin_memory=pin)
    return tr, te


def get_calib_loader(n=1000, batch_size=50, seed=0, root="./data"):
    """Random subset of the *training* set, no augmentation."""
    ds = _dataset(True, _tf(False), root)
    idx = np.random.RandomState(seed).choice(len(ds), min(n, len(ds)), replace=False)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                                       batch_size=batch_size, shuffle=False)


# ----------------------------------------------------------------- model
def named_blocks(model):
    for ln in ("layer1", "layer2", "layer3", "layer4"):
        for j, b in enumerate(getattr(model, ln)):
            yield f"{ln}.{j}", b


def build_model(cfg=None):
    """CIFAR ResNet-18 (quantization-ready torchvision variant).
    cfg: optional list of 8 ints = kept channels of conv1 inside each BasicBlock
    (used to rebuild a structurally pruned network)."""
    m = _q_resnet18(weights=None, quantize=False)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 10)
    if cfg is not None:
        for (_, b), k in zip(named_blocks(m), cfg):
            c1, c2 = b.conv1, b.conv2
            b.conv1 = nn.Conv2d(c1.in_channels, k, 3, c1.stride, 1, bias=False)
            b.bn1 = nn.BatchNorm2d(k)
            b.conv2 = nn.Conv2d(k, c2.out_channels, 3, 1, 1, bias=False)
    return m


def save_fp32(model, path, cfg=None):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)


def set_quant_backend():
    engines = torch.backends.quantized.supported_engines
    if "fbgemm" in engines and platform.machine().lower() in ("x86_64", "amd64"):
        backend = "fbgemm"
    elif "qnnpack" in engines:
        backend = "qnnpack"
    else:
        raise RuntimeError(f"No usable quantization backend in {engines}")
    torch.backends.quantized.engine = backend
    return backend


def save_int8(model, name):
    """Save converted INT8 model as TorchScript (executable) + state_dict."""
    model = model.cpu().eval()
    ex = torch.randn(1, 3, 32, 32)
    ts = torch.jit.trace(model, ex, check_trace=False)
    torch.jit.save(ts, PATHS[name])
    torch.save(model.state_dict(), PATHS[name].replace(".ts", "_state_dict.pth"))
    chk = torch.jit.load(PATHS[name]); 
    print(f"saved {PATHS[name]} (max |diff| eager vs scripted: "
          f"{(chk(ex) - model(ex)).abs().max().item():.2e})")


class OrtModel:
    """ONNX Runtime wrapper exposing the torch-like call interface."""
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
    def eval(self): return self
    def __call__(self, x):
        return torch.from_numpy(self.sess.run(None, {self.inp: x.numpy()})[0])


def load_model(name):
    """Load any registered model, CPU, eval mode."""
    if name in ONNX_PATHS:
        return OrtModel(ONNX_PATHS[name])
    path = PATHS[name]
    if path.endswith(".ts"):
        set_quant_backend()
        return torch.jit.load(path, map_location="cpu").eval()
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck.get("cfg"))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


# ----------------------------------------------------------------- train / eval
def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    model.train(); crit = nn.CrossEntropyLoss()
    loss_sum = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x); loss = crit(out, y)
        loss.backward(); optimizer.step()
        if scheduler is not None: scheduler.step()
        loss_sum += loss.item() * y.size(0); correct += (out.argmax(1) == y).sum().item(); n += y.size(0)
    return loss_sum / n, 100.0 * correct / n


@torch.no_grad()
def predict_all(model, loader, device="cpu"):
    """Returns labels, preds, confidences (numpy) and pure forward time (s)."""
    L, P, C, t = [], [], [], 0.0
    for x, y in loader:
        x = x.to(device)
        t0 = time.perf_counter(); out = model(x); t += time.perf_counter() - t0
        prob = torch.softmax(out.float(), 1)
        c, p = prob.max(1)
        L.append(y.numpy()); P.append(p.cpu().numpy()); C.append(c.cpu().numpy())
    return np.concatenate(L), np.concatenate(P), np.concatenate(C), t


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    l, p, _, _ = predict_all(model, loader, device)
    return 100.0 * (l == p).mean()


# ----------------------------------------------------------------- stats
def count_params(model):
    ps = [p for p in model.parameters()]
    return sum(p.numel() for p in ps), sum((p != 0).sum().item() for p in ps)


@torch.no_grad()
def count_macs(model, size=(1, 3, 32, 32)):
    macs, hooks = [0], []
    def conv(m, i, o): macs[0] += (o.numel() // o.shape[0]) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def fc(m, i, o): macs[0] += m.in_features * m.out_features
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(fc))
    model.eval(); model(torch.zeros(size))
    for h in hooks: h.remove()
    return macs[0]


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def gz_size_mb(path):
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), 6)) / 1e6
