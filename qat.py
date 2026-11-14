"""Deliverable 3: Quantization-Aware Training initialised from the FP32 checkpoint."""
import argparse, torch
import torch.ao.nn.intrinsic.qat as nniqat
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert, disable_observer
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.005)
ap.add_argument("--batch-size", type=int, default=128)
a = ap.parse_args()
ensure_dirs(); seed_everything()

backend = set_quant_backend(); print("quantization backend:", backend)
dev = get_device()
train_loader, test_loader = get_loaders(a.batch_size)

model = load_model("fp32")                       # pretrained FP32 init
model.train()
model.fuse_model(is_qat=True)                    # fused QAT modules
model.qconfig = get_default_qat_qconfig(backend)
prepare_qat(model, inplace=True)                 # insert FakeQuantize modules
model.to(dev)

opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_loader))
for ep in range(1, a.epochs + 1):
    if ep > max(1, a.epochs // 2):  model.apply(disable_observer)          # freeze ranges
    if ep > max(1, a.epochs // 2 - 1): model.apply(nniqat.freeze_bn_stats)  # freeze BN stats
    loss, acc = train_one_epoch(model, train_loader, opt, dev, sched)
    print(f"QAT epoch {ep}/{a.epochs} loss {loss:.4f} train {acc:.2f}%")

model.cpu().eval()
int8 = convert(model, inplace=False)             # -> real INT8 model
print(f"QAT INT8 test acc: {evaluate(int8, test_loader):.2f}%")
save_int8(int8, "qat")
