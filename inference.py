"""Deliverable 6: unified inference for fp32 | ptq | qat | unstructured | structured (+ ONNX variants).
  python inference.py --model qat --image cat.png dog.jpg
  python inference.py --model structured --test-samples 8"""
import argparse, torch, torchvision.transforms as T
from PIL import Image
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=ALL_MODELS)
ap.add_argument("--image", nargs="*", default=[])
ap.add_argument("--test-samples", type=int, default=0)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)

model = load_model(a.model)
prep = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])

@torch.no_grad()
def classify(x):
    prob = torch.softmax(model(x).float(), 1)[0]
    c, p = prob.max(0)
    return CLASSES[p], c.item()

for path in a.image:
    cls, conf = classify(prep(Image.open(path).convert("RGB")).unsqueeze(0))
    print(f"{path}: {cls} ({conf:.1%})")

if a.test_samples:
    _, te = get_loaders(1, workers=0)
    for i, (x, y) in enumerate(te):
        if i >= a.test_samples: break
        cls, conf = classify(x)
        print(f"test[{i}] true={CLASSES[y.item()]:<10} pred={cls:<10} ({conf:.1%})")
