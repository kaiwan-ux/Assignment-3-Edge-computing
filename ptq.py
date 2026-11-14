"""Deliverable 2: Post-Training Static Quantization (fuse -> observe -> calibrate -> convert)."""
import argparse, torch
from torch.ao.quantization import get_default_qconfig, prepare, convert
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--calib-samples", type=int, default=1000)
a = ap.parse_args()
ensure_dirs(); seed_everything()

backend = set_quant_backend(); print("quantization backend:", backend)
_, test_loader = get_loaders(workers=0)
model = load_model("fp32")                       # trained FP32 weights, CPU, eval
fp32_acc = evaluate(model, test_loader)

model.fuse_model(is_qat=False)                   # Conv+BN(+ReLU) fusion
model.qconfig = get_default_qconfig(backend)     # static qconfig for the chosen backend
prepare(model, inplace=True)                     # insert observers

calib = get_calib_loader(a.calib_samples)        # calibration subset of the TRAIN set
with torch.no_grad():
    for x, _ in calib:
        model(x)
print(f"calibrated on {len(calib.dataset)} training images")

convert(model, inplace=True)                     # -> INT8
int8_acc = evaluate(model, test_loader)
print(f"FP32 acc {fp32_acc:.2f}% -> PTQ INT8 acc {int8_acc:.2f}%")
save_int8(model, "ptq")
