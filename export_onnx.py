"""Deliverable 8: ONNX export (FP32, unstructured, structured) + ORT static-INT8 model, verified on CPU.
Note: PyTorch eager-mode INT8 (PTQ/QAT) models are deployed as TorchScript; the ONNX INT8 model here is
produced by ONNX Runtime static quantization of the FP32 graph."""
import numpy as np, torch
from common import *

ensure_dirs(); set_quant_backend()
dummy = torch.randn(2, 3, 32, 32)

def export(name, onnx_name):
    m = load_model(name); path = ONNX_PATHS[onnx_name]
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=13,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:  torch.onnx.export(m, dummy, path, dynamo=False, **kw)
    except TypeError: torch.onnx.export(m, dummy, path, **kw)
    diff = np.abs(OrtModel(path)(dummy).numpy() - m(dummy).detach().numpy()).max()
    print(f"exported {path} (max |torch - onnxruntime| = {diff:.2e})")

for n in ("fp32", "unstructured", "structured"):
    if os.path.exists(PATHS[n]): export(n, n + "_onnx")

# ---- ONNX Runtime static INT8 quantization of the FP32 ONNX graph
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
class Reader(CalibrationDataReader):
    def __init__(self): self.it = iter([x.numpy() for x, _ in get_calib_loader(500, 50)])
    def get_next(self):
        b = next(self.it, None); return None if b is None else {"input": b}
quantize_static(ONNX_PATHS["fp32_onnx"], ONNX_PATHS["fp32_onnx_int8"], Reader(),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
print("exported", ONNX_PATHS["fp32_onnx_int8"])
