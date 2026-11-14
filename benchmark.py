"""Deliverable 7: evaluate every available model on the full CIFAR-10 test set (CPU).
Writes results/benchmark_results.csv (summary) and results/predictions.csv (per-image predictions)."""
import argparse, csv, os, time
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=ALL_MODELS)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--lat-iters", type=int, default=200)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
ensure_dirs(); torch.set_num_threads(a.threads); set_quant_backend()
_, test_loader = get_loaders(a.batch_size, workers=0)
x1 = next(iter(get_loaders(1, workers=0)[1]))[0]

BASE = {"ptq": "fp32", "qat": "fp32", "fp32_onnx": "fp32", "fp32_onnx_int8": "fp32",
        "unstructured_onnx": "unstructured", "structured_onnx": "structured"}
arch_cache = {}
def arch_stats(base):   # params/MACs come from the FP32-format twin of each model
    if base not in arch_cache:
        m = load_model(base); tot, nz = count_params(m); arch_cache[base] = (tot, nz, count_macs(m))
    return arch_cache[base]

rows, preds, labels = [], {}, None
for name in a.models:
    path = PATHS.get(name) or ONNX_PATHS[name]
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found"); continue
    model = load_model(name)
    labels, p, conf, fwd = predict_all(model, test_loader)
    preds[name] = (p, conf)
    with torch.no_grad():
        for _ in range(20): model(x1)
        ts = []
        for _ in range(a.lat_iters):
            t0 = time.perf_counter(); model(x1); ts.append((time.perf_counter() - t0) * 1000)
    tot, nz, macs = arch_stats(BASE.get(name, name))
    rows.append({
        "model": name, "format": os.path.splitext(path)[1][1:],
        "accuracy_pct": round(100 * (labels == p).mean(), 2),
        "params_total": tot, "params_nonzero": nz, "sparsity_pct": round(100 * (1 - nz / tot), 2),
        "macs_millions": round(macs / 1e6, 1),
        "file_size_mb": round(file_size_mb(path), 3), "gzip_size_mb": round(gz_size_mb(path), 3),
        "latency_bs1_mean_ms": round(np.mean(ts), 3), "latency_bs1_p50_ms": round(np.percentile(ts, 50), 3),
        "latency_bs1_p95_ms": round(np.percentile(ts, 95), 3),
        f"throughput_img_s_bs{a.batch_size}": round(len(labels) / fwd, 1)})
    print(f"{name:18s} acc {rows[-1]['accuracy_pct']:6.2f}%  lat {rows[-1]['latency_bs1_mean_ms']:.2f} ms  "
          f"size {rows[-1]['file_size_mb']:.2f} MB")

if rows:
    ref = next((r for r in rows if r["model"] == "fp32"), rows[0])["latency_bs1_mean_ms"]
    for r in rows: r["speedup_vs_fp32"] = round(ref / r["latency_bs1_mean_ms"], 2)
    with open(f"{RESULTS_DIR}/benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(f"{RESULTS_DIR}/predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_label", "true_class"] + [f"{n}_{s}" for n in preds for s in ("pred", "conf")])
        for i in range(len(labels)):
            w.writerow([i, labels[i], CLASSES[labels[i]]] +
                       [v for n in preds for v in (preds[n][0][i], f"{preds[n][1][i]:.4f}")])
    print(f"wrote {RESULTS_DIR}/benchmark_results.csv and {RESULTS_DIR}/predictions.csv")
