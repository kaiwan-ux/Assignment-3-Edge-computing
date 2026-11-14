"""Deliverable 5: structured channel pruning with physical architecture reconstruction.

Each ResNet BasicBlock is conv1->bn1->relu->conv2->bn2 (+ skip). The block's *output*
channels are tied to the residual stream, so we prune the INTERNAL channels only
(conv1 filters / bn1 / conv2 input channels). This keeps every tensor dimension valid
and yields a genuinely smaller dense network (no masked zero channels)."""
import argparse, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=0.5, help="fraction of internal channels removed")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=0.01)
a = ap.parse_args()
ensure_dirs(); seed_everything(); dev = get_device()
train_loader, test_loader = get_loaders()

old = load_model("fp32")
osd, keep, cfg = old.state_dict(), {}, []
for name, b in named_blocks(old):
    score = b.conv1.weight.detach().abs().sum((1, 2, 3))          # L1 norm per filter
    n = score.numel()
    k = min(n, max(8, int(round(n * (1 - a.ratio) / 8)) * 8))     # keep a multiple of 8
    keep[name] = torch.topk(score, k).indices.sort().values
    cfg.append(k)
print("kept internal channels per block:", cfg)

new = build_model(cfg)                                            # reconstructed, smaller net
nsd = new.state_dict()
for k in nsd:                                                     # copy all same-shaped tensors
    if nsd[k].shape == osd[k].shape: nsd[k] = osd[k].clone()
for name, idx in keep.items():                                    # slice the pruned ones
    nsd[f"{name}.conv1.weight"] = osd[f"{name}.conv1.weight"][idx].clone()
    for s in ("weight", "bias", "running_mean", "running_var"):
        nsd[f"{name}.bn1.{s}"] = osd[f"{name}.bn1.{s}"][idx].clone()
    nsd[f"{name}.conv2.weight"] = osd[f"{name}.conv2.weight"][:, idx].clone()
new.load_state_dict(nsd)

p0, _ = count_params(old); p1, _ = count_params(new)
print(f"params {p0/1e6:.2f}M -> {p1/1e6:.2f}M | MACs {count_macs(old)/1e6:.1f}M -> {count_macs(new)/1e6:.1f}M")
print(f"acc right after pruning: {evaluate(new, test_loader):.2f}%")

new.to(dev)
opt = torch.optim.SGD(new.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_loader))
for ep in range(1, a.epochs + 1):
    loss, acc = train_one_epoch(new, train_loader, opt, dev, sch)
    print(f"fine-tune {ep}/{a.epochs} loss {loss:.4f} train {acc:.2f}% test {evaluate(new, test_loader, dev):.2f}%")

new.cpu()
save_fp32(new, PATHS["structured"], cfg=cfg)
print("saved", PATHS["structured"])
