"""Deliverable 4: iterative global magnitude (L1) unstructured pruning + masked fine-tuning."""
import argparse, torch, torch.nn as nn
import torch.nn.utils.prune as prune
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--final-sparsity", type=float, default=0.80)
ap.add_argument("--iterations", type=int, default=5)
ap.add_argument("--ft-epochs", type=int, default=2, help="fine-tune epochs after each pruning step")
ap.add_argument("--final-epochs", type=int, default=8, help="extra fine-tune epochs at final sparsity")
ap.add_argument("--lr", type=float, default=0.01)
a = ap.parse_args()
ensure_dirs(); seed_everything(); dev = get_device()
train_loader, test_loader = get_loaders()

model = load_model("fp32").to(dev)
targets = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

def sparsity():
    z = sum((getattr(m, n) == 0).sum().item() for m, n in targets)
    t = sum(getattr(m, n).numel() for m, n in targets)
    return 100.0 * z / t

def finetune(epochs, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs) * len(train_loader))
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, dev, sch)

print(f"dense acc: {evaluate(model, test_loader, dev):.2f}%")
prev = 0.0
for i in range(1, a.iterations + 1):
    # cubic schedule; amount is a fraction of the *remaining* weights
    target = a.final_sparsity * (1 - (1 - i / a.iterations) ** 3)
    amount = (target - prev) / (1 - prev); prev = target
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)  # adds weight_mask
    acc0 = evaluate(model, test_loader, dev)
    finetune(a.final_epochs if i == a.iterations else a.ft_epochs, a.lr)   # masks stay applied
    print(f"iter {i}/{a.iterations} sparsity {sparsity():.2f}%  acc pruned {acc0:.2f}% -> fine-tuned "
          f"{evaluate(model, test_loader, dev):.2f}%")

for m, n in targets:
    prune.remove(m, n)                            # make pruning permanent (bake mask into weight)
model.cpu()
print(f"final sparsity {sparsity():.2f}% | test acc {evaluate(model, test_loader):.2f}%")
save_fp32(model, PATHS["unstructured"])
print("saved", PATHS["unstructured"])
