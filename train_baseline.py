"""Deliverable 1: train FP32 ResNet-18 on CIFAR-10 and save the checkpoint."""
import argparse, csv
import torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

ensure_dirs(); seed_everything(a.seed); dev = get_device()
train_loader, test_loader = get_loaders(a.batch_size)
model = build_model().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(train_loader))

with open(f"{LOG_DIR}/train_baseline.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
    for ep in range(1, a.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, opt, dev, sched)
        te = evaluate(model, test_loader, dev)
        w.writerow([ep, f"{loss:.4f}", f"{acc:.2f}", f"{te:.2f}"]); f.flush()
        print(f"epoch {ep:3d}/{a.epochs} loss {loss:.4f} train {acc:.2f}% test {te:.2f}%")

save_fp32(model.cpu(), PATHS["fp32"])
print("saved", PATHS["fp32"])
