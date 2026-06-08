"""
Person A – Part 1: Centralized Baseline
Trains PointNet++ and DGCNN on the full training set.
Logs train/val loss & accuracy per epoch to CSV.
Saves best checkpoint per architecture (used as teacher in Part 3).

Usage:
    python train_centralized.py --csv data/labeled_dataset.csv --epochs 100
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import SEED, BATCH_SIZE
from data_pipeline import get_dataloaders

# Your partner's models
from models.pointnet2 import PointNet2          # (B, N, C) → (B, num_classes)
from models.dgcnn import DGCNN                  # (B, N, C) → (B, num_classes)
from utils import count_parameters

torch.manual_seed(SEED)


# ── helpers ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            logits = model(pts)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def train_model(model_name: str,
                model: nn.Module,
                train_loader,
                val_loader,
                num_classes: int,
                args,
                device):

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path  = os.path.join(out_dir, f"{model_name}_log.csv")
    ckpt_path = os.path.join(out_dir, f"{model_name}_best.pt")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'='*60}")
    print(f"  Training  : {model_name}")
    print(f"  Params    : {count_parameters(model):,}")
    print(f"  Epochs    : {args.epochs}  |  Device: {device}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
            scheduler.step()
            elapsed = time.time() - t0

            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                             f"{va_loss:.4f}", f"{va_acc:.4f}", f"{elapsed:.1f}"])
            f.flush()

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc,
                    "num_classes": num_classes,
                }, ckpt_path)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d}/{args.epochs} | "
                      f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                      f"val loss={va_loss:.4f} acc={va_acc:.3f} | "
                      f"best_val={best_val_acc:.3f} | {elapsed:.1f}s")

    print(f"\n  → Best val acc : {best_val_acc:.4f}")
    print(f"  → Checkpoint   : {ckpt_path}")
    print(f"  → Log          : {log_path}")
    return ckpt_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True, help="Path to labeled_dataset.csv")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--out_dir", default="checkpoints/centralized")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, num_classes = get_dataloaders(
        args.csv, batch_size=args.batch, num_workers=args.workers
    )

    architectures = {
        "pointnet2": PointNet2(num_classes=num_classes),
        "dgcnn":     DGCNN(num_classes=num_classes),
    }

    for name, model in architectures.items():
        train_model(name, model, train_loader, val_loader, num_classes, args, device)


if __name__ == "__main__":
    main()
