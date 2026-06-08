"""
Person A – Part 3: Knowledge Distillation

For each architecture (PointNet++, DGCNN):
  • Load frozen centralized teacher checkpoint
  • Train a lightweight student with KD loss:
        L = (1-α)·CE(ŷ, y) + α·τ²·KL(σ(zs/τ) ∥ σ(zt/τ))
  • Grid search over τ ∈ {1,2,4,8} and α ∈ {0.3,0.5,0.7}
  • Produce a heatmap of best validation accuracy per (τ, α) cell

Usage:
    python train_distillation.py \\
        --csv data/labeled_dataset.csv \\
        --ckpt_pn2 checkpoints/centralized/pointnet2_best.pt \\
        --ckpt_dgcnn checkpoints/centralized/dgcnn_best.pt \\
        --epochs 60
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import SEED, BATCH_SIZE
from data_pipeline import get_dataloaders
from student_models import StudentPointNet, StudentDGCNN
from models.pointnet2 import PointNet2
from models.dgcnn import DGCNN
from utils import count_parameters

torch.manual_seed(SEED)

TAUS   = [1, 2, 4, 8]
ALPHAS = [0.3, 0.5, 0.7]


# ── KD loss ───────────────────────────────────────────────────────────────────

def kd_loss(student_logits, teacher_logits, labels, alpha, tau):
    """
    L = (1-α)·CE(ŷ, y)  +  α·τ²·KL(σ(zs/τ) ∥ σ(zt/τ))
    """
    ce   = F.cross_entropy(student_logits, labels)
    p_s  = F.log_softmax(student_logits / tau, dim=-1)
    p_t  = F.softmax(teacher_logits    / tau, dim=-1)
    kl   = F.kl_div(p_s, p_t, reduction="batchmean") * (tau ** 2)
    return (1 - alpha) * ce + alpha * kl


# ── single training run ───────────────────────────────────────────────────────

def train_student(student, teacher, train_loader, val_loader,
                  alpha, tau, epochs, lr, device) -> float:
    """Train student and return best validation accuracy."""
    student = student.to(device)
    teacher.eval()

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ---- train ----
        student.train()
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(pts)
            s_logits = student(pts)
            loss = kd_loss(s_logits, t_logits, labels, alpha, tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ---- validate ----
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts, labels = pts.to(device), labels.to(device)
                preds = student(pts).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


# ── grid search ──────────────────────────────────────────────────────────────

def grid_search(arch_name: str,
                teacher: nn.Module,
                StudentCls,
                in_channels: int,
                num_classes: int,
                train_loader,
                val_loader,
                args,
                device):

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"{arch_name}_kd_grid.csv")
    results  = np.zeros((len(TAUS), len(ALPHAS)))   # rows=τ, cols=α

    teacher = teacher.to(device)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arch", "tau", "alpha", "best_val_acc"])

        for i, tau in enumerate(TAUS):
            for j, alpha in enumerate(ALPHAS):
                print(f"  [{arch_name}] τ={tau}  α={alpha} …", end=" ", flush=True)
                student = StudentCls(in_channels=in_channels, num_classes=num_classes)
                val_acc = train_student(
                    student, teacher,
                    train_loader, val_loader,
                    alpha=alpha, tau=tau,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                )
                results[i, j] = val_acc
                writer.writerow([arch_name, tau, alpha, f"{val_acc:.4f}"])
                f.flush()
                print(f"val_acc={val_acc:.4f}")

    # ---- save best student ----
    best_idx   = np.unravel_index(results.argmax(), results.shape)
    best_tau   = TAUS[best_idx[0]]
    best_alpha = ALPHAS[best_idx[1]]
    print(f"\n  Best for {arch_name}: τ={best_tau}, α={best_alpha}, val_acc={results.max():.4f}")

    best_student = StudentCls(in_channels=in_channels, num_classes=num_classes)
    val_acc = train_student(
        best_student, teacher,
        train_loader, val_loader,
        alpha=best_alpha, tau=best_tau,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    ckpt_path = os.path.join(out_dir, f"{arch_name}_student_best.pt")
    torch.save({
        "model_state_dict": best_student.state_dict(),
        "tau": best_tau, "alpha": best_alpha, "val_acc": val_acc,
    }, ckpt_path)

    return results


# ── heatmap ──────────────────────────────────────────────────────────────────

def plot_heatmap(results: np.ndarray, arch_name: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(results, aspect="auto", cmap="viridis",
                   vmin=results.min(), vmax=results.max())
    plt.colorbar(im, ax=ax, label="Best Val Accuracy")

    ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_yticks(range(len(TAUS)));   ax.set_yticklabels([str(t) for t in TAUS])
    ax.set_xlabel("α (distillation weight)")
    ax.set_ylabel("τ (temperature)")
    ax.set_title(f"KD Grid Search — {arch_name}")

    # annotate cells
    for i in range(len(TAUS)):
        for j in range(len(ALPHAS)):
            ax.text(j, i, f"{results[i,j]:.3f}", ha="center", va="center",
                    color="white" if results[i, j] < results.mean() else "black",
                    fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{arch_name}_kd_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Heatmap saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def load_teacher(ckpt_path: str, ModelCls, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model = ModelCls(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--ckpt_pn2",   required=True, help="PointNet++ teacher checkpoint")
    parser.add_argument("--ckpt_dgcnn", required=True, help="DGCNN teacher checkpoint")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--out_dir",    default="checkpoints/distillation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _ = get_dataloaders(
        args.csv, batch_size=args.batch, num_workers=args.workers
    )

    in_channels = 6   # XYZ + RGB

    configs = [
        ("pointnet2", args.ckpt_pn2,   PointNet2, StudentPointNet),
        ("dgcnn",     args.ckpt_dgcnn, DGCNN,     StudentDGCNN),
    ]

    for arch_name, ckpt_path, TeacherCls, StudentCls in configs:
        print(f"\n{'='*60}")
        print(f"  Architecture : {arch_name}")
        teacher, num_classes = load_teacher(ckpt_path, TeacherCls, device)
        dummy_student = StudentCls(in_channels=in_channels, num_classes=num_classes)
        print(f"  Teacher params : {count_parameters(teacher):,}")
        print(f"  Student params : {count_parameters(dummy_student):,}")
        ratio = count_parameters(teacher) / count_parameters(dummy_student)
        print(f"  Compression ratio : {ratio:.1f}×")
        assert ratio >= 4.0, f"Student must be ≥4× smaller! Got {ratio:.1f}×"
        print(f"{'='*60}")

        results = grid_search(
            arch_name, teacher, StudentCls,
            in_channels, num_classes,
            train_loader, val_loader, args, device
        )
        plot_heatmap(results, arch_name, args.out_dir)


if __name__ == "__main__":
    main()
