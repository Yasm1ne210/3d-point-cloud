import torch
from models import PointNetPlusPlus, DGCNN
from data.dataset import get_dataloaders
from federation.vfl import run_vfl
from utils import set_seed
from config import SEED, NUM_CLASSES, IN_CHANNELS
import csv, os

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader = get_dataloaders(data_dir="data/raw")

def save_log(log, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_acc"])
        writer.writerows(log)

for Model, name in [(PointNetPlusPlus, "pointnet2"), (DGCNN, "dgcnn")]:
    print(f"\n{'='*40}")
    print(f"VFL — {name}")
    model_a = Model(num_classes=NUM_CLASSES, in_channels=3)
    model_b = Model(num_classes=NUM_CLASSES, in_channels=3)
    entity_a, entity_b, server, log = run_vfl(
        model_a=model_a,
        model_b=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        epochs=50,
        device=device
    )
    torch.save({
        "entity_a": entity_a.state_dict(),
        "entity_b": entity_b.state_dict(),
        "server":   server.state_dict()
    }, f"checkpoints/vfl_{name}.pth")
    save_log(log, f"logs/vfl_{name}.csv")
    print(f"Saved VFL checkpoint and log for {name}")