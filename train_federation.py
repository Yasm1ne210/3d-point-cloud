import torch
from models import PointNetPlusPlus, DGCNN
from data.dataset import get_dataloaders
from federation.hfl import run_hfl
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
        writer.writerow(["round", "val_acc"])
        writer.writerows(log)

for Model, name in [(PointNetPlusPlus, "pointnet2"), (DGCNN, "dgcnn")]:
    for iid, label in [(True, "iid"), (False, "noniid")]:
        print(f"\n{'='*40}")
        print(f"HFL {label.upper()} — {name}")
        model_fn = lambda: Model(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
        global_model = model_fn()
        trained_model, log = run_hfl(
            global_model=global_model,
            dataset=train_loader.dataset,
            model_fn=model_fn,
            n_rounds=50,
            local_epochs=5,
            iid=iid,
            device=device,
            val_loader=val_loader
        )
        torch.save(trained_model.state_dict(), f"checkpoints/hfl_{label}_{name}.pth")
        save_log(log, f"logs/hfl_{label}_{name}.csv")
        print(f"Saved checkpoint and log for HFL {label} {name}")