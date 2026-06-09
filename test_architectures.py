import torch
from models import PointNetPlusPlus, DGCNN
from data.dataset import get_dataloaders
from utils import count_parameters, set_seed
from config import NUM_CLASSES, IN_CHANNELS, SEED

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

train_loader, val_loader, _ = get_dataloaders(data_dir="data/raw")

for Model, name in [(PointNetPlusPlus, "PointNet++"), (DGCNN, "DGCNN")]:
    print(f"{'='*40}")
    print(f"Testing {name}")
    model = Model(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS).to(device)
    points, labels = next(iter(train_loader))
    points, labels = points.to(device), labels.to(device)
    logits = model(points)
    assert logits.shape == (points.shape[0], NUM_CLASSES), \
        f"Expected ({points.shape[0]}, {NUM_CLASSES}), got {logits.shape}"
    print(f"Output shape: {logits.shape} ✓")
    count_parameters(model)
    print()