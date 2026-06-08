# test_architectures.py
import torch
from models.pointnet2 import PointNetPlusPlus
from models.dgcnn import DGCNN
from utils import count_parameters

def test(num_classes=10):
    dummy = torch.randn(4, 1024, 3)  # replace 3 with 6 once RGB confirmed
    for Model, name in [(PointNetPlusPlus, "PointNet++"), (DGCNN, "DGCNN")]:
        model = Model(num_classes=num_classes)
        out = model(dummy)
        assert out.shape == (4, num_classes), f"{name} output shape mismatch"
        print(f"\n{name} — output shape: {out.shape}")
        count_parameters(model)

test()