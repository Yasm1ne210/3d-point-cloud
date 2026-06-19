import torch
from models import PointNetPlusPlus, DGCNN, PointGNN, ThreeDGCN
from data.dataset import get_dataloaders
from evaluation.evaluate import collect_all_results, save_results_csv
from utils import count_parameters, set_seed
from config import SEED, NUM_CLASSES, IN_CHANNELS

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, test_loader = get_dataloaders(data_dir="data/raw")

def load_model(Model, path):
    model = Model(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

ModelMap = {
    "pointnet2": PointNetPlusPlus,
    "dgcnn": DGCNN,
    "pointgnn": PointGNN,
    "threedgcn": ThreeDGCN,
}

checkpoints = {
    "C1_baseline_pointnet2":    ("pointnet2", "checkpoints/baseline_pointnet2.pth"),
    "C1_baseline_dgcnn":        ("dgcnn",     "checkpoints/baseline_dgcnn.pth"),
    "C1_baseline_pointgnn":     ("pointgnn",  "checkpoints/baseline_pointgnn.pth"),
    "C1_baseline_threedgcn":    ("threedgcn", "checkpoints/baseline_threedgcn.pth"),
    "C2_hfl_iid_pointnet2":     ("pointnet2", "checkpoints/hfl_iid_pointnet2.pth"),
    "C2_hfl_iid_dgcnn":         ("dgcnn",     "checkpoints/hfl_iid_dgcnn.pth"),
    "C2_hfl_iid_pointgnn":      ("pointgnn",  "checkpoints/hfl_iid_pointgnn.pth"),
    "C2_hfl_iid_threedgcn":     ("threedgcn", "checkpoints/hfl_iid_threedgcn.pth"),
    "C3_hfl_noniid_pointnet2":  ("pointnet2", "checkpoints/hfl_noniid_pointnet2.pth"),
    "C3_hfl_noniid_dgcnn":      ("dgcnn",     "checkpoints/hfl_noniid_dgcnn.pth"),
    "C3_hfl_noniid_pointgnn":   ("pointgnn",  "checkpoints/hfl_noniid_pointgnn.pth"),
    "C3_hfl_noniid_threedgcn":  ("threedgcn", "checkpoints/hfl_noniid_threedgcn.pth"),
    "C4_distill_pointnet2":     ("pointnet2", "checkpoints/student_pointnet2.pth"),
    "C4_distill_dgcnn":         ("dgcnn",     "checkpoints/student_dgcnn.pth"),
    "C4_distill_pointgnn":      ("pointgnn",  "checkpoints/student_pointgnn.pth"),
    "C4_distill_threedgcn":     ("threedgcn", "checkpoints/student_threedgcn.pth"),
}

configs = []
loaders = {}
for config_name, (arch, ckpt_path) in checkpoints.items():
    model = load_model(ModelMap[arch], ckpt_path)
    total, _ = count_parameters(model)
    configs.append({"name": config_name, "model": model, "param_count": total})
    loaders[config_name] = test_loader

results = collect_all_results(configs, loaders, device)
save_results_csv(results, path="logs/comparison.csv")