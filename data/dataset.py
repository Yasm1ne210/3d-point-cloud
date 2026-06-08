"""
data/dataset.py  —  Person A's data pipeline
Agreed interface with Person B:

    from data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(data_dir="data/raw")

Each loader yields batches of:
    points : (B, 1024, 6)  float32   [XYZ + RGB, normalised]
    labels : (B,)          int64

Extra helpers exported for Person A's own training scripts:
    get_dataloaders_with_num_classes(data_dir) → (train, val, test, num_classes)
    NUM_CLASSES  – set after the first call to either function above
"""

import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── optional I/O backends ─────────────────────────────────────────────────────
try:
    import open3d as o3d
    _BACKEND = "open3d"
except ImportError:
    import trimesh
    _BACKEND = "trimesh"

from config import BATCH_SIZE, NUM_POINTS, SEED

# populated on first call — Person A's training scripts can read this
NUM_CLASSES: int = 0


# ── 1. PLY loading ────────────────────────────────────────────────────────────

def load_ply(path: str) -> np.ndarray:
    """Return (N, 6) float32 array  [x, y, z, r, g, b],  RGB in [0, 1]."""
    if _BACKEND == "open3d":
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() \
              else np.zeros_like(xyz)
    else:
        obj = trimesh.load(path, process=False)
        xyz = np.asarray(obj.vertices, dtype=np.float32)
        if hasattr(obj, "visual") and hasattr(obj.visual, "vertex_colors"):
            rgb = obj.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            rgb = np.zeros_like(xyz)

    return np.concatenate([xyz, rgb], axis=1)   # (N, 6)


# ── 2. Farthest Point Sampling ────────────────────────────────────────────────

def farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Greedy FPS → (n_samples, C).  Up-samples with replacement if N < n_samples."""
    N = len(points)
    if N == 0:
        raise ValueError(f"Empty point cloud at index.")
    if N <= n_samples:
        return points[np.random.choice(N, n_samples, replace=True)]

    xyz      = points[:, :3]
    selected = np.empty(n_samples, dtype=np.int64)
    dists    = np.full(N, np.inf, dtype=np.float32)

    selected[0] = np.random.randint(N)
    for i in range(1, n_samples):
        d = np.sum((xyz - xyz[selected[i - 1]]) ** 2, axis=1)
        dists      = np.minimum(dists, d)
        selected[i] = np.argmax(dists)

    return points[selected]


# ── 3. Normalisation ──────────────────────────────────────────────────────────

def normalize(points: np.ndarray) -> np.ndarray:
    """Zero-mean + unit-sphere on XYZ; RGB left unchanged."""
    pts = points.copy()
    pts[:, :3] -= pts[:, :3].mean(axis=0)
    r = np.linalg.norm(pts[:, :3], axis=1).max()
    if r > 0:
        pts[:, :3] /= r
    return pts


# ── 4. Augmentations (train time only) ───────────────────────────────────────

def _augment(points: np.ndarray) -> np.ndarray:
    # random rotation around Z (gravity) axis
    a = np.random.uniform(0, 2 * math.pi)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    points = points.copy()
    points[:, :3] = points[:, :3] @ R.T
    # jitter
    points[:, :3] += np.clip(0.01 * np.random.randn(len(points), 3), -0.05, 0.05).astype(np.float32)
    # scale
    points[:, :3] *= np.random.uniform(0.8, 1.25)
    return points


# ── 5. Dataset ────────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_points: int = NUM_POINTS, train: bool = False):
        self.records  = df[["filepath", "label"]].reset_index(drop=True)
        self.n_points = n_points
        self.train    = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        pts   = load_ply(row["filepath"])
        pts   = farthest_point_sample(pts, self.n_points)
        pts   = normalize(pts)
        if self.train:
            pts = _augment(pts)
        return (
            torch.from_numpy(pts.astype(np.float32)),      # (1024, 6)
            torch.tensor(int(row["label"]), dtype=torch.long),
        )


# ── 6. Split helper ───────────────────────────────────────────────────────────

def _make_splits(csv_path: str, seed: int = SEED):
    df = pd.read_csv(csv_path)
    train_df, tmp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.50, stratify=tmp["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _build_loader(ds, shuffle: bool, batch_size: int, num_workers: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g if shuffle else None,
    )


# ── 7. Public API ─────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:    str = "data/raw",
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
    seed:        int = SEED,
):
    """
    Agreed interface — what Person B imports:

        from data.dataset import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(data_dir="data/raw")

    Discovers labeled_dataset.csv inside data_dir automatically.
    Also sets the module-level NUM_CLASSES for Person A's own scripts.
    """
    global NUM_CLASSES

    csv_path = os.path.join(data_dir, "labeled_dataset.csv")
    if not os.path.exists(csv_path):
        # allow csv sitting one level up  (data/labeled_dataset.csv)
        alt = os.path.join(os.path.dirname(data_dir), "labeled_dataset.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(
                f"labeled_dataset.csv not found in {data_dir!r} or {os.path.dirname(data_dir)!r}"
            )

    train_df, val_df, test_df = _make_splits(csv_path, seed=seed)
    NUM_CLASSES = int(train_df["label"].nunique())

    train_loader = _build_loader(PointCloudDataset(train_df, train=True),  True,  batch_size, num_workers, seed)
    val_loader   = _build_loader(PointCloudDataset(val_df,   train=False), False, batch_size, num_workers, seed)
    test_loader  = _build_loader(PointCloudDataset(test_df,  train=False), False, batch_size, num_workers, seed)

    return train_loader, val_loader, test_loader


def get_dataloaders_with_num_classes(
    data_dir:    str = "data/raw",
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
    seed:        int = SEED,
):
    """
    Variant for Person A's own training scripts that also need num_classes:

        train_loader, val_loader, test_loader, num_classes = \\
            get_dataloaders_with_num_classes(data_dir="data/raw")
    """
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size, num_workers, seed
    )
    return train_loader, val_loader, test_loader, NUM_CLASSES
