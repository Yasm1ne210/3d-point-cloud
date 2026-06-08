"""
Person A – Data Pipeline
Handles: .ply loading, FPS to 1024 pts, normalization, augmentations,
stratified 70/15/15 split from labeled_dataset.csv, PyTorch Dataset/DataLoader.

DataLoader output: (points: B×1024×6 float32, labels: B int64)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── optional backends (prefer open3d, fall back to trimesh) ──────────────────
try:
    import open3d as o3d
    _BACKEND = "open3d"
except ImportError:
    import trimesh
    _BACKEND = "trimesh"

from config import SEED, NUM_POINTS, BATCH_SIZE


# ── 1. I/O helpers ───────────────────────────────────────────────────────────

def load_ply(path: str) -> np.ndarray:
    """
    Load a .ply file and return an (N, 6) array [x, y, z, r, g, b].
    RGB values are normalised to [0, 1].
    """
    if _BACKEND == "open3d":
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)          # (N, 3)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float32)      # already [0,1]
        else:
            rgb = np.zeros_like(xyz)
    else:
        mesh = trimesh.load(path, process=False)
        if hasattr(mesh, "vertices"):
            xyz = np.asarray(mesh.vertices, dtype=np.float32)
            if hasattr(mesh.visual, "vertex_colors"):
                rgb = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            else:
                rgb = np.zeros_like(xyz)
        else:  # PointCloud object
            xyz = np.asarray(mesh.vertices, dtype=np.float32)
            rgb = np.zeros_like(xyz)

    return np.concatenate([xyz, rgb], axis=1)   # (N, 6)


# ── 2. Farthest Point Sampling ───────────────────────────────────────────────

def farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Greedy FPS. Returns (n_samples, C) array.
    If the cloud has fewer points than n_samples, repeats with replacement.
    """
    N, C = points.shape
    if N == 0:
        raise ValueError("Empty point cloud.")
    if N <= n_samples:
        idx = np.random.choice(N, n_samples, replace=True)
        return points[idx]

    xyz = points[:, :3]
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)

    # start from a random point
    selected[0] = np.random.randint(0, N)
    for i in range(1, n_samples):
        last = xyz[selected[i - 1]]
        dist = np.sum((xyz - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)

    return points[selected]


# ── 3. Normalisation ─────────────────────────────────────────────────────────

def normalize(points: np.ndarray) -> np.ndarray:
    """
    Shift XYZ to zero mean, scale to unit sphere.
    RGB channels are left untouched.
    """
    pts = points.copy()
    centroid = pts[:, :3].mean(axis=0)
    pts[:, :3] -= centroid
    scale = np.max(np.linalg.norm(pts[:, :3], axis=1))
    if scale > 0:
        pts[:, :3] /= scale
    return pts


# ── 4. Augmentations (train time only) ──────────────────────────────────────

def random_rotation(points: np.ndarray) -> np.ndarray:
    """Random rotation around the gravity (Z) axis."""
    angle = np.random.uniform(0, 2 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    pts = points.copy()
    pts[:, :3] = pts[:, :3] @ R.T
    return pts


def random_jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to XYZ coordinates."""
    pts = points.copy()
    noise = np.clip(sigma * np.random.randn(*pts[:, :3].shape), -clip, clip).astype(np.float32)
    pts[:, :3] += noise
    return pts


def random_scale(points: np.ndarray, lo: float = 0.8, hi: float = 1.25) -> np.ndarray:
    """Uniform random scaling of XYZ."""
    pts = points.copy()
    scale = np.random.uniform(lo, hi)
    pts[:, :3] *= scale
    return pts


def augment(points: np.ndarray) -> np.ndarray:
    points = random_rotation(points)
    points = random_jitter(points)
    points = random_scale(points)
    return points


# ── 5. Dataset ───────────────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """
    Args:
        df         – DataFrame with columns ['filepath', 'label']
        n_points   – number of points after FPS (default NUM_POINTS=1024)
        augment    – whether to apply train-time augmentations
    """

    def __init__(self, df: pd.DataFrame, n_points: int = NUM_POINTS, augment: bool = False):
        self.records  = df[["filepath", "label"]].reset_index(drop=True)
        self.n_points = n_points
        self.augment  = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row    = self.records.iloc[idx]
        path   = row["filepath"]
        label  = int(row["label"])

        raw    = load_ply(path)                          # (N, 6)
        pts    = farthest_point_sample(raw, self.n_points)   # (1024, 6)
        pts    = normalize(pts)

        if self.augment:
            pts = augment(pts)

        pts_t  = torch.from_numpy(pts.astype(np.float32))   # (1024, 6)
        lbl_t  = torch.tensor(label, dtype=torch.long)

        return pts_t, lbl_t


# ── 6. Split & DataLoaders ───────────────────────────────────────────────────

def make_splits(csv_path: str,
                train_ratio: float = 0.70,
                val_ratio:   float = 0.15,
                seed: int = SEED):
    """
    Stratified 70 / 15 / 15 split from labeled_dataset.csv.
    Expected columns: filepath, label  (additional columns are ignored).
    Returns (train_df, val_df, test_df).
    """
    df = pd.read_csv(csv_path)

    # ---- train vs temp (30 %) ----
    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # ---- val vs test (50 / 50 of the 30 %) ----
    val_ratio_of_temp = val_ratio / (1.0 - train_ratio)  # 0.15/0.30 = 0.5
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_ratio_of_temp,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_dataloaders(csv_path: str,
                    batch_size: int  = BATCH_SIZE,
                    num_workers: int = 4,
                    seed: int        = SEED):
    """
    Returns (train_loader, val_loader, test_loader, num_classes).
    """
    train_df, val_df, test_df = make_splits(csv_path, seed=seed)

    num_classes = int(train_df["label"].nunique())

    train_ds = PointCloudDataset(train_df, augment=True)
    val_ds   = PointCloudDataset(val_df,   augment=False)
    test_ds  = PointCloudDataset(test_df,  augment=False)

    g = torch.Generator()
    g.manual_seed(seed)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=g if shuffle else None,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False), num_classes
