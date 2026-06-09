# data/dataset.py
# Author: Person A
# Template — fill in the TODOs

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import SEED, NUM_POINTS, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO

# TODO: import open3d or trimesh depending on your preference
# import open3d as o3d
# import trimesh

class PointCloudDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels     = labels
        self.augment    = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        points = self._load_ply(self.file_paths[idx])  # (N, 6) XYZ+RGB
        points = self._subsample(points)                # (1024, 6)
        points = self._normalize(points)                # zero mean, unit sphere
        if self.augment:
            points = self._augment(points)
        return torch.tensor(points, dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.int64)

    def _load_ply(self, path):
        # TODO: load .ply and return (N, 6) numpy array (XYZ + RGB)
        raise NotImplementedError

    def _subsample(self, points):
        # TODO: farthest point sampling or random down to NUM_POINTS
        raise NotImplementedError

    def _normalize(self, points):
        # Center XYZ to zero mean, scale to unit sphere
        xyz = points[:, :3]
        xyz -= xyz.mean(axis=0)
        xyz /= np.max(np.linalg.norm(xyz, axis=1))
        points[:, :3] = xyz
        return points

    def _augment(self, points):
        # TODO: random rotation, jitter, random scale on XYZ only
        raise NotImplementedError


def get_dataloaders(data_dir="data/raw", csv_path="data/labeled_dataset.csv"):
    df = pd.read_csv(csv_path)
    # TODO: confirm column names with your friend
    # expected: df["file_path"], df["label"]
    file_paths = df["file_path"].values
    labels     = df["label"].values

    # Stratified split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=SEED
    )
    val_ratio_adjusted = VAL_RATIO / (1 - TRAIN_RATIO)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=SEED
    )

    train_ds = PointCloudDataset(train_files, train_labels, augment=True)
    val_ds   = PointCloudDataset(val_files,   val_labels,   augment=False)
    test_ds  = PointCloudDataset(test_files,  test_labels,  augment=False)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    )