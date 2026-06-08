# models/pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sampling(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def ball_query(xyz, new_xyz, radius, max_sample):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    idx = torch.zeros(B, S, max_sample, dtype=torch.long, device=xyz.device)
    for b in range(B):
        for s in range(S):
            dists = torch.sum((xyz[b] - new_xyz[b, s]) ** 2, dim=-1)
            neighbors = (dists <= radius ** 2).nonzero(as_tuple=False).squeeze(-1)
            if neighbors.shape[0] == 0:
                idx[b, s] = 0
            elif neighbors.shape[0] >= max_sample:
                idx[b, s] = neighbors[:max_sample]
            else:
                pad = neighbors[torch.randint(0, neighbors.shape[0], (max_sample - neighbors.shape[0],))]
                idx[b, s] = torch.cat([neighbors, pad])
    return idx

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, max_sample, in_channels, mlp_dims):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.max_sample = max_sample
        layers = []
        last_dim = in_channels + 3  # +3 for XYZ concatenation
        for out_dim in mlp_dims:
            layers += [nn.Conv2d(last_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU()]
            last_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        # xyz: (B, N, 3), features: (B, N, C) or None
        fps_idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)                         # (B, npoint, 3)
        idx = ball_query(xyz, new_xyz, self.radius, self.max_sample) # (B, npoint, max_sample)
        grouped_xyz = index_points(xyz, idx)                         # (B, npoint, max_sample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)                          # relative coords

        if features is not None:
            grouped_feat = index_points(features, idx)               # (B, npoint, max_sample, C)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1) # (B, npoint, max_sample, C+3)
        else:
            grouped = grouped_xyz

        grouped = grouped.permute(0, 3, 2, 1)  # (B, C+3, max_sample, npoint)
        grouped = self.mlp(grouped)
        new_feat = grouped.max(dim=2)[0]        # (B, out_dim, npoint)
        new_feat = new_feat.permute(0, 2, 1)    # (B, npoint, out_dim)
        return new_xyz, new_feat


class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        # 3 set abstraction levels
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channels - 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(1,   1.0, 128, 256, [256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, N, C) where C >= 3
        xyz = x[:, :, :3]
        features = x[:, :, 3:] if x.shape[2] > 3 else None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)
        out = features.squeeze(1)    # (B, 1024)
        return self.classifier(out)  # (B, num_classes)