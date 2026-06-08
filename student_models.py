"""
Person A – Lightweight Student Models (≥4× fewer parameters than teachers)

StudentPointNet  – stripped-down PointNet-style MLP (no set abstraction)
StudentDGCNN     – shallow EdgeConv with a single graph layer

Both follow the same interface as the teacher models:
    forward(x: Tensor[B, N, C]) → logits: Tensor[B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared block ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dims, bn=True, last_bn=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if bn and (last_bn or i < len(dims) - 2):
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Student A: lightweight PointNet ───────────────────────────────────────────

class StudentPointNet(nn.Module):
    """
    Tiny PointNet without hierarchical sampling.
    Teacher PointNet++ is typically ~3–4 M params; this sits under ~300 K.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 10):
        super().__init__()
        # per-point feature extraction (shared MLP via Conv1d = linear on each pt)
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),   nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64, 128, 1),            nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 256, 1),           nn.BatchNorm1d(256), nn.ReLU(True),
        )
        # global head
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x.permute(0, 2, 1)          # (B, C, N)
        x = self.feat(x)                 # (B, 256, N)
        x = x.max(dim=-1).values         # (B, 256)  global max-pool
        return self.head(x)              # (B, num_classes)


# ── Student B: lightweight DGCNN ─────────────────────────────────────────────

def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)  – use only XYZ for the graph
    returns edge_index: (B, N, k)  indices of k nearest neighbours
    """
    inner = -2 * torch.bmm(x, x.transpose(2, 1))   # (B, N, N)
    sq    = (x ** 2).sum(dim=-1, keepdim=True)       # (B, N, 1)
    dist  = sq + inner + sq.transpose(2, 1)          # (B, N, N)
    # exclude self by setting diagonal to large value
    dist  = dist + torch.eye(x.size(1), device=x.device).unsqueeze(0) * 1e9
    idx   = dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)
    return idx


def edge_conv(x: torch.Tensor, idx: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
    """
    x  : (B, N, C)
    idx: (B, N, k)
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    # gather neighbours
    idx_flat = idx.reshape(B, -1)                          # (B, N*k)
    nbr = torch.gather(x, 1, idx_flat.unsqueeze(-1).expand(B, N * k, C))
    nbr = nbr.reshape(B, N, k, C)                          # (B, N, k, C)
    xi  = x.unsqueeze(2).expand_as(nbr)                    # (B, N, k, C)
    edge_feat = torch.cat([xi, nbr - xi], dim=-1)          # (B, N, k, 2C)
    # apply shared MLP per edge, then max-pool over k
    edge_feat = edge_feat.reshape(B * N * k, 2 * C)
    out = mlp(edge_feat)                                    # (B*N*k, C_out)
    out = out.reshape(B, N, k, -1).max(dim=2).values       # (B, N, C_out)
    return out


class StudentDGCNN(nn.Module):
    """
    Single-layer EdgeConv student – roughly 1/5 the parameters of the teacher.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 10, k: int = 10):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128),             nn.BatchNorm1d(128), nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, N, C)
        idx = knn_graph(x[:, :, :3].detach(), self.k)   # graph on XYZ only
        x   = edge_conv(x, idx, self.edge_mlp)           # (B, N, 128)
        x   = x.max(dim=1).values                        # (B, 128) global pool
        return self.head(x)
