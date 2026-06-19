import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(xyz, k):
    """Build k-NN graph indices based on XYZ coordinates.
    xyz: (B, N, 3) -> returns idx: (B, N, k)
    """
    inner = -2 * torch.matmul(xyz, xyz.transpose(2, 1))
    xx = torch.sum(xyz ** 2, dim=-1, keepdim=True)
    dist = xx + inner + xx.transpose(2, 1)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def gather_neighbors(x, idx):
    """x: (B, N, C), idx: (B, N, k) -> (B, N, k, C)"""
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    x_flat = x.view(B * N, C)
    neighbors = x_flat[idx_flat].view(B, N, k, C)
    return neighbors


class PointGNNLayer(nn.Module):
    def __init__(self, state_dim, k):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(state_dim + 3, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, xyz, state):
        idx = knn_graph(xyz, self.k)
        neighbor_state = gather_neighbors(state, idx)
        neighbor_xyz   = gather_neighbors(xyz, idx)
        rel_xyz = neighbor_xyz - xyz.unsqueeze(2)

        edge_input = torch.cat([neighbor_state, rel_xyz], dim=-1)
        messages = self.edge_mlp(edge_input)
        agg = messages.max(dim=2)[0]

        updated = self.update_mlp(torch.cat([state, agg], dim=-1))
        return self.norm(state + updated)


class PointGNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, state_dim=128, k=20, num_layers=3):
        super().__init__()
        self.k = k
        self.embed = nn.Sequential(
            nn.Linear(in_channels, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        self.layers = nn.ModuleList([
            PointGNNLayer(state_dim, k) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        xyz = x[:, :, :3]
        state = self.embed(x)
        for layer in self.layers:
            state = layer(xyz, state)
        global_feat = state.max(dim=1)[0]
        return self.classifier(global_feat)