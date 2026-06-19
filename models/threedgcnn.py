import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(xyz, k):
    inner = -2 * torch.matmul(xyz, xyz.transpose(2, 1))
    xx = torch.sum(xyz ** 2, dim=-1, keepdim=True)
    dist = xx + inner + xx.transpose(2, 1)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def gather_neighbors(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    x_flat = x.view(B * N, C)
    neighbors = x_flat[idx_flat].view(B, N, k, C)
    return neighbors


class DeformableKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, num_kernel_points=16):
        super().__init__()
        self.k = k
        self.num_kernel_points = num_kernel_points
        self.kernel_dirs = nn.Parameter(torch.randn(num_kernel_points, 3))
        self.weight = nn.Parameter(torch.randn(num_kernel_points, in_channels, out_channels) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, xyz, features):
        idx = knn_graph(xyz, self.k)
        neighbor_xyz  = gather_neighbors(xyz, idx)
        neighbor_feat = gather_neighbors(features, idx)
        rel_xyz = neighbor_xyz - xyz.unsqueeze(2)

        rel_norm = F.normalize(rel_xyz, dim=-1, eps=1e-6)
        kernel_norm = F.normalize(self.kernel_dirs, dim=-1)

        sim = torch.einsum('bnkd,md->bnkm', rel_norm, kernel_norm)
        sim = F.softmax(sim, dim=-1)

        transformed = torch.einsum('bnkc,mco->bnkmo', neighbor_feat, self.weight)
        out = torch.einsum('bnkm,bnkmo->bno', sim, transformed)
        return out + self.bias


class ThreeDGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.conv = DeformableKernelConv(in_channels, out_channels, k)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, xyz, features):
        out = self.conv(xyz, features)
        out = out.permute(0, 2, 1)
        out = self.act(self.bn(out))
        return out.permute(0, 2, 1)


class ThreeDGCN(nn.Module):
    def __init__(self, num_classes, in_channels=3, k=20):
        super().__init__()
        self.layer1 = ThreeDGCNLayer(in_channels, 64, k)
        self.layer2 = ThreeDGCNLayer(64, 128, k)
        self.layer3 = ThreeDGCNLayer(128, 256, k)
        self.layer4 = ThreeDGCNLayer(256, 512, k)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        xyz = x[:, :, :3]
        features = x
        features = self.layer1(xyz, features)
        features = self.layer2(xyz, features)
        features = self.layer3(xyz, features)
        features = self.layer4(xyz, features)
        global_feat = features.max(dim=1)[0]
        return self.classifier(global_feat)