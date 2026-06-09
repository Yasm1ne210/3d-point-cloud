import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = -xx - inner - xx.transpose(2, 1)
    return dist.topk(k=k, dim=-1)[1]

def get_edge_features(x, k):
    B, C, N = x.shape
    idx = knn(x, k)
    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x_t = x.transpose(2, 1).contiguous().view(B * N, C)
    neighbors = x_t[idx].view(B, N, k, C)
    x_expanded = x.transpose(2, 1).unsqueeze(2).expand(B, N, k, C)
    edge_feat = torch.cat([x_expanded, neighbors - x_expanded], dim=3)
    return edge_feat.permute(0, 3, 1, 2)

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        edge_feat = get_edge_features(x, self.k)
        out = self.conv(edge_feat)
        return out.max(dim=-1)[0]

class DGCNN(nn.Module):
    def __init__(self, num_classes, k=20, in_channels=3):
        super().__init__()
        self.k = k
        self.edge1 = EdgeConv(in_channels, 64,  k)
        self.edge2 = EdgeConv(64,          64,  k)
        self.edge3 = EdgeConv(64,          128, k)
        self.edge4 = EdgeConv(128,         256, k)
        self.conv  = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        x4 = self.edge4(x3)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_cat = self.conv(x_cat)
        out = x_cat.max(dim=-1)[0]
        return self.classifier(out)