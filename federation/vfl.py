import torch
import torch.nn as nn
from config import VFL_EMBED_DIM, SEED
from utils import set_seed

class EntityModel(nn.Module):
    def __init__(self, base_model, embed_dim=VFL_EMBED_DIM):
        super().__init__()
        self.base = base_model
        self.projector = nn.Linear(1024, embed_dim)

    def forward(self, x):
        return self.projector(self._extract_features(x))

    def _extract_features(self, x):
        model = self.base
        if hasattr(model, 'sa1'):
            xyz = x[:, :, :3]
            features = x[:, :, 3:] if x.shape[2] > 3 else None
            xyz, features = model.sa1(xyz, features)
            xyz, features = model.sa2(xyz, features)
            xyz, features = model.sa3(xyz, features)
            return features.squeeze(1)
        elif hasattr(model, 'edge1'):
            x = x.permute(0, 2, 1)
            x1 = model.edge1(x)
            x2 = model.edge2(x1)
            x3 = model.edge3(x2)
            x4 = model.edge4(x3)
            x_cat = torch.cat([x1, x2, x3, x4], dim=1)
            x_cat = model.conv(x_cat)
            return x_cat.max(dim=-1)[0]
        else:
            raise ValueError("Unknown backbone architecture")

class VFLServer(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, emb_a, emb_b):
        return self.fusion(torch.cat([emb_a, emb_b], dim=1))

def _evaluate_vfl(entity_a, entity_b, server, loader, device):
    entity_a.eval(); entity_b.eval(); server.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            logits = server(entity_a(points[:, :, :3]), entity_b(points[:, :, 3:]))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def run_vfl(model_a, model_b, train_loader, val_loader,
            num_classes, epochs=50, lr=1e-3, device="cpu"):
    set_seed(SEED)
    entity_a = EntityModel(model_a).to(device)
    entity_b = EntityModel(model_b).to(device)
    server   = VFLServer(VFL_EMBED_DIM, num_classes).to(device)
    params = (list(entity_a.parameters()) +
              list(entity_b.parameters()) +
              list(server.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    log = []
    for epoch in range(1, epochs + 1):
        entity_a.train(); entity_b.train(); server.train()
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = server(entity_a(points[:, :, :3]), entity_b(points[:, :, 3:]))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        acc = _evaluate_vfl(entity_a, entity_b, server, val_loader, device)
        log.append((epoch, acc))
        print(f"VFL Epoch {epoch}/{epochs} — val acc: {acc:.4f}")
    return entity_a, entity_b, server, log