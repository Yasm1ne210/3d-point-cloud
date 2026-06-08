import torch
import torch.nn as nn
from config import VFL_EMBED_DIM, BATCH_SIZE, SEED
from utils import set_seed


class EntityModel(nn.Module):
    """
    Each VFL entity produces an embedding from its feature slice.
    Entity A: XYZ (channels 0:3)
    Entity B: RGB (channels 3:6)
    """
    def __init__(self, base_model, embed_dim=VFL_EMBED_DIM):
        super().__init__()
        self.base = base_model
        # Replace classifier with an embedding projector
        in_features = self._get_feature_dim()
        self.projector = nn.Linear(in_features, embed_dim)

    def _get_feature_dim(self):
        # PointNet++ and DGCNN both produce 1024-dim before classifier
        return 1024

    def forward(self, x):
        # Extract features before the final classifier
        features = self._extract_features(x)
        return self.projector(features)

    def _extract_features(self, x):
        """Forward through backbone only, skip classifier."""
        model = self.base
        # PointNet++
        if hasattr(model, 'sa1'):
            xyz = x[:, :, :3]
            features = x[:, :, 3:] if x.shape[2] > 3 else None
            xyz, features = model.sa1(xyz, features)
            xyz, features = model.sa2(xyz, features)
            xyz, features = model.sa3(xyz, features)
            return features.squeeze(1)
        # DGCNN
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
    """Fuses embeddings from both entities and classifies."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, emb_a, emb_b):
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.fusion(fused)


def run_vfl(model_a, model_b, train_loader, val_loader,
            num_classes, epochs=50, lr=1e-3, device="cpu"):
    """
    VFL training loop.
    Entity A gets XYZ (x[:,:,:3]), Entity B gets RGB (x[:,:,3:]).
    Gradients flow back to both entities through the server.
    """
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
            xyz = points[:, :, :3]
            rgb = points[:, :, 3:]

            optimizer.zero_grad()
            emb_a = entity_a(xyz)
            emb_b = entity_b(rgb)
            logits = server(emb_a, emb_b)
            loss = criterion(logits, labels)
            loss.backward()   # gradients flow back to both entities
            optimizer.step()

        acc = _evaluate_vfl(entity_a, entity_b, server, val_loader, device)
        log.append((epoch, acc))
        print(f"VFL Epoch {epoch}/{epochs} — val acc: {acc:.4f}")

    return entity_a, entity_b, server, log


def _evaluate_vfl(entity_a, entity_b, server, loader, device):
    entity_a.eval(); entity_b.eval(); server.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            xyz = points[:, :, :3]
            rgb = points[:, :, 3:]
            logits = server(entity_a(xyz), entity_b(rgb))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total