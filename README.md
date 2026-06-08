# 3D Point Cloud Classification

PointNet++ and DGCNN trained under centralized, federated, and knowledge distillation paradigms.

## Project Structure

- `models/` — PointNet++ and DGCNN architectures (Person B)
- `data/` — DataLoader and preprocessing (Person A)
- `federation/` — FedAvg, HFL, VFL (Person B)
- `distillation/` — Knowledge distillation (Person A)
- `evaluation/` — Metrics and comparison (Person B)

## Setup

```bash
pip install torch open3d trimesh numpy pandas
```

## Shared Constants

All constants live in `config.py`. Never hardcode them elsewhere.

## Branching

- `main` — stable only
- `dev` — integration branch
- `person-a/` — Person A's branches
- `person-b/` — Person B's branches
