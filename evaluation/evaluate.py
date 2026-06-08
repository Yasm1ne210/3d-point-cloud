import torch
import numpy as np
import csv
import os
from config import NUM_CLASSES


def evaluate_model(model, loader, device, class_names=None):
    """
    Returns global accuracy and per-class accuracy dict.
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total   = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            preds = model(points).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    global_acc = correct / total
    per_class  = {
        (class_names[c] if class_names else c): (
            class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        )
        for c in range(NUM_CLASSES)
    }
    return global_acc, per_class


def collect_all_results(configs, loaders, device, class_names=None):
    """
    configs: list of dicts with keys: name, model, param_count
    loaders: dict with keys matching config names → DataLoader
    Returns list of result dicts for the comparison table.
    """
    results = []
    for cfg in configs:
        model      = cfg["model"]
        name       = cfg["name"]
        loader     = loaders[name]
        params     = cfg["param_count"]
        global_acc, per_class = evaluate_model(model, loader, device, class_names)
        results.append({
            "config":      name,
            "global_acc":  global_acc,
            "per_class":   per_class,
            "param_count": params
        })
        print(f"{name} — global acc: {global_acc:.4f} | params: {params:,}")
    return results


def save_results_csv(results, path="logs/comparison.csv"):
    """Save comparison table to CSV for the report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["config", "global_acc", "param_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"Saved to {path}")


def generalization_gap(model, full_loader, client_loaders, device):
    """
    Evaluate global model on classes absent from each client.
    Useful for documenting the non-IID generalization gap.
    client_loaders: list of (client_name, absent_classes, loader)
    """
    model.eval()
    model.to(device)
    gaps = []
    global_acc, _ = evaluate_model(model, full_loader, device)

    for client_name, absent_classes, loader in client_loaders:
        _, per_class = evaluate_model(model, loader, device)
        absent_acc = np.mean([
            per_class[c] for c in absent_classes if c in per_class
        ])
        gaps.append({
            "client":       client_name,
            "absent_acc":   absent_acc,
            "global_acc":   global_acc,
            "gap":          global_acc - absent_acc
        })
        print(f"{client_name} — absent class acc: {absent_acc:.4f} | gap: {global_acc - absent_acc:.4f}")

    return gaps