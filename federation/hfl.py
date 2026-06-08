import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from config import SEED, BATCH_SIZE
from utils import set_seed
from .fedavg import FedAvgServer


class FederatedClient:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def local_train(self, epochs, lr=1e-3):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for points, labels in self.dataloader:
                points, labels = points.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(points)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()


def make_iid_clients(dataset, model_fn, n_clients, device):
    """Randomly shuffle data across n_clients (IID control condition)."""
    set_seed(SEED)
    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, n_clients)
    clients = []
    for split in splits:
        loader = DataLoader(Subset(dataset, split), batch_size=BATCH_SIZE, shuffle=True)
        clients.append(FederatedClient(copy.deepcopy(model_fn()), loader, device))
    return clients


def make_noniid_clients(dataset, model_fn, n_clients, device):
    """
    Semantic (non-IID) partition: each client gets data from a subset of classes.
    4-client partition from project spec:
      client 0 → classes 0,1,2
      client 1 → classes 2,3,4
      client 2 → classes 4,5,6
      client 3 → classes 6,7,8,9
    Overlapping classes ensure no class is fully absent globally.
    """
    client_classes = [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8, 9]
    ]
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    clients = []
    for classes in client_classes:
        indices = np.where(np.isin(labels, classes))[0]
        loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
        clients.append(FederatedClient(copy.deepcopy(model_fn()), loader, device))
    return clients


def run_hfl(global_model, dataset, model_fn, n_rounds, local_epochs,
            iid=True, device="cpu", val_loader=None):
    """
    Full HFL training loop.
    Returns list of (round, val_accuracy) tuples for logging.
    """
    server = FedAvgServer(global_model)
    make_clients = make_iid_clients if iid else make_noniid_clients
    clients = make_clients(dataset, model_fn, n_clients=4, device=device)
    client_sizes = [len(c.dataloader.dataset) for c in clients]

    log = []
    for r in range(1, n_rounds + 1):
        server.broadcast(clients)
        for client in clients:
            client.local_train(epochs=local_epochs)
        server.aggregate(clients, client_sizes)

        if val_loader is not None:
            acc = evaluate_model(server.get_global_model(), val_loader, device)
            log.append((r, acc))
            print(f"Round {r}/{n_rounds} — val acc: {acc:.4f}")

    return server.get_global_model(), log


def evaluate_model(model, loader, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            preds = model(points).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total