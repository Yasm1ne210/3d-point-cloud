# data/dummy_dataset.py
# Temporary stand-in until Person A finishes dataset.py
# Drop-in replacement — same interface, same output shapes

import torch
from torch.utils.data import DataLoader, TensorDataset
from config import NUM_POINTS, IN_CHANNELS, BATCH_SIZE, NUM_CLASSES, SEED
from utils import set_seed

def get_dummy_dataloaders(n_train=640, n_val=160, n_test=160):
    set_seed(SEED)

    def make_loader(n, shuffle):
        points = torch.randn(n, NUM_POINTS, IN_CHANNELS)
        labels = torch.randint(0, NUM_CLASSES, (n,))
        return DataLoader(TensorDataset(points, labels),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        make_loader(n_train, shuffle=True),
        make_loader(n_val,   shuffle=False),
        make_loader(n_test,  shuffle=False)
    )