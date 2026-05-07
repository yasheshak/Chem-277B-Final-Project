import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import random_split
from read_multi_ase_att import *


def get_data(dataset):
    """Extract min HOMO-LUMO gap (single scalar) + spin."""
    result = []
    for i, mol in enumerate(dataset):
        try:
            atoms = mol[0]
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

            homo_lumo_gap = atoms.info["homo_lumo_gap"]
            spin = atoms.info["spin"]

            if isinstance(homo_lumo_gap, (list, tuple, np.ndarray)):
                y = torch.tensor([min(homo_lumo_gap)], dtype=torch.float)
            else:
                y = torch.tensor([homo_lumo_gap], dtype=torch.float)

            result.append(Data(z=z, pos=pos, y=y, spin=torch.tensor(spin, dtype=torch.float)))
        except Exception as e:
            print(f"Error at molecule {i}: {e}")
            continue

    print(f"Processed {len(result)} molecules")
    return result


def split_data(dataset, val_size_pct=0.2, test_size_pct=0.2):
    test_size = int(test_size_pct * len(dataset))
    temp_size = len(dataset) - test_size
    temp_train, test = random_split(dataset, [temp_size, test_size])
    val_size = int(val_size_pct * len(temp_train))
    train_size = len(temp_train) - val_size
    train, val = random_split(temp_train, [train_size, val_size])
    return train, val, test


def obtain_mean_std(train_data):
    """Mean/std from training targets."""
    ys = torch.stack([d.y for d in train_data])
    return ys.mean(), ys.std()


def normalize_target(data, mean, std):
    for d in data:
        d.y = (d.y - mean) / std
    return data
