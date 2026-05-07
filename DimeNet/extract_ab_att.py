import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import random_split
from read_multi_ase_att import *


def get_data(dataset):
    """Extract alpha/beta HOMO-LUMO gaps with mask + spin. No extra features."""
    result = []
    for i, mol in enumerate(dataset):
        try:
            atoms = mol[0]
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

            homo_lumo_gap = atoms.info["homo_lumo_gap"]
            spin = atoms.info["spin"]

            y = torch.zeros(2, dtype=torch.float)
            y_mask = torch.zeros(2, dtype=torch.float)

            if spin > 1:
                y[0] = homo_lumo_gap[0]
                y[1] = homo_lumo_gap[1]
                y_mask[:] = 1
            else:
                if isinstance(homo_lumo_gap, (list, tuple, np.ndarray)):
                    y[0] = homo_lumo_gap[0]
                else:
                    y[0] = homo_lumo_gap
                y_mask[0] = 1

            result.append(Data(z=z, pos=pos, y=y, y_mask=y_mask, spin=torch.tensor(spin, dtype=torch.float)))
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
    """Masked per-channel mean/std from training targets."""
    total = torch.zeros(2)
    sq = torch.zeros(2)
    count = torch.zeros(2)
    for data in train_data:
        total += data.y * data.y_mask
        sq += data.y ** 2 * data.y_mask
        count += data.y_mask
    mean = total / count
    std = ((sq / count) - mean ** 2) ** 0.5
    return mean, std


def normalize_target(data, mean, std):
    for d in data:
        d.y = (d.y - mean) / std * d.y_mask
    return data
