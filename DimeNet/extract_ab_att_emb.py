import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import random_split
from read_multi_ase_att import *

electronegativity_z = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.60, 42: 2.16, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 55: 0.79, 56: 0.89,
    57: 1.10, 58: 1.12, 62: 1.17, 63: 1.20, 66: 1.22, 68: 1.23, 69: 1.24,
    70: 1.25, 71: 1.10, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20,
    77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02,
}


def get_data(dataset):
    """Extract alpha/beta gaps + Lowdin charges + electronegativity + spin."""
    result = []
    for i, mol in enumerate(dataset):
        try:
            atoms = mol[0]
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)

            # Lowdin charges
            lowdin = atoms.info.get("lowdin_charges", None)
            if lowdin is None:
                lowdin = np.zeros(len(atoms))
            lowdin = torch.tensor(lowdin, dtype=torch.float).unsqueeze(1)

            # Electronegativity
            en = torch.tensor(
                [electronegativity_z.get(int(a), 2.0) for a in atoms.get_atomic_numbers()],
                dtype=torch.float
            ).unsqueeze(1)

            # Combine: [n_atoms, 2]
            extra_feat = torch.cat([lowdin, en], dim=1)

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

            result.append(Data(z=z, pos=pos, y=y, y_mask=y_mask,
                              extra_feat=extra_feat, spin=torch.tensor(spin, dtype=torch.float)))
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
