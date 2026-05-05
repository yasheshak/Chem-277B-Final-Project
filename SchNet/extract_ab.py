import torch

from read_multi_ase import *
from torch_geometric.data import Data
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

electronegativity_z= {
    1: 2.20,   # H
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    31: 1.81,  # Ga
    32: 2.01,  # Ge
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    37: 0.82,  # Rb
    38: 0.95,  # Sr
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.60,  # Nb
    42: 2.16,  # Mo
    43: 1.90,  # Tc (approx)
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    49: 1.78,  # In
    50: 1.96,  # Sn
    51: 2.05,  # Sb
    52: 2.10,  # Te
    53: 2.66,  # I
    55: 0.79,  # Cs
    56: 0.89,  # Ba
    57: 1.10,  # La
    58: 1.12,  # Ce
    61: None,  # Pm (no stable data)
    62: 1.17,  # Sm
    63: 1.20,  # Eu
    66: 1.22,  # Dy
    68: 1.23,  # Er
    69: 1.24,  # Tm
    70: 1.25,  # Yb
    71: 1.10,  # Lu
    72: 1.30,  # Hf
    73: 1.50,  # Ta
    74: 2.36,  # W
    75: 1.90,  # Re
    76: 2.20,  # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54,  # Au
    80: 2.00,  # Hg
    81: 1.62,  # Tl
    82: 2.33,  # Pb
    83: 2.02   # Bi
}


def get_max_atoms(dataset):
    length = np.zeros((len(dataset), 1))
    for idx, mol in enumerate(dataset):
        length[idx, 0] = int(len(mol[0].numbers))
    return int(np.max(length))

def mol_to_data(z, pos, y, extra_feat, y_mask):
    z = torch.tensor(z, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    extra_feat = torch.tensor(extra_feat, dtype=torch.float)
    y_mask = torch.tensor(y_mask, dtype=torch.bool)

    return Data(z=z, pos=pos, y=y, extra_feat=extra_feat, y_mask=y_mask)

def get_data(dataset, features: list):

    max_atoms = get_max_atoms(dataset)
    N = len(dataset)
    N_features = len(features)

    # pre-allocate arrays
    Z = np.zeros((N, max_atoms), dtype=np.int32) # atomic numbers
    pos = np.zeros((N, max_atoms, 3), dtype=np.float32) # positions
    mask = np.zeros((N, max_atoms), dtype=bool) # for padding
    y = np.zeros((N, 2), dtype=np.float32) # target: homo_lumo
    y_mask = np.zeros((N, 2), dtype=bool) #padding for target

    electronegativity = np.zeros((N, max_atoms, 1), dtype=np.float32)
    extra_feat = np.zeros((N, max_atoms, N_features), dtype=np.float32)

    for i, mol in enumerate(dataset):

        n = len(mol[0])

        Z[i, :n] = mol[0].get_atomic_numbers() # get and store atomic nums
        pos[i, :n] = mol[0].get_positions() # get and store positions
        mask[i, :n] = 1 # gets rid of padding /1 = real atom
        homo_lumo_gap = mol[0].info["homo_lumo_gap"]

        #Determine which molecules have alpha and beta homo lumo gaps based on their spin
        #Store target gaps and fill mask accordingly
        spin = mol[0].info["spin"]
        if spin > 1:
            y[i] = homo_lumo_gap
            y_mask[i] = [1, 1]
        else:
            y[i][0] = homo_lumo_gap[0]
            y_mask[i] = [1, 0]


        electronegativity_i = np.array([electronegativity_z.get(int(z)) for z in Z[i, :n]])
        electronegativity[i, :n, 0] = electronegativity_i
        extra_feat[i, :n] = np.stack([np.asarray(mol[0].info[feature]) for feature in features], axis = -1)

    combined_feat = np.concatenate([extra_feat, electronegativity], axis = -1)

    result = [mol_to_data(Z[i][mask[i]], pos[i][mask[i]], y[i], combined_feat[i][mask[i]], y_mask[i]) for i in range(len(Z))]
    print(f"Processed {len(result)} atoms")
    return result


def feature_scaler(data):

    total_features = torch.cat([mol.extra_feat for mol in data], dim = 0)

    scaler = StandardScaler()
    scaler.fit(total_features.cpu().numpy())

    return scaler


def scale_features(data, train_scaler):

    for mol in data:

        extra_feat = mol.extra_feat

        scaled_extra_feat = train_scaler.transform(extra_feat.cpu().numpy())

        mol.extra_feat = torch.tensor(scaled_extra_feat, dtype = torch.float32)

    return data


def split_data(dataset, val_size_pct: float = 0.2, test_size_pct: float = 0.2):

    test_size = int(test_size_pct * len(dataset))
    temp_train_size = len(dataset) - test_size

    temp_train_dataset, test_dataset = random_split(
        dataset, [temp_train_size, test_size])

    val_size = int(val_size_pct * len(temp_train_dataset))
    train_size = len(temp_train_dataset) - val_size

    train_dataset, val_dataset = random_split(temp_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

def obtain_mean_std(train_data):
    total_sum = 0
    count = 0
    sum_sq = 0

    for data in train_data:

        #Add each target value to total sum
        total_sum += (data.y * data.y_mask).sum(dim = 0)
        #Add the size of target data to count
        count += data.y_mask.sum(dim = 0)
        #Add the squared target value to total sum squared
        sum_sq += (data.y ** 2 * data.y_mask).sum(dim = 0)

    #Calculate the mean of training data
    mean = total_sum / count
    #Std calculations based on simplified variance formula
    std = ((sum_sq / count) - mean ** 2) ** 0.5

    return mean, std

def normalize_target(data, mean: float, std: float ):
    #Iterate through each tensor object to normalize target (HOMO LUMO gap)
    for target in data:
        #Normalize the target values in place with PyTorch operations
        target.y.sub_(mean).div_(std).mul_(target.y_mask)

    return data

"""
EXAMPLE USAGE
"""

'''
# Get files, given data directory
files_list = find_files('data')

# Process files from list and extract specified number of molecules and type
dataset = process_file(files_list, molecule_type = 'biomolecules', max_molecules = 100)

# Get final dataset for use with PyG
torch_data = get_data(dataset)

final_dataset = split_data(torch_data)
'''

