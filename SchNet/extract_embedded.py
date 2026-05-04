import torch

from read_multi_ase import *
from torch_geometric.data import Data
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

electronegativity_z = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    12: 1.31,  # Mg
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    20: 1.00,   #Ca
    34: 2.55,  # Se
    35: 2.96,  # Br
    53: 2.66,  # I
    76: 2.20   # Os
}


def get_max_atoms(dataset):
    length = np.zeros((len(dataset), 1))
    for idx, mol in enumerate(dataset):
        length[idx, 0] = int(len(mol[0].numbers))
    return int(np.max(length))

def mol_to_data(z, pos, y, extra_feat):
    z = torch.tensor(z, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    extra_feat = torch.tensor(extra_feat, dtype=torch.float)
    # print(extra_feat)

    return Data(z=z, pos=pos, y=y, extra_feat=extra_feat)

def get_data(dataset, features: list):

    max_atoms = get_max_atoms(dataset)
    N = len(dataset)
    N_features = len(features)

    # pre-allocate arrays
    Z = np.zeros((N, max_atoms), dtype=np.int32) # atomic numbers
    pos = np.zeros((N, max_atoms, 3), dtype=np.float32) # positions
    mask = np.zeros((N, max_atoms), dtype=bool) # for padding
    y = np.zeros(N, dtype=np.float32) # target: homo_lumo

    electronegativity = np.zeros((N, max_atoms, 1), dtype=np.float32)
    extra_feat = np.zeros((N, max_atoms, N_features), dtype=np.float32)
    for i, mol in enumerate(dataset):

        n = len(mol[0])

        Z[i, :n] = mol[0].get_atomic_numbers() # get and store atomic nums
        pos[i, :n] = mol[0].get_positions() # get and store positions
        mask[i, :n] = 1 # gets rid of padding /1 = real atom
        homo_lumo_gap = mol[0].info["homo_lumo_gap"]
        if isinstance(homo_lumo_gap, (list, tuple, np.ndarray)):
            homo_lumo_gap = np.mean(homo_lumo_gap)

        y[i] = homo_lumo_gap # get and store target

        electronegativity_i = np.array([electronegativity_z.get(int(z)) for z in Z[i, :n]])
        electronegativity[i, :n, 0] = electronegativity_i
        extra_feat[i, :n] = np.stack([np.asarray(mol[0].info[feature]) for feature in features], axis = -1)

    combined_feat = np.concatenate([extra_feat, electronegativity], axis = -1)
    # print(combined_feat.shape)
    print(np.unique(Z))

    result = [mol_to_data(Z[i][mask[i]], pos[i][mask[i]], y[i], combined_feat[i][mask[i]]) for i in range(len(Z))]
    # result = [mol_to_data(Z[i][mask[i]], pos[i][mask[i]], y[i], extra_feat[i][mask[i]]) for i in range(len(Z))]
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
    #Initialize variables to keep track of the total sum, total count, and total sum squared
    total_sum = 0
    count = 0
    sum_sq = 0

    for data in train_data:
        #Convert target data into shape (N, 1) or just one column
        y = data.y.view(-1, 1)
        #Add each target value to total sum
        total_sum += y.item()
        #Add the size of target data to count
        count += y.size(0)
        #Add the squared target value to total sum squared
        sum_sq += (y ** 2).item()

    #Calculate the mean of training data
    mean = total_sum / count
    #Std calculations based on simplified variance formula
    std = ((sum_sq / count) - mean ** 2) ** 0.5

    return mean, std

def normalize_target(data, mean: float, std: float ):
    #Iterate through each tensor object to normalize target (HOMO LUMO gap)
    for target in data:
        #Normalize the target values in place with PyTorch operations
        target.y.sub_(mean).div_(std)

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

