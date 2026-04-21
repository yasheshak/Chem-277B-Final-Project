import torch

from read_multi_ase import * 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


def get_max_atoms(dataset): 
    length = np.zeros((len(dataset), 1))
    for idx, mol in enumerate(dataset): 
        length[idx, 0] = int(len(mol[0].numbers))
    return int(np.max(length))

def mol_to_data(z, pos, y):
    z = torch.tensor(z, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return Data(z=z, pos=pos, y=y)

def get_data(dataset): 

    max_atoms = get_max_atoms(dataset)
    N = len(dataset)

    # pre-allocate arrays 
    Z = np.zeros((N, max_atoms), dtype=np.int32) # atomic numbers 
    pos = np.zeros((N, max_atoms, 3), dtype=np.float32) # positions 
    mask = np.zeros((N, max_atoms), dtype=bool) # for padding 
    y = np.zeros(N, dtype=np.float32) # target: homo_lumo 

    for i, atom in enumerate(dataset):

        n = len(atom[0])

        Z[i, :n] = atom[0].get_atomic_numbers() # get and store atomic nums 
        pos[i, :n] = atom[0].get_positions() # get and store positions 
        mask[i, :n] = 1 # gets rid of padding /1 = real atom 
        homo_lumo_gap = atom[0].info["homo_lumo_gap"]
        if isinstance(homo_lumo_gap, (list, tuple, np.ndarray)):
            homo_lumo_gap = np.mean(homo_lumo_gap)

        y[i] = homo_lumo_gap # get and store target 
    
    result = [mol_to_data(Z[i][mask[i]], pos[i][mask[i]], y[i]) for i in range(len(Z))]
    print(f"Processed {len(result)} atoms")
    return result


def split_data(dataset, train_size_pct: float): 

    train_size = int(train_size_pct * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


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

