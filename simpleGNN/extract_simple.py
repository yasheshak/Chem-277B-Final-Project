import torch

from read_multi_ase_simple import * 
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_cluster import radius_graph


def get_max_atoms(dataset): 
    length = np.zeros((len(dataset), 1))
    for idx, mol in enumerate(dataset): 
        length[idx, 0] = int(len(mol[0].numbers))
    return int(np.max(length))

def mol_to_data(z, pos, y, cutoff = 4.0):
    z = torch.tensor(z, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # atomic number as a scalar feature
    x = z.reshape(-1, 1).float()

    # build edges from 3D coordinates
    edge_index = radius_graph(pos, r=cutoff, loop=False)

    return Data(x=x, z=z, pos=pos, edge_index=edge_index, y=y)

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


def split_data(dataset, val_size_pct: float = 0.2, test_size_pct: float = 0.2):

    test_size = int(test_size_pct * len(dataset))
    temp_train_size = len(dataset) - test_size

    temp_train_dataset, test_dataset = random_split(dataset, [temp_train_size, test_size])

    val_size = int(val_size_pct * len(temp_train_dataset))
    train_size = len(temp_train_dataset) - val_size

    train_dataset, val_dataset = random_split(temp_train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


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

final_dataset = split_data(torch_data, val_size_pct = 0.2, test_size_pct = 0.2)
'''