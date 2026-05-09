import glob
import numpy as np

from typing import Union
from fairchem.core.datasets import AseDBDataset


def find_files(directory: str = 'data', ext: str = '.aselmdb') -> list[str]:
    return glob.glob(f"{directory}/*{ext}")


def read_aselmdb(file_path: Union[str, list[str]] = './data'):
    """
    Reads one or more .aselmdb files.
    """
    if isinstance(file_path, str):
        file_path = [file_path]
    return MultiAseDBDataset(file_path)


def get_molecules_by_type(dataset, mol_type=['biomolecules', 'metal_complexes', 'elytes']):
    n = len(dataset)
    atoms = np.array([dataset.get_atoms(i) for i in range(n)], dtype=object)

    if mol_type is None:
        return atoms.reshape(-1, 1)

    mask = np.fromiter((atom.info.get('data_id') not in mol_type for atom in atoms),\
        dtype=bool, count=n)
    return atoms[mask].reshape(-1, 1)


def process_file(file: Union[str, list[str]], 
                molecule_type: str = 'biomolecules', 
                max_molecules: int = 1000):
    
    """
    Reads one or more .aselmdb files and returns filtered molecules.
    """

    # read in dataset from file names 
    dataset = read_aselmdb(file) 

    # total num of mols in entire dataset 
    n_total = len(dataset) 

    # define array and pre-allocate 
    alloc_size = max_molecules if max_molecules is not None else n_total 
    result = np.zeros((alloc_size, 1), dtype=object)

    count = 0 # for tracking 

    for i in range(n_total):

        if count >= alloc_size:
            break

        atoms = dataset.get_atoms(i) 

        if molecule_type is None or atoms.info.get('data_id') in molecule_type:
            result[count, 0] = atoms
            count += 1

    # cut any unfilled rows if fewer matches than alloc_size
    return result[:count]


class MultiAseDBDataset(AseDBDataset):
    """
    Extends AseDBDataset to support multiple .aselmdb files.
    """

    def __init__(self, file_paths: list[str]):
        super().__init__({"src": file_paths})


"""
EXAMPLE USAGE
"""
 
'''
files_list = find_files('data')
 
# Single or multiple files — same interface either way
single_result = process_file(files_list[0], max_molecules=500)
multi_result  = process_file(files_list, max_molecules=1000)
 
# No cap — returns all matching molecules
all_result = process_file(files_list, max_molecules=None)
 
# Accept every molecule regardless of type
all_types = process_file(files_list, molecule_type=None, max_molecules=200)
 
print(f"Collected {len(single_result)} molecules from a single file.")
print(f"Collected {len(multi_result)} molecules across all files.")
'''
