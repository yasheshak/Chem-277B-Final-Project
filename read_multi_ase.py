import glob
from importlib.metadata import files
from typing import Union

import numpy as np

from fairchem.core.datasets import AseDBDataset


def read_aselmdb(file_path: Union[str, list[str]] = './data'):
    """
    Reads one or more .aselmdb files.
    Returns a MultiAseDBDataset regardless — single path is just a list of one.
    """
    if isinstance(file_path, str):
        file_path = [file_path]
    return MultiAseDBDataset(file_path)


def get_molecules_by_type(dataset, mol_type='biomolecules'):
    n = len(dataset)
    atoms = np.array([dataset.get_atoms(i) for i in range(n)], dtype=object)

    if mol_type is None:
        return atoms.reshape(-1, 1)

    mask = np.fromiter((atom.info.get('data_id') == mol_type for atom in atoms),\
        dtype=bool, count=n)
    return atoms[mask].reshape(-1, 1)


def find_files(directory: str = 'data', ext: str = '.aselmdb'):
    return glob.glob(f"{directory}/*{ext}")


def process_file(file: Union[str, list[str]]):
    data = read_aselmdb(file)
    return get_molecules_by_type(data)


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

# Single or multiple files; same interface either way
single = read_aselmdb(files_list[0])
multi  = read_aselmdb(files_list)

# Full AseDBDataset API available on both single and multi file 
print(len(multi))
atoms_0 = multi.get_atoms(0)
sample  = multi[0]

# Filter by molecule type across all files
processed = process_file(files_list)
'''

if __name__ == "__main__": 

    files_list = find_files('data')

    # change for max number of files to process 
    max = 2
    processed = process_file(files_list[:2])
    print(f"{len(processed)} molecules were processed.")