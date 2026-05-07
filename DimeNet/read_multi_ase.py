import glob
import numpy as np
from typing import Union
from fairchem.core.datasets import AseDBDataset


def find_files(directory: str = 'data', ext: str = '.aselmdb') -> list[str]:
    return glob.glob(f"{directory}/*{ext}")


def read_aselmdb(file_path: Union[str, list[str]] = './data'):
    if isinstance(file_path, str):
        file_path = [file_path]
    return MultiAseDBDataset(file_path)


def process_file(file: Union[str, list[str]], molecule_type: str = 'biomolecules', max_molecules: int = 1000):
    dataset = read_aselmdb(file)
    n_total = len(dataset)
    alloc_size = max_molecules if max_molecules is not None else n_total
    result = np.zeros((alloc_size, 1), dtype=object)
    count = 0

    for i in range(n_total):
        if count >= alloc_size:
            break
        atoms = dataset.get_atoms(i)
        if molecule_type is None or atoms.info.get('data_id') in molecule_type:
            result[count, 0] = atoms
            count += 1

    return result[:count]


class MultiAseDBDataset(AseDBDataset):
    def __init__(self, file_paths: list[str]):
        super().__init__({"src": file_paths})
