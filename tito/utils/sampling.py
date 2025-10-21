import itertools

import mdtraj as md
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from tito import utils
from tito.data.ala2 import ALA2


def get_ala2_batch(batch_size):
    ds = ALA2()
    dl = GeometricDataLoader(ds, batch_size=batch_size)
    return next(iter(dl))["target"]


def get_cyclohexane_batch(batch_size=20):
    path = "resources/cyclohexane.pdb"
    traj = md.load(path)
    x = torch.tensor(traj.xyz, dtype=torch.float32)

    mol = Chem.MolFromPDBFile(path, removeHs=False)

    atoms = [a.element.atomic_number for a in traj.topology.atoms]
    atoms = get_atoms(mol)
    edge_index, bonds = utils.get_edge_attr_and_edge_index(mol)

    datalist = []

    for i, x in enumerate(itertools.cycle(traj.xyz)):
        datalist.append(
            Data(
                x=torch.tensor(x, dtype=torch.float32),
                atoms=torch.LongTensor(atoms),
                edge_index=edge_index,
                bonds=bonds,
            )
        )
        if i + 1 >= batch_size:
            break

    return Batch.from_data_list(datalist)


def get_atoms(mol):
    return [a.GetAtomicNum() for a in mol.GetAtoms()]
