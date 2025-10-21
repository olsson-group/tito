import h5py
import numpy as np
import torch
import torch_geometric as geom
from rdkit import Chem
from tqdm import tqdm

from tito import utils
from tito.data.datasets import LaggedDatasetMixin, LazyH5DatasetMixin, StandardDatasetMixin

#  SCALING_FACTOR = 1 / 0.48458207
SCALING_FACTOR = 1 / 0.277


class TimewarpBase(LazyH5DatasetMixin):
    def __init__(
        self,
        path,
        sub_data_set="large",
        split=None,
        normalize=False,
        lazy_load=False,
        protein=None,
    ):
        self.path = path
        self.sub_data_set = sub_data_set
        self.split = "train" if split is None else split

        self.h5file = h5py.File(self.path, "r")
        self.normalize = normalize
        self.bond_index = {}
        self.bonds = {}
        self.atoms = {}
        self.molecule_idxs = []

        self.mol_suppl = []
        self.traj_lens = []
        self.traj_names = []
        self.lags = []

        for molecule_idx, (name, group) in enumerate(tqdm(self.h5file[self.split].items())):
            if protein and protein != name:
                continue

            mol = Chem.MolFromMolBlock(group["mol"][()], sanitize=True, removeHs=False)
            self.mol_suppl.append(mol)
            bond_index, bond_type = utils.get_bonds_from_rdkit(mol)
            atoms = utils.get_atoms_from_rdkit(mol)

            self.bond_index[molecule_idx] = torch.tensor(bond_index)
            self.bonds[molecule_idx] = torch.tensor(bond_type)
            self.atoms[molecule_idx] = torch.tensor(atoms)
            self.traj_names.append(name)
            traj = group["traj"][()]

            self.molecule_idxs.append(molecule_idx)
            self.traj_lens.append(len(traj))
            self.lags.append(5.0)

        self.scaling_factor = SCALING_FACTOR  # type:ignore

        self.traj_boundaries = np.append([0], np.cumsum(self.traj_lens))
        LazyH5DatasetMixin.__init__(self, path=self.path, lazy_load=lazy_load)

    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, index):
        LazyH5DatasetMixin.lazy_load(self)

        traj_idx = np.searchsorted(self.traj_boundaries, index, "right") - 1
        molecule_idx = self.molecule_idxs[traj_idx]
        config_idx = index - self.traj_boundaries[traj_idx]
        traj_name = self.traj_names[traj_idx]
        config = self.h5file[self.split][traj_name]["traj"][config_idx]  # type:ignore

        x = torch.tensor(config)
        x = utils.center_coordinates(x)

        if self.normalize:
            x *= self.scaling_factor

        data = geom.data.Data(
            x=x,
            node_type=self.atoms[molecule_idx],
            bond_index=self.bond_index[molecule_idx],
            bond_type=self.bonds[molecule_idx],
            index=torch.tensor([index]),
        )
        return data

    def get_lag(self, molecule_idx):
        return self.lags[molecule_idx]
    
    def get_traj(self, mol_idx):
        """
        Returns the trajectory for a given molecule index.
        """
        traj_name = self.traj_names[mol_idx]
        traj = self.h5file[self.split][traj_name]["traj"][()]
        if self.normalize:
            traj *= self.scaling_factor
        return torch.tensor(traj, dtype=torch.float32)


class Timewarp(StandardDatasetMixin, TimewarpBase):
    def __init__(self, path=None, normalize=False, split=None, protein=None, **kwargs):
        TimewarpBase.__init__(self, path=path, normalize=normalize, split=split, protein=protein)
        StandardDatasetMixin.__init__(self, kwargs)


class LaggedTimewarp(LaggedDatasetMixin, TimewarpBase):
    def __init__(
        self,
        path=None,
        sub_data_set="large",
        max_lag=5,
        fixed_lag=False,
        split=None,
        normalize=False,
        lazy_load=False,
        transform=None,
        protein=None,
        ot_coupling=True,
        **kwargs,
    ):
        TimewarpBase.__init__(
            self, path=path, sub_data_set=sub_data_set, split=split, normalize=normalize, lazy_load=lazy_load, protein=protein
        )
        self.tau = 5.0  # time in ps
        max_lag = int(max_lag / 5)  # turn a physical lag into a discrete lag
        LaggedDatasetMixin.__init__(self, max_lag=max_lag, fixed_lag=fixed_lag, 
            transform=transform, ot_coupling=ot_coupling, **kwargs)

    def __getitem__(self, idx):
        item = LaggedDatasetMixin.__getitem__(self, idx)
        lag = self.tau * item["lag"]
        item["lag"] = lag

        return item
