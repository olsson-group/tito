import os
import tempfile

import h5py
import mdtraj as md
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data as GeometricData
from tqdm import tqdm

from tito import utils
from tito.data.datasets import LaggedDatasetMixin, LazyH5DatasetMixin, StandardDatasetMixin

# SCALING_FACTOR = 1 / 0.1504218429327011
SCALING_FACTOR = 1 / 0.20754094


class MDQM9Base(LazyH5DatasetMixin):
    scaling_factor = SCALING_FACTOR

    def __init__(self, path, sub_data_set="version_0", sub_sampling_indices_path=None, split=None, normalize=True, lazy_load=False):
        self.path = path
        self.h5_path = path + "mdqm9-nc.hdf5"
        self.sdf_file = path + "mdqm9-nc.sdf"
        self.split = split
        mol_suppl = Chem.SDMolSupplier(self.sdf_file, removeHs=False)

        h5file = h5py.File(self.h5_path, "r")
        self.normalize = normalize
        if split in ["train", "val", "test"]:
            print(f"Using {split} split ...")
            if sub_sampling_indices_path is not None:
                split_path = sub_sampling_indices_path
            else:
                split_path = self.path+f"splits/{split}_indices.npy"
            molecule_idxs = np.load(split_path)

        elif split == "mini":
            molecule_idxs = [621, 684, 724]

        elif split in [str(i) for i in range(1, 10)]:
            split_path = self.path+f"splits/{split}_ha.npy"
            with open(split_path, "rb") as f:
                molecule_idxs = np.load(f)

        elif isinstance(split, list) or isinstance(split, np.ndarray):
            molecule_idxs = split

        else:
            molecule_idxs = h5file.keys()

        molecule_idxs = [str(idx).zfill(5) for idx in molecule_idxs]

        if unavailable := set(molecule_idxs) - set(h5file.keys()):
            raise ValueError(f"Molecules {unavailable} not in dataset.")

        traj_lens = []
        self.molecule_idxs = []
        self.trajs = []
        self.traj_names = []
        self.atoms = {}
        self.lags = {}
        self.n_heavy = {}
        self.bonds = {}
        self.bond_index = {}
        self.mol_suppl = [] #change name here and for timewarp

        print("Loading data ...")
        for molecule_idx, molecule in enumerate(tqdm(molecule_idxs)):
            atoms = h5file[molecule]["data"]["atoms"][()]  # type: ignore
            lags = h5file[molecule]["data"]["time_lag"][()]  # type: ignore
            #partial_charges = h5file[molecule]["data"]["partial_charges"][()]  # type: ignore

            mol = mol_suppl[int(molecule)]
            bond_index, bonds = utils.get_bond_index_and_bonds(mol)

            self.bond_index[molecule_idx] = bond_index
            self.bonds[molecule_idx] = bonds
            self.atoms[molecule_idx] = atoms
            self.lags[molecule_idx] = lags

            traj = h5file[molecule]["trajectories"]["md_0"]  # type: ignore
            self.molecule_idxs.append(molecule_idx)
            traj_lens.append(len(traj))  # type: ignore
            self.traj_names.append(traj.name)
            self.mol_suppl.append(mol)

        self.traj_boundaries = np.append([0], np.cumsum(traj_lens))

        LazyH5DatasetMixin.__init__(self, path=self.h5_path, lazy_load=lazy_load)

    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, index):
        LazyH5DatasetMixin.lazy_load(self)

        traj_idx = np.searchsorted(self.traj_boundaries, index, "right") - 1
        molecule_idx = self.molecule_idxs[traj_idx]
        config_idx = index - self.traj_boundaries[traj_idx]
        traj_name = self.traj_names[traj_idx]
        atoms = np.array(self.atoms[molecule_idx])
        lag = self.lags[molecule_idx]
        config = np.array(self.h5file[traj_name][config_idx])

        x = torch.tensor(config)
        #  x -= x.mean(dim=0)

        if self.normalize:
            x *= self.scaling_factor

        return GeometricData(
            x=x,
            node_type=torch.LongTensor(atoms),
            tau=torch.FloatTensor([lag]),
            bond_index=self.bond_index[molecule_idx],
            bond_type=self.bonds[molecule_idx],
        )

    def get_lag(self, molecule_idx):
        return self.lags[molecule_idx]
    
    def get_traj(self, mol_idx, as_numpy=False):
        """
        Returns the trajectory for a given molecule index.
        """
        LazyH5DatasetMixin.lazy_load(self)
        traj_name = self.traj_names[mol_idx]
        traj = self.h5file[traj_name][:] #self.h5file[self.split][traj_name]["traj"][()]
        if self.normalize:
            traj *= self.scaling_factor
        if as_numpy:
            return traj
        else:
            return torch.tensor(traj, dtype=torch.float32)
        
    def get_replica_exchange_traj(self, mol_idx, as_numpy=False):
        """
        Returns the replica exchange trajectory for a given molecule index.
        """
        LazyH5DatasetMixin.lazy_load(self)
        traj_name = self.traj_names[mol_idx].replace("md_0", "re_0")
        traj = self.h5file[traj_name][:] #self.h5file[self.split][traj_name]["traj"][()]
        if self.normalize:
            traj *= self.scaling_factor
        if as_numpy:
            return traj
        else:
            return torch.tensor(traj, dtype=torch.float32)


# class MDQM9(StandardDatasetMixin, MDQM9Base):
#     def __init__(
#         self,
#         path=None,
#         normalize=False,
#         split=None,
#         **kwargs,
#     ):
#         MDQM9Base.__init__(self, path=path, normalize=normalize, split=split)
#         StandardDatasetMixin.__init__(self)


class LaggedMDQM9(LaggedDatasetMixin, MDQM9Base):
    def __init__(
        self,
        path=None,
        sub_data_set="version_0",
        max_lag=1, # in ps, default is 1 ps
        fixed_lag=False,
        split=None,
        sub_sampling_indices_path=None,
        normalize=False,
        lazy_load=False,
        transform=None,
        ot_coupling=True,
        **kwargs,
    ):
        MDQM9Base.__init__(self, path=path, split=split, sub_sampling_indices_path=sub_sampling_indices_path, normalize=normalize, lazy_load=lazy_load)
        
        max_lag_in_steps = [ int(max(max_lag / self.lags[mol_idx], 1)) for mol_idx in self.molecule_idxs ]     
        LaggedDatasetMixin.__init__(
            self, max_lag=max_lag_in_steps, fixed_lag=fixed_lag, transform=transform, ot_coupling=ot_coupling, **kwargs
        )

    def __getitem__(self, idx):
        item = LaggedDatasetMixin.__getitem__(self, idx)
        # Should maybe be called Ntau?
        item["lag"] = item["cond"]["tau"] * item["lag"]

        return item

    def get_max_physical_lag(self):
        return max(self.lags.values()) * self.max_lag


def get_rdkit_mol(idx, path=None):
    path = path or "storage/mdqm9/mdqm9.sdf"
    mol = Chem.SDMolSupplier(path, removeHs=False)[int(idx)]
    return mol


def load_binary_topology(topology):
    mol = Chem.MolFromPDBBlock(topology.decode())
    return mol
