import h5py
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import mdtraj
from rdkit import Chem

import tito.utils as utils
import torch_geometric as geom

from tito import DEVICE


class BaseDensity:
    def __init__(self, std):
        self.standard_deviation = std
    
    def sample_as(self, batch):
        if isinstance(batch, geom.data.Batch):
            x0 = torch.normal(0, 1, size=batch.x.shape, device=batch.x.device)*self.standard_deviation
            return utils.center_coordinates_batch(x0, batch.batch)

        elif isinstance(batch, torch.Tensor):
            x0 = torch.normal(0, 1, size=batch.shape, device=batch.device)*self.standard_deviation 
            return utils.center_coordinates(x0)

        #x0 = torch.normal(0, 1, size=batch.x.shape)*self.standard_deviation

def rot_ot(base, target):
    """
    Computes the optimal rotation and applies it in place to base, in a torch_geometric batched manner.
    :param base: tensor of coordinates, center at origin
    :param target: tensor of coordinates, centered at origin
    :param batch: tensor of batch indices torch-geometric
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """

    H = torch.matmul(base[:].transpose(0, 1), target[:])  
    U, S, Vt = torch.linalg.svd(H)  
    # check if rotation is improper -- if so make it proper
    d = torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1)))  #
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0
    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))
    #rotate noise
    base[:] = torch.matmul(base[:], R.transpose(0,1)) 
    return base, target

def permute_ot(base, target):
    """
    Finds optimal permutation and applies it in place to base, in a torch_geometric batched manner.
    :param base: tensor of coordinates, center at origin
    :param target: tensor of coordinates, centered at origin
    :param batch: tensor of batch indices torch-geometric
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    cost_matrix = torch.cdist(base, target, p=2).T
    #cost_matrix = (base[:].unsqueeze(0) - target[:].unsqueeze(1)).norm(dim=-1)
    row_indices, _ = linear_sum_assignment(cost_matrix.cpu().numpy())
    base[:] = base[:][row_indices, :]
    return base, target

def OT_coupler(base, target, plan="pr"):
    """
        Performs permutation and rotation OT coupling of minibatch. Assumes molecules are zero-centered.
        :param base: tensor  --- samples from source/base distribution
        :param target: tensor --- samples from target distribution
        :param batch: tensor --- index of different batch members
        :param plan: string --- describing the order of the different coupling operatorions, p: permute, r:rotation
    """
    plan_map = {"r": rot_ot,
                "p": permute_ot}
    for c in plan:
        base, target = plan_map[c](base, target)

    return base, target


class LaggedDatasetMixin:
    def __init__(self, max_lag, fixed_lag=False, transform=None, uniform=False, ot_coupling=True):
        #if max_lag is not float, is provided per molecule and we need create a max_lag per this dataset datapoint
        self.preprocess(max_lag) #this can be max(max_lag) for now
        self.ot_coupling = ot_coupling
        self.fixed_lag = fixed_lag
        self.transform = transform if transform is not None else lambda x: x
        self.uniform = uniform
        self.ot_coupling = ot_coupling
        self.basedistribution = BaseDensity(std=1.0) #removed name-magling to be able to use from hygher level datasets

        if not ot_coupling:
            print("OT coupling is disabled...")

    def preprocess(self, max_lag):
        #print("Preprocessing  LaggedDatasetMixin ... ")
        if isinstance(max_lag, (float, int)):  # this should be made more robust
            max_lag = [max_lag] * (len(self.traj_boundaries) - 1)
        max_lag = np.array(max_lag, dtype=int)
        total_size = sum(
            max(0, end - start - max_lag[i_mol])
            for i_mol, (start, end) in enumerate(zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]))
        )
        data0_idxs = np.empty(total_size, dtype=int)
        new_max_lag = np.empty(total_size, dtype=int)
        lag_traj_boundaries = [0]
        idx = 0
        for i_mol, (start, end) in enumerate(zip(self.traj_boundaries[:-1], self.traj_boundaries[1:])):
            traj_len = end - start
            non_lagged_length = max(0, traj_len - max_lag[i_mol])
            data0_idxs[idx : idx + non_lagged_length] = np.arange(start, start + non_lagged_length)
            new_max_lag[idx : idx + non_lagged_length] = max_lag[i_mol]
            lag_traj_boundaries.append(lag_traj_boundaries[-1] + non_lagged_length)
            idx += non_lagged_length

        self.lag_traj_boundaries = np.array(lag_traj_boundaries, dtype=int)
        self.data0_idx = data0_idxs
        self.max_lag = new_max_lag
    
    # def compute_data0_idxs(self, max_lag):
    #     print("Building data0 index ... ")
    #     max_lag = int(max_lag)
    #     total_size = sum(
    #         max(0, end - start - max_lag)
    #         for start, end in zip(self.traj_boundaries[:-1], self.traj_boundaries[1:])
    #     )
    #     data0_idxs = np.empty(total_size, dtype=int)
    #     idx = 0
    #     for start, end in zip(self.traj_boundaries[:-1], tqdm(self.traj_boundaries[1:])):
    #         traj_len = end - start
    #         non_lagged_length = max(0, traj_len - max_lag)
    #         data0_idxs[idx : idx + non_lagged_length] = np.arange(start, start + non_lagged_length)
    #         idx += non_lagged_length

    #     return data0_idxs

    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
        max_lag_item = self.max_lag[idx]
        if self.fixed_lag:
            lag = max_lag_item 
        elif self.uniform:
            lag = np.random.randint(1, max_lag_item+1)
        else:
            log_lag = np.random.uniform(0, np.log(max_lag_item+1))
            lag = int(np.floor(np.exp(log_lag)))

        data0_idx = self.data0_idx[idx]
        data0 = super().__getitem__(data0_idx)
        datat = super().__getitem__(data0_idx + lag)

        base_sample = self.basedistribution.sample_as(datat.x)

        if self.ot_coupling:
            base, target = OT_coupler(base_sample, datat.x, plan="pr")
        else:
            base, target = base_sample, datat.x

        datat.x = target
        datat.xbase = base
        

        item = {"cond": data0, "target": datat, "lag": torch.Tensor([lag])}
        return self.transform(item)


class StandardDatasetMixin:
    def __init__(self, transform=None):
        self.transform = transform if transform is not None else lambda x: x

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        item = {"target": data}
        return self.transform(item)


class LazyH5DatasetMixin:
    def __init__(self, path, lazy_load=False):
        self.path = path
        self.loaded = False
        if lazy_load:
            self.lazy_load()

    def lazy_load(self):
        if not self.loaded:
            self.h5file = h5py.File(self.path, "r")
            self.loaded = True

class PDBDataset:
    def __init__(self, pdb_path, scaling_factor):
        self.pdb_path = pdb_path
        self.traj = mdtraj.load_pdb(pdb_path)
        self.positions = self.traj.xyz
        self.mol_suppl = [Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)]
        self.scaling_factor = scaling_factor

        bond_index, bonds = utils.get_bond_index_and_bonds(self.mol_suppl[0])
        atoms = [atom.GetAtomicNum() for atom in self.mol_suppl[0].GetAtoms()]

        self.bond_index = [bond_index]
        self.bonds = [bonds]
        self.atoms = [atoms]

        # Placeholders expected by downstream code
        self.normalize = True #careful here, provide positions as model expects them
        self.basedistribution = BaseDensity(1)
        self.name = "pdb_input"

    def __len__(self):
        return self.positions.shape[0]
    def __getitem__(self, index):
        pos = torch.tensor(self.positions[index])
        atoms = torch.LongTensor(self.atoms[0])
        bond_index = self.bond_index[0]
        bonds = self.bonds[0]

        x = pos
        if self.normalize:
            x *= self.scaling_factor

        item = {"cond": geom.data.Data(
            x=x,
            node_type=atoms,
            bond_index=bond_index,
            bond_type=bonds,
        )}
        return item
