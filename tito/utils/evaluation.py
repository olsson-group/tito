import mdtraj as md
import numpy as np
import torch
from deeptime.decomposition import VAMP

from tito.utils import utils


class VAMPScorer:
    def __init__(self, val_data):
        self.vm_val = VAMP(1).fit_fetch(val_data.T)

    def get_vamp2_score(self, sim_data):
        vm_sim = VAMP(1).fit_fetch(sim_data.T)
        return self.vm_val.score(2, vm_sim)


def get_dihedrals(mdtraj, dihedral_atoms):
    if isinstance(mdtraj, np.ndarray) or isinstance(mdtraj, torch.Tensor):
        mdtraj = utils.get_mdtraj(mdtraj, np.ones(mdtraj.shape[-2]))
    dihedrals = md.compute_dihedrals(mdtraj, dihedral_atoms)
    return dihedrals.T


def calculate_vamp2(data, name):
    pass
