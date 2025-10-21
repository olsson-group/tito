import os
import tempfile
from argparse import ArgumentParser

import h5py
import mdtraj as md
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from tito import utils
from tito.data.timewarp import SCALING_FACTOR


def main(args):
    basepath = args.path
    splits = args.splits
    name = get_name(basepath, splits)
    naa = 4 if "4AA" in basepath else 2

    with h5py.File(name, "w") as file:
        for split in splits:
            print(f"Processing {split}...")
            path = os.path.join(basepath, split)
            files = os.listdir(path)
            if not files:
                continue

            topologies = [f[:naa] for f in files if f.endswith(".pdb")]
            trajs = [f[:naa] for f in files if f.endswith(".npz")]

            get_traj_name = lambda peptide: os.path.join(path, f"{peptide}-traj-arrays.npz")
            get_top_name = lambda peptide: os.path.join(path, f"{peptide}-traj-state0.pdb")

            missing_trajs = set(topologies) - set(trajs)
            missing_topologies = set(trajs) - set(topologies)
            intersection = set(topologies) & set(trajs)

            print(f"Missing trajs: {missing_trajs}")
            print(f"Missing topologies: {missing_topologies}")

            if split not in ["train", "val", "test"]:
                split = "train"

            grp = file.create_group(split)
            unique_lags = None

            for name in tqdm(intersection):
                print(f"\rProcessing {name}...", end="")
                top = get_top_name(name)
                traj = get_traj_name(name)

                protein_grp = grp.create_group(name)

                traj = np.load(traj)

                x = traj["positions"][()][::5]
                t = traj["time"][()][::5]
                e = traj["energies"][()][::5]
                x = utils.center_traj(x)

                t_diff = np.round(np.diff(t), 5)
                assert np.all(t_diff == 5), print(f"all steps are not 5ps in {name}")

                protein_grp.create_dataset("traj", data=x)
                protein_grp.create_dataset("time", data=t)
                protein_grp.create_dataset("energies", data=e)

                ## THE TOPOLOGIES IN THE PDB FILES ARE UNPHYSICAL SO WE NEED TO USE A TOP FROM THE TRAJ ##
                top = md.load(get_top_name(name))
                top.xyz = x[None, 0]
                top.save("/tmp/traj.pdb")
                mol = Chem.MolFromPDBFile("/tmp/traj.pdb", removeHs=False, sanitize=True)
                ##########################################################################################

                mol_block = Chem.MolToMolBlock(mol)
                protein_grp.create_dataset("mol", data=mol_block)

        data = file.create_group("data")
        data.create_dataset("std", data=SCALING_FACTOR)


def to_tuple(arr):
    return tuple(map(tuple, arr))


def load_binary_topology(topology):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as tmp:
        tmp.write(topology.decode())
        tmp.flush()
        return md.load_topology(tmp.name)


def get_name(path, splits):
    h5_name_postfix = ""
    if not ("train" in splits and "val" in splits and "test" in splits):
        splits = sorted(splits)
        h5_name_postfix = f"_{'_'.join(splits)}"

    basename = os.path.basename(os.path.normpath(path))
    name = os.path.join(os.path.dirname(path), f"{basename}{h5_name_postfix}.h5")
    return name


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--splits", type=str, nargs="+", default=["test", "train", "val"])

    args = parser.parse_args()

    main(args)
