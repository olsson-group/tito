#This script computes dihedrals, computes and saves TICA models and computes VAMP gaps for reference trajectories.
import argparse
from tqdm import tqdm
import numpy as np
from tito.utils import utils
from tito.utils.data import get_base_dataset
from tito.utils.analysis import compute_and_save_dihedrals_and_sinusoids, compute_and_save_ticas, compute_and_save_vamp_singular_values_and_gaps, check_and_get_paths, get_tica_projections_path

import tito.mlops as mlops 
from types import SimpleNamespace
import pickle

def analyze(args):
    print("Loading dataset...")
    dataset = get_base_dataset(args) 
    if args.data_set == "ala2" or args.data_set == "mdqm9":
        args.sub_data_set = "version_0"

    if args.custom_system_initial_condition:
        mol_indices = [0]

    #Check if molecules have been sampled with specified parameters
    paths, mol_indices, missing_indices = check_and_get_paths(args)

    print("Analyzing molecules...")
    for i_mol in tqdm(mol_indices):
        
        if args.custom_system_initial_condition:
            mol = mlops.load(paths[0])["mol"]
        else:
            mol = dataset.mol_suppl[i_mol]
        
        if args.custom_system_initial_condition:
            #md_trajs = mlops.load(args.custom_system_initial_condition)["traj"]
            system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
            md_trajs = np.load(f"/proj/berzelius-2025-189/users/x_juavi/md/results/all/{system_name}/traj.npz")["positions"]  #assuming only one trajectory per custom systems
        else:
            md_trajs = dataset.get_traj(i_mol)
        if md_trajs.ndim == 3:
            md_trajs = np.expand_dims(md_trajs, axis=0) 
        dihedrals_md, sinusoids_md = compute_and_save_dihedrals_and_sinusoids(mol, md_trajs, mol_idx=i_mol, args=args, mode="md")  # mode="md" to save in md folder
        tica_models, tica_projections_md = compute_and_save_ticas(sinusoids_md, mol_idx=i_mol, args=args)
        if args.process_replica_exchange_trajectory:
            re_trajs = dataset.get_replica_exchange_traj(i_mol)
            dihedrals_re, sinusoids_re = compute_and_save_dihedrals_and_sinusoids(mol, re_trajs, mol_idx=i_mol, args=args, mode="re")
            tica_projections_re = tica_models.transform(sinusoids_re)
            re_projections_path = f"results/{args.data_set}/{args.sub_data_set}/re/{args.split}/mol_{str(i_mol).zfill(5)}/tica_projections_re.npy"
            np.save(re_projections_path, tica_projections_re)
        if args.process_mdft_data:
            for mdft_ps in args.mdft_ps:
                pre_path = paths[i_mol][0].rsplit("/", maxsplit=1)[0]
                mdft_filename = f"{pre_path}ft_positions_{mdft_ps}_ps_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy.npy" #potentialy implement non-random conditions
                mdft_data = np.load(mdft_filename)
                dihedrals_mdft, sinusoids_mdft = compute_and_save_dihedrals_and_sinusoids(mol, mdft_data, mol_idx=i_mol, args=args, mode="mdft")
                tica_projections_mdft = tica_models.transform(sinusoids_mdft)
                mdft_projections_path = f"{pre_path}/tica_projections_ft_positions_{mdft_ps}_ps_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
                np.save(mdft_projections_path, tica_projections_mdft)

        # Aggregate data from different jobs with same parameters
        model_trajs = []
        mol_paths = paths[i_mol]
        for path in mol_paths:
            with open(path, "rb") as f:
                model_trajs.append(pickle.load(f)["traj"])
        model_trajs = np.concatenate(model_trajs, axis=0)

        dihedrals_tito, sinusoids_tito = compute_and_save_dihedrals_and_sinusoids(mol, model_trajs, mol_idx=i_mol, args=args, mode="model") #careful here with loaded_data['traj'][:,1] if nested sampling is used
        tica_projections_tito = tica_models.transform(sinusoids_tito)
        model_tica_projections_path = get_tica_projections_path(i_mol, args)
        np.save(model_tica_projections_path, tica_projections_tito)
        
        # Compute VAMP singular values
        if args.custom_system_initial_condition:
            md_report_interval = 10 #ps
        else:
            md_report_interval = dataset.lags[i_mol]
        lag_factor = int(args.lag / md_report_interval)
        vamp_scores_ref, vamp_scores_pred, vamp_gap = compute_and_save_vamp_singular_values_and_gaps(sinusoids_md, sinusoids_tito, i_mol, args, lag_factor=lag_factor)


def main():
    parser = argparse.ArgumentParser(description="TITO reference trajectory analysis script.")
    parser.add_argument('--data_set', type=str, default="mdqm9", required=False, help="Dataset options: [ala2, mdqm9, timewarp]")
    parser.add_argument("--sub_data_set", type=str, default="huge", required=False, help="Sub-dataset options for timewarp [large, huge].")
    parser.add_argument("--custom_system_initial_condition", type=str, default=None, help="Path to custom test system for sampling. Loads initial condition from this file. If set, uses initial_condition_index to select state if multiple. Uses dataset normalization and scaling.")
    parser.add_argument('--split', type=str, default="test", required=False, help="[train, val, test]")
    parser.add_argument('--mol_indices', type=int, nargs='+', default=[0], help="Molecule indices to analyze. Default is [0].")
    parser.add_argument('--model', type=str, default=None, help="wandb model name.")
    parser.add_argument('--lag', type=float, default=1000., help="lag of sampled trajectories.")
    parser.add_argument('--nested_samples', type=int, default=1000, help="Number of nested samples in model file.")
    parser.add_argument('--initialization', type=str, default="random", help="initialization of samples.")
    parser.add_argument('--ode_steps', type=int, default=40, help="Number of ODE steps.")
    parser.add_argument('--lag_tica', type=int, default=1, help="lag in steps (md) for TICA computation.")
    parser.add_argument('--lag_vamp', type=int, default=1, help="lag in steps (model) for VAMP computation.")
    parser.add_argument('--jobs', type=int, nargs='+', default=None, help="Indices of jobs to aggregate data from.")
    parser.add_argument('--process_replica_exchange_trajectory', action='store_true', help="If set, processes the replica exchange trajectory.")
    parser.add_argument('--process_mdft_data', action='store_true', help="If set, processes the MD fine-tuned data.")
    parser.add_argument('--mdft_ps', type=int,  nargs='+', default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                                                    200, 300, 400, 500, 600, 700, 800, 900, 1000], help="MD fine-tuned data time step (in ps).")
    args = parser.parse_args()
    analyze(args)

if __name__ == "__main__":
    main()