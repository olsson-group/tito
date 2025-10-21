import numpy as np
import torch
import mdtraj as md
import torch_geometric as geom

from torch_geometric.data import Batch

import tito.data as data
from tito.utils.utils import rdkit_to_mdtraj_topology
import tito.mlops as mlops 
from tito.utils.utils import sample_to_batch
from tito import utils
import tito.mlops as mlops 

def get_dataset(args):
    datasets = {
        "mdqm9": data.mdqm9.LaggedMDQM9,
        "timewarp": data.timewarp.LaggedTimewarp,
    }
    if args.data_set not in datasets:
        raise ValueError(f"Dataset {args.data_set} not supported. Choose from {list(datasets.keys())}.")
    dataset_class = datasets[args.data_set]
    ot_coupling = not args.no_ot if hasattr(args, 'no_ot') else True

    if args.mode == "inference" or args.mode == "sample": #inference
        if hasattr(args, "distinguish_atoms") and args.distinguish_atoms and args.data_set == "ala2":
            dataset = dataset_class(path=args.data_path, max_lag=args.lag, lazy_load=True, normalize=True, fixed_lag=args.lag,
                uniform=False, split=args.split, distinguish=True, ot_coupling=ot_coupling) 
        else:
            dataset = dataset_class(path=args.data_path, sub_data_set=args.sub_data_set, max_lag=args.lag, lazy_load=True, normalize=True, fixed_lag=args.lag, 
                uniform=False, split=args.split, ot_coupling=ot_coupling)
        return dataset
        
    else:
        uniform_lag = not args.no_uniform_lag if hasattr(args, 'no_uniform_lag') else True
        if args.sub_sampling_strategy:
            train_indices_path = args.sub_sampling_train_indices
            val_indices_path = args.sub_sampling_val_indices
        else:
            train_indices_path = None
            val_indices_path = None
        if hasattr(args, "distinguish_atoms") and args.distinguish_atoms and args.data_set == "ala2":
            train_dataset = dataset_class(path=args.data_path, max_lag=args.max_lag, lazy_load=True, normalize=True, fixed_lag=False,
                uniform=uniform_lag, split="train", distinguish=True, ot_coupling=ot_coupling) 
            val_dataset = dataset_class(path=args.data_path, max_lag=args.max_lag, lazy_load=True, normalize=True,  fixed_lag=False, 
                uniform=uniform_lag, split="val", distinguish=True, ot_coupling=ot_coupling) 
        else:
            train_dataset = dataset_class(path=args.data_path, max_lag=args.max_lag, lazy_load=True, normalize=True, fixed_lag=False,
                uniform=uniform_lag, split="train", sub_sampling_indices_path=train_indices_path, ot_coupling=ot_coupling) 
            val_dataset = dataset_class(path=args.data_path, max_lag=args.max_lag, lazy_load=True, normalize=True,  fixed_lag=False, 
                uniform=uniform_lag, split="val", sub_sampling_indices_path=val_indices_path, ot_coupling=ot_coupling) 

        return train_dataset, val_dataset

def get_base_dataset(args):
    datasets = {
        "ala2": data.ala2.ALA2Base,
        "mdqm9": data.mdqm9.MDQM9Base,
        "timewarp": data.timewarp.TimewarpBase,
    }
    if args.data_set not in datasets:
        raise ValueError(f"Dataset {args.data_set} not supported. Choose from {list(datasets.keys())}.")
    dataset_class = datasets[args.data_set]

    dataset = dataset_class(path=None, sub_data_set=args.sub_data_set, split=args.split, normalize=False, lazy_load=True) #inference
    return dataset

def get_batch(args, dataset, i_mol):
    if args.pdb_path is not None:
        positions = dataset.positions
        if args.batch_size > positions.shape[0]:
            indices = np.random.choice(range(positions.shape[0]), size=args.batch_size, replace=True)
        else:
            indices = np.random.choice(range(positions.shape[0]), size=args.batch_size, replace=False)
        samples = [dataset[i] for i in indices]
    elif args.unique_initial_condition:
        samples = [ dataset[dataset.lag_traj_boundaries[int(i_mol)]+args.initial_condition_index] for _ in range(args.batch_size) ]  
        if args.re_initial_condition:
            print("Using samples from Replica Exchange simulations ...", flush=True)
            for i_sample in range(len(samples)):
                samples[i_sample]["cond"].x = dataset.get_replica_exchange_traj(int(i_mol), as_numpy=False)[args.initial_condition_index]
    else:
        low_bound = dataset.lag_traj_boundaries[int(i_mol)]
        high_bound = dataset.lag_traj_boundaries[int(i_mol)+1]
        cond_indices = np.random.choice(range(low_bound, high_bound), size=args.batch_size, replace=False)
        samples = [ dataset[i] for i in cond_indices ]
 
    for item in samples: item["lag"] = torch.tensor([args.lag]) #set lag because its rounded to the closets dataset lag time

    cond_batch = Batch.from_data_list([item["cond"] for item in samples])
    lag_batch = torch.cat([item["lag"] for item in samples])
    lag_batch = lag_batch.unsqueeze(1)  
    batch = {"cond": cond_batch, "lag": lag_batch}

    batch["corr"] = batch['cond'].clone()
    #base_samples = torch.normal(0, 1, size=batch["cond"].x.shape) #add samples from the base distribution, this could be improved
    #base_samples = center_coordinates_batch(base_samples, batch["cond"].batch) 
    base_samples = dataset.basedistribution.sample_as(batch["cond"].x)
    
    batch["corr"].x = base_samples
    
    return batch

def re_scale_samples(batch, dataset):
    """
    Rescale the samples to the original scale of the dataset.
    """
    scaling_factor = dataset.scaling_factor if hasattr(dataset, 'scaling_factor') else 1.0
    for key in ['cond', 'corr', "traj"]:
        batch[key].x = batch[key].x / scaling_factor
    return batch
def get_save_path(args, i_mol, i_job=None):
    """
    Get the path to save the samples.
    """

    model_name = args.model_path.split("/")[-1].replace(".ckpt", "")
    if args.custom_system_initial_condition:
        system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
        
        if args.unique_initial_condition:
            pkl_path = f"results/custom/{system_name}/{model_name}/init_{str(args.initial_condition_index).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl"
            pdb_path = f"results/custom/{system_name}/{model_name}/init_{str(args.initial_condition_index).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pdb"
        else:
            pkl_path = f"results/custom/{system_name}/{model_name}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl"
            pdb_path = f"results/custom/{system_name}/{model_name}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pdb"
    else:
         if args.unique_initial_condition:
            pkl_path = f"results/{args.data_set}/{args.sub_data_set}/{model_name}/{args.split}/mol_{str(i_mol).zfill(5)}/init_{str(args.initial_condition_index).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl"
            pdb_path = f"results/{args.data_set}/{args.sub_data_set}/{model_name}/{args.split}/mol_{str(i_mol).zfill(5)}/init_{str(args.initial_condition_index).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pdb"
         else:
            pkl_path = f"results/{args.data_set}/{args.sub_data_set}/{model_name}/{args.split}/mol_{str(i_mol).zfill(5)}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl"
            pdb_path = f"results/{args.data_set}/{args.sub_data_set}/{model_name}/{args.split}/mol_{str(i_mol).zfill(5)}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pdb"

    if i_job is not None:
        pkl_path = pkl_path.replace(".pkl", f"_job_{str(i_job).zfill(3)}.pkl")
        pdb_path = pdb_path.replace(".pdb", f"_job_{str(i_job).zfill(3)}.pdb")
    return pkl_path, pdb_path

def save_results(batch, dataset, args, i_mol, i_job=None): 
    pkl_path, pdb_path  = get_save_path(args, i_mol, i_job)
    
    #init_conf = batch["cond"].x.reshape(len(torch.unique(batch["cond"].batch)), -1, 3).detach().cpu().numpy()[np.newaxis, ...]
    #trajs = batch["traj"].x.reshape(args.nested_samples, len(torch.unique(batch["traj"].batch)), -1, 3).detach().cpu().numpy()
    trajs = batch["traj"].x.reshape(args.nested_samples+1, len(torch.unique(batch["traj"].batch)), -1, 3).detach().cpu().numpy()
    #trajs = np.concatenate([init_conf, trajs], axis=0)
    trajs = np.swapaxes(trajs, 0, 1)
    if args.custom_system_initial_condition:
        custom_mol = mlops.load(args.custom_system_initial_condition)
        node_type = np.array(custom_mol["node_type"])
        bond_index, bond_type = custom_mol["bond_index"], custom_mol["bond_type"]
        mol = utils.create_rdkit_mol(node_type, bond_index.numpy(), bond_type.numpy())
    else:
        mol = dataset.mol_suppl[int(i_mol)]
        node_type = np.array(utils.get_atoms_from_rdkit(mol))
        bond_index, bond_type = utils.get_bonds_from_rdkit(mol)
    
    results = {
        "args": vars(args),
        "traj": trajs,
        "atoms": node_type,
        "bond_index": bond_index.numpy(),
        "bond_type": bond_type.numpy(),
        "timestamp": utils.get_timestamp(),
        "mol": mol,
    }
    
    mlops.save(results, pkl_path)
    print(f"Samples saved to {pkl_path}", flush=True)
    if args.save_pdb:
        #save pdb with the first trajectory
        traj_to_pdb(trajs[0], mol, dataset, pdb_path) #need to adapt to new output format
        print(f"PDB saved to {pdb_path}", flush=True)

def build_custom_initial_condition_batch(args, dataset):
    """
    Load a custom initial condition from a file.
    """
    
    print(f"Loading custom initial condition from {args.custom_system_initial_condition} ...", flush=True)
    cond = sample_to_batch(mlops.load(args.custom_system_initial_condition))
    if dataset.normalize:
        print("Dataset is normalized, scaling loaded samples to align with model scale ...", flush=True)
        if hasattr(dataset, 'scaling_factor'):
            _sf = dataset.scaling_factor
        else:
            _sf = 1.0
        cond.x = cond.x * _sf
    
    if args.unique_initial_condition:
        print(f"Using initial condition index {args.initial_condition_index} from the custom system.", flush=True)
        cond = [cond[args.initial_condition_index] for _ in range(args.batch_size)]
    else:
        print("Using random initial conditions from the custom system.", flush=True)
        random_indices = np.random.choice(range(0, len(cond)), size=args.batch_size, replace=False)
        cond = [cond[i] for i in random_indices]

    for item in cond: 
        item["lag"] = torch.tensor([args.lag])    

    cond_batch = geom.data.Batch.from_data_list([c for c in cond])
    lag_batch = torch.cat([item["lag"] for item in cond])
    lag_batch = lag_batch.unsqueeze(1)  
    batch = {"cond": cond_batch, "lag": lag_batch}

    batch["corr"] = batch['cond'].clone()
    base_samples = dataset.basedistribution.sample_as(batch["cond"].x)
    
    batch["corr"].x = base_samples
    
    return batch
    
def traj_to_pdb(trajs, mol, dataset, filename):
    """
    Convert a batch of data to a PDB file.
    """

    top = rdkit_to_mdtraj_topology(mol)
    md.Trajectory(trajs, top).save(filename)
    