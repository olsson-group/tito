import argparse
import tito.models.model as model

from tito.data.datasets import PDBDataset
from tito.utils.data import get_dataset, get_batch, re_scale_samples, save_results, build_custom_initial_condition_batch
from tito.data.datasets import BaseDensity


def sample_model(args, cfm, dataset, i_mol):
    """
    Sample from the model using the provided model path and initial condition.
    """
    print(f"Sampling from model with the following parameters:", flush=True)
    print(f"Lag: {args.lag}")
    if args.unique_initial_condition:
        print(f"Initial Condition: {args.initial_condition_index}", flush=True)
    else:
        print("Using random initial conditions.", flush=True)
    print("Molecule index:", i_mol, flush=True)

    if args.base_density_std != 1.0:
        print(f"Using custom base distribution with std={args.base_density_std}.", flush=True)
        dataset.basedistribution = BaseDensity(std=args.base_density_std)

    if args.custom_system_initial_condition:
        print(f"Using custom initial condition from {args.custom_system_initial_condition}", flush=True)
        batch = build_custom_initial_condition_batch(args, dataset)
    else:
        batch = get_batch(args, dataset, i_mol)
    device = next(cfm.parameters()).device
    initc = {k: v.to(device) for k, v in batch.items()}  # Move to the same device as the model
    
    # Sample from the model
    out_batch = cfm.sample(initc, ode_steps=args.ode_steps, nested_samples=args.nested_samples, base_distribution=BaseDensity(std=args.base_density_std))
    
    # Save the samples
    if dataset.normalize:
        print("Rescaling samples to original scale ...", flush=True)
        out_batch = re_scale_samples(out_batch, dataset)
        
    return out_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--tag", type=str, default="best", help="Model wandb tag.")
    parser.add_argument("--pdb_path", type=str, default=None, help="Path of PDB file of molecule to sample.")
    parser.add_argument("--data_set", type=str, default="mdqm9", required=False, help="Dataset options: [ala2, mdqm9, timewarp]")
    parser.add_argument("--data_path", type=str, required=False, help="Path to dataset directory.")
    parser.add_argument("--sub_data_set", type=str, default="huge", required=False, help="Sub-dataset options for timewarp [large, huge].")
    parser.add_argument("--split", type=str, default="test", required=False, help="[train, val, test]")
    parser.add_argument("--mol_indices", type=int, nargs="+", default=[0], help="List of molecule ids for the dataset.")
    parser.add_argument("--lag", type=float, default=10.0, help="lag time in ps for mdqm9 and timewarp, multiples of data dt for ALA2.")
    parser.add_argument("--unique_initial_condition", action="store_true", help="If set, use a unique initial condition for sampling. Otherwise, use random initial conditions.")
    parser.add_argument("--initial_condition_index", type=int, default=0, help="If unique_initial_condition, index of initial condition to use from the dataset.")
    parser.add_argument("--nested_samples", type=int, default=1, help="Number of nested steps to sample.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples processed in parallel by the model.")
    parser.add_argument("--ode_steps", type=int, default=20, help="Number of ODE steps to sample.")
    parser.add_argument("--save_pdb", action="store_true", help="Save samples as PDB file. If batch size > 1, the pdb contains the first trajetory.")
    parser.add_argument('--distinguish_atoms', action='store_true', help="If set, distinguish atoms in the dataset.") #this could be read from model, not priority since only for ala2
    parser.add_argument("--custom_system_initial_condition", type=str, default=None, help="Path to custom test system for sampling. Loads initial condition from this file. If set, uses initial_condition_index to select state if multiple. Uses dataset normalization and scaling.")
    parser.add_argument("--i_job", type=int, default=None, help="Job index for parallel sampling, used to distinguish output files.") #consider changing default to 0 in final version
    parser.add_argument("--re_initial_condition", action="store_true", help="If set, use initial conditions from Replica Exchange simulations.")
    parser.add_argument('--base_density_std', type=float, default=1.0, help="standard deviation of the base distribution for sampling. Used for Flory exponent extrapolation experiments.") 
    args = parser.parse_args()
    args.fixed_lag = True
    args.mode = "sample"  # Set mode to sample for compatibility with dataset loading function
    if args.data_set == "ala2" or args.data_set == "mdqm9":
        args.sub_data_set = "version_0"

    if args.pdb_path is not None:
        
        if args.data_set == "mdqm9":
            from tito.data.mdqm9 import SCALING_FACTOR
        elif args.data_set == "timewarp":
            from tito.data.timewarp import SCALING_FACTOR
        dataset = PDBDataset(args.pdb_path, scaling_factor=SCALING_FACTOR)
    else:
        dataset = get_dataset(args)
    cfm = model.CFM.load_from_checkpoint(checkpoint_path=args.model_path)
    print("Model loaded ...", flush=True)
    for i_mol in args.mol_indices:
        out_batch = sample_model(args=args, cfm=cfm, dataset=dataset, i_mol=i_mol)
        save_results(batch=out_batch, dataset=dataset, args=args, i_mol=i_mol, i_job=args.i_job)
