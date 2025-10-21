import tempfile
import os
import copy
import warnings
import os.path as osp
import mdtraj as md
import numpy as np
#import ot
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from deeptime.decomposition import TICA
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from deeptime.decomposition import VAMP

from tito.utils import utils
from tito import mlops

def kl_divergence(p, q):
    mask = (p != 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def jensen_shannon_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

def emd(x,y):
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("Y1 and Y2 must have the same number of points.")
    
    d = cdist(x, y, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(d)
    emd_value = d[row_ind, col_ind].sum() / n
    return emd_value


def get_bond_lengths(traj, bond_index):
    from_ = bond_index[0]
    to_ = bond_index[1]
    diff = traj[..., to_, :] - traj[..., from_, :]
    dists = np.linalg.norm(diff, axis=-1)
    return dists

def find_dihedral_atoms(mol):
    dihedral_atoms = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()

        neighbors1 = [
            atom.GetIdx() for atom in mol.GetAtomWithIdx(atom1).GetNeighbors() if atom.GetIdx() != atom2
        ]

        neighbors2 = [
            atom.GetIdx() for atom in mol.GetAtomWithIdx(atom2).GetNeighbors() if atom.GetIdx() != atom1
        ]

        for n1 in neighbors1:
            for n2 in neighbors2:
                dihedral_atoms.append((n1, atom1, atom2, n2))
                break
            break

    return dihedral_atoms


def get_dihedrals(trajs, dihedral_atoms):
    dihedrals = []
    for t in trajs:
        mdtraj = utils.get_mdtraj(t, np.ones(trajs.shape[-2]))
        dihedrals.append(md.compute_dihedrals(mdtraj, dihedral_atoms))
    return np.array(dihedrals)

def get_model_samples_paths(args):
    paths = []
    
    for i_mol in args.mol_indices:
        mol_paths = []

        if args.custom_system_initial_condition:
            system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
            pre_path = f"results/custom/{system_name}/{args.model}/"
        else:
            pre_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(i_mol).zfill(5)}/"

        if args.initialization == "random":
            if args.jobs is not None:
                for i_job in args.jobs:
                    mol_paths.append(f"{pre_path}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}_job_{str(i_job).zfill(3)}.pkl")
            else:
                mol_paths.append(f"{pre_path}/random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl") #_ode_steps_{args.ode_steps}
        else:
            if args.jobs is not None:
                for i_job in args.jobs:
                    mol_paths.append(f"{pre_path}/init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}_job_{str(i_job).zfill(3)}.pkl")
            else:
                mol_paths.append(f"{pre_path}/init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.pkl")
        paths.append(mol_paths)
        
    return paths

def get_tica_projections_path(mol_idx, args):
    if args.custom_system_initial_condition:
        system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
        model_tica_projections_pre_path = f"results/custom/{system_name}/{args.model}/"
    else:
        model_tica_projections_pre_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/"
    if args.initialization == "random":
        model_tica_projections_path = f"{model_tica_projections_pre_path}/tica_projections_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
    else:
        model_tica_projections_path = f"{model_tica_projections_pre_path}/tica_projections_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"

    return model_tica_projections_path

def check_and_get_paths(args):
    """
    Checks if the necessary directories exist, and creates them if they don't.
    """
    mol_indices = copy.deepcopy(args.mol_indices)
    missing_indices = []
    paths = get_model_samples_paths(args)

    for i_mol, mol_paths in zip(args.mol_indices, paths):
        for path in mol_paths:
            if not osp.exists(path):
                missing_indices.append(i_mol)
                mol_indices.remove(i_mol)
    if len(missing_indices) > 0:
        warnings.warn(f"Missing sampled trajectories for molecules: {missing_indices}")

    return paths, mol_indices, missing_indices

def compute_and_save_dihedrals_and_sinusoids(mol, trajs, mol_idx, args, mode="md"):
    """
    Computes dihedrals for a given molecule and trajectory.
    """
    dihedral_atoms = find_dihedral_atoms(mol)
    dihedrals = get_dihedrals(trajs, dihedral_atoms)
    sinusoids = utils.get_sinusoids(dihedrals)

    # Save dihedrals to file
    if args.custom_system_initial_condition:
        system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
        if mode == "md":
            path = f"results/custom/{system_name}/md/dihedrals.npy"
        elif mode == "model":
            if args.initialization == "random":
                path = f"results/custom/{system_name}/{args.model}/dihedrals_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
            else:
                path = f"results/custom/{system_name}/{args.model}/dihedrals_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
    else:   
        if mode == "md":
            path = f"results/{args.data_set}/{args.sub_data_set}/md/{args.split}/mol_{str(mol_idx).zfill(5)}/dihedrals.npy"
        elif mode == "re":
            path = f"results/{args.data_set}/{args.sub_data_set}/re/{args.split}/mol_{str(mol_idx).zfill(5)}/dihedrals.npy"
        elif mode == "model":
            if args.initialization == "random":
                path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/dihedrals_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
            else:
                path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/dihedrals_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        elif mode == "mdft":
            pre_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}"
            path = f"{pre_path}/dihedrals_ft_positions_{args.mdft_ps}_ps_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        else:
            raise ValueError("Mode must be either 'md', 're' or 'model'.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, dihedrals)
    #print(f"Dihedrals saved to {path}")
    
    return dihedrals, sinusoids

def compute_and_save_ticas(sinusoids, mol_idx, args):
    """
    Computes TICA models for the dihedrals.
    Note: This function is only used for reference md trajectories.
    """
    #following deeptime docs, but different from previous implementation
    estimator = TICA(lagtime=args.lag_tica, dim=2).fit(sinusoids) #careful with dim
    model = estimator.fetch_model()
    projections = model.transform(sinusoids)
    
    # Save TICA model to file
    #model_path = f"results/{args.data_set}/md/{args.split}/mol_{str(mol_idx).zfill(5)}/tica_model.pkl"
    if args.custom_system_initial_condition:
        system_name = args.custom_system_initial_condition.split("/")[-1].split(".")[0]
        projections_path = f"results/custom/{system_name}/md/tica_projections.npy"
    else:
        projections_path = f"results/{args.data_set}/{args.sub_data_set}/md/{args.split}/mol_{str(mol_idx).zfill(5)}/tica_projections.npy"
    os.makedirs(os.path.dirname(projections_path), exist_ok=True)
    np.save(projections_path, projections)
    #print(f"TICA projections saved to {projections_path}")
    #mlops.save(model, model_path)
    #print(f"TICA model saved to {model_path}")
    
    return model, projections


def compute_and_save_vamp_singular_values_and_gaps(ref_data, predict_data, mol_idx, args, lag_factor=1): #n_singular_values=10

    vamp_model_ref = VAMP(lagtime=args.lag_vamp*lag_factor).fit(ref_data).fetch_model()
    vamp_model_pred = VAMP(lagtime=args.lag_vamp).fit(predict_data).fetch_model()

    vamp_svs_ref = vamp_model_ref.singular_values
    vamp_svs_pred = vamp_model_pred.singular_values

    ref_score = vamp_model_ref.score(2, vamp_model_ref)
    pred_score = vamp_model_ref.score(2, vamp_model_pred)
    vamp_gap = ref_score - pred_score

    # Save VAMP singular values to file
    if args.custom_system_initial_condition:
        vamp_svs_ref_path = f"results/custom/{args.custom_system_initial_condition.split('/')[-1].split('.')[0]}/md/vamp_singular_values_lag_{int(args.lag)}.npy"
        if args.initialization == "random":
            vamp_svs_pred_path = f"results/custom/{args.custom_system_initial_condition.split('/')[-1].split('.')[0]}/{args.model}/vamp_singular_values_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        else:
            vamp_svs_pred_path = f"results/custom/{args.custom_system_initial_condition.split('/')[-1].split('.')[0]}/{args.model}/vamp_singular_values_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
    else:
        vamp_svs_ref_path = f"results/{args.data_set}/{args.sub_data_set}/md/{args.split}/mol_{str(mol_idx).zfill(5)}/vamp_singular_values_lag_{int(args.lag)}.npy"
        if args.initialization == "random":
            vamp_svs_pred_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/vamp_singular_values_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        else:
            vamp_svs_pred_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/vamp_singular_values_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"

    np.save(vamp_svs_ref_path, vamp_svs_ref)
    np.save(vamp_svs_pred_path, vamp_svs_pred)
    #print(f"VAMP singular values saved to {vamp_svs_ref_path} and {vamp_svs_pred_path}")

    # Save VAMP gap to file
    if args.custom_system_initial_condition:
        if args.initialization == "random":
            vamp_gap_path = f"results/custom/{args.custom_system_initial_condition.split('/')[-1].split('.')[0]}/{args.model}/vamp_gap_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        else:
            vamp_gap_path = f"results/custom/{args.custom_system_initial_condition.split('/')[-1].split('.')[0]}/{args.model}/vamp_gap_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
    else:
        if args.initialization == "random":
            vamp_gap_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/vamp_gap_random_init_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
        else:
            vamp_gap_path = f"results/{args.data_set}/{args.sub_data_set}/{args.model}/{args.split}/mol_{str(mol_idx).zfill(5)}/vamp_gap_init_{str(int(args.initialization)).zfill(6)}_lag_{int(args.lag)}_nested_{args.nested_samples}_ode_steps_{args.ode_steps}.npy"
    np.save(vamp_gap_path, vamp_gap)
    #print(f"VAMP gap saved to {vamp_gap_path}")

    return vamp_svs_ref, vamp_svs_pred, vamp_gap


def update_histogram(current_histogram, new_data, bins):
    hist, _ = np.histogram(new_data, bins=bins)
    if current_histogram is None:
        current_histogram = hist
    else:
        current_histogram += hist
    return current_histogram


def update_rdkit_mol_positions(mol, positions):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol


def cast_input(input_):
    if isinstance(input_, np.ndarray):
        input_ = input_.tolist()
    return input_


def wasserstein_distance(d1, d2):
    return ot.wasserstein_1d(d1, d2) #rewrite with scipy 


def get_rdkit_mol(
    positions,
    atoms,
    bond_index,
    bond_types,
):
    mol = Chem.RWMol()
    atom_map = {}

    positions = cast_input(positions)
    atoms = cast_input(atoms)
    bond_index = cast_input(bond_index)
    bond_types = cast_input(bond_types)

    for i, atom_type in enumerate(atoms):
        atom = Chem.Atom(atom_type)
        atom_idx = mol.AddAtom(atom)
        atom_map[i] = atom_idx

    added_bonds = ()
    for bond_idx, bond_type in zip(bond_index, bond_types):
        added_bonds += (bond_idx,)
        if bond_idx[::-1] in added_bonds:
            continue
        atom1, atom2 = int(bond_idx[0]), int(bond_idx[1])
        mol.AddBond(atom_map[atom1], atom_map[atom2], Chem.BondType.values[bond_type])

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    Chem.MolToPDBFile(mol, "molecule.pdb")
    return mol


def get_ala2_chirality(pos):
    mol = Chem.MolFromPDBFile("storage/ala2/alanine-dipeptide-nowater.pdb", removeHs=False)
    conf = mol.GetConformer()

    if hasattr(pos, "tolist"):
        pos *= 10  # Convert to angstroms
        pos = pos.tolist()

    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*p))

    with tempfile.NamedTemporaryFile() as f:
        Chem.MolToPDBFile(mol, f.name)
        mol = Chem.MolFromPDBFile(f.name, removeHs=False)

    chiral_centers = get_chiral_centers(mol)
    return chiral_centers[0][1]


def get_chirality(mol, pos):
    conf = mol.GetConformer()

    if hasattr(pos, "tolist"):
        pos *= 10  # Convert to angstroms
        pos = pos.tolist()

    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*p))

    with tempfile.NamedTemporaryFile() as f:
        Chem.MolToPDBFile(mol, f.name)
        mol = Chem.MolFromPDBFile(f.name, removeHs=False)

    chiral_centers = get_chiral_centers(mol)
    return chiral_centers


def get_chiral_centers(mol):
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    return chiral_centers
