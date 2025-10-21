import os
import tempfile
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
#import ot
import scipy.stats as stats
import torch
import torch_geometric as geom
from mdtraj import element
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_scatter import scatter
from rdkit.Chem import AllChem

from tito import DEVICE

ELEMENTS = {
    1: element.hydrogen,
    2: element.helium,
    3: element.lithium,
    4: element.beryllium,
    5: element.boron,
    6: element.carbon,
    7: element.nitrogen,
    8: element.oxygen,
    9: element.fluorine,
    10: element.neon,
    11: element.sodium,
    12: element.magnesium,
    13: element.aluminum,
    14: element.silicon,
    15: element.phosphorus,
    16: element.sulfur,
    17: element.chlorine,
    18: element.argon,
    19: element.potassium,
    20: element.calcium,
    21: element.scandium,
    22: element.titanium,
    23: element.vanadium,
    24: element.chromium,
    25: element.manganese,
    26: element.iron,
    27: element.cobalt,
}


def get_unique_bond_indices(bond_index):
    bond_index = bond_index
    sorted_edges = np.sort(bond_index, axis=0)
    _, unique_indices = np.unique(sorted_edges, axis=1, return_index=True)
    return np.sort(unique_indices).T


def is_centered(batch):
    return torch.allclose(
        com := batch.x.mean(axis=0),
        torch.zeros_like(com),
        atol=1e-6,
    )


def get_element_names(atom_numbers):
    return [ELEMENTS[int(atom_number)].symbol for atom_number in atom_numbers]


def replace_virtual_node_with_flour(atoms):
    if hasattr(atoms, "clone"):
        atoms = atoms.clone()
    atoms = np.array(atoms)
    atoms[atoms == 0] = 9
    return atoms


def remove_angular_momentum(dx, batch):
    center_batch(batch)

    L = get_angular_momentum(batch, dx)
    I = get_moment_of_inertia(batch)

    omega = torch.linalg.solve(I, L)  # [batch.batch]
    omega = omega[batch.batch]
    v = torch.linalg.cross(omega, batch.x)

    corrected = dx - v
    L_corrected = get_angular_momentum(batch, corrected)
    assert L_corrected.mean() < 0.0001

    return corrected


def compute_principal_axes(r):
    I = get_moment_of_inertia(r)
    eigvals, eigvecs = torch.linalg.eigh(I)
    return eigvals, eigvecs


def get_moment_of_inertia(batch):
    center_batch(batch)

    x, y, z = batch.x[:, 0], batch.x[:, 1], batch.x[:, 2]

    batch_idx = batch.batch

    diag_xx = scatter(y**2 + z**2, batch_idx, reduce="sum")
    diag_yy = scatter(x**2 + z**2, batch_idx, reduce="sum")
    diag_zz = scatter(x**2 + y**2, batch_idx, reduce="sum")

    off_diag_xy = -scatter(x * y, batch_idx, reduce="sum")
    off_diag_xz = -scatter(x * z, batch_idx, reduce="sum")
    off_diag_yz = -scatter(y * z, batch_idx, reduce="sum")

    n_conf = diag_xx.size(0)
    I = torch.zeros((n_conf, 3, 3), device=batch.x.device)

    I[:, 0, 0] = diag_xx
    I[:, 1, 1] = diag_yy
    I[:, 2, 2] = diag_zz

    I[:, 0, 1] = off_diag_xy
    I[:, 1, 0] = off_diag_xy
    I[:, 0, 2] = off_diag_xz
    I[:, 2, 0] = off_diag_xz
    I[:, 1, 2] = off_diag_yz
    I[:, 2, 1] = off_diag_yz

    return I


def get_moment_of_inertia1(batch):
    center_batch(batch)
    n_conf = len(batch)
    r = batch.x.reshape(n_conf, batch.x.shape[0] // n_conf, 3)
    r_squared = batch.x**2
    diag = (
        torch.eye(3).repeat(n_conf, 1, 1)
        * scatter(r_squared, batch.batch, reduce="sum", dim=0).sum(dim=1)[:, None, None]
    )
    off_diag = -torch.matmul(r.swapaxes(-1, -2), r)
    I = diag + off_diag
    return I


def get_angular_momentum(batch, dx):
    batch = center_batch(batch)
    L = torch.linalg.cross(batch.x, dx)
    L_total = scatter(L, batch.batch, reduce="add", dim=0)
    return L_total


def plot_tica(ax, tica, range=None):
    ax.hist2d(
        *tica.T,
        bins=[100, 100],
        cmap=plt.cm.viridis,
        norm=mpl.colors.LogNorm(),
        density=True,
        range=range,
    )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")


def sample_to_batch(sample):
    #  mol = sample["mol"]
    traj = torch.tensor(sample["traj"])[0]

    traj = center_traj(traj)
    traj = traj.reshape(-1, *traj.shape[-2:])

    node_type = torch.tensor(sample["node_type"])
    #  bond_index, bonds = get_bonds_from_rdkit(mol)

    data_list = [
        geom.data.Data(
            x=x, node_type=node_type, bond_type=sample["bond_type"], bond_index=sample["bond_index"]
        )
        for x in traj
    ]
    batch = geom.data.Batch.from_data_list(data_list)

    return batch


def get_sinusoids(dihedrals):
    cos = np.cos(dihedrals)
    sin = np.sin(dihedrals)

    sinusoids = np.concatenate((cos, sin), axis=-1)
    return sinusoids


def color_cycle():
    for c in plt.rcParams["axes.prop_cycle"]:
        yield c["color"]


def batch_to_device(batch, device):
    for key in batch.keys():
        if hasattr(batch[key], "to"):
            batch[key] = batch[key].to(device)
        if isinstance(batch[key], dict):
            batch[key] = batch_to_device(batch[key], device)

    return batch


def get_topology(atom_numbers):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = ELEMENTS[int(atom_number)]
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology


def get_topology_with_bonds(atom_numbers, bond_index):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    atom_indices = []
    for i, atom_num in enumerate(atom_numbers):
        e = ELEMENTS[int(atom_num)]
        atom = topology.add_atom(f"atom{i}", e, residue)
        atom_indices.append(atom)

    for atom1, atom2 in bond_index:
        if atom2 < atom1:
            continue
        topology.add_bond(atom_indices[atom1], atom_indices[atom2])

    return topology


def center_batch(batch):
    com = scatter(batch.x, batch.batch, dim=0, reduce="mean")
    batch.x = batch.x - com[batch.batch]
    return batch


def center_traj(traj):
    com = np.repeat(traj.mean(axis=1, keepdims=True), traj.shape[1], axis=1)
    traj -= com

    return traj


def center_coordinates(x):
    com = x.mean(dim=0, keepdim=True)
    return x - com


def sample_to_pdb(sample_path):  # pragma: no cover
    sample = md.load(sample_path)
    sample.save_pdb("/tmp/sample.pdb")


def get_bonds_from_topology(top):
    edge_index = []
    edge_attr = []

    for bond in top.bonds:
        i = bond.atom1.index
        j = bond.atom2.index

        bond_order = bond.order if bond.order else 1

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bond_order)
        edge_attr.append(bond_order)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return edge_index, edge_attr


def get_atoms_from_topology(top):
    return [a.element.atomic_number for a in top.atoms]


def get_data_from_pdb(path):
    if isinstance(path, str):
        traj = md.load(path)
    else:
        traj = path

    node_type = [a.element.atomic_number for a in traj.top.atoms]
    x = traj.xyz[0]
    edge_index, edge_type = get_bonds_from_topology(traj.top)

    return geom.data.Data(
        x=torch.tensor(x, dtype=torch.float32),
        node_type=torch.tensor(node_type, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
    )


def get_mdtraj(traj, atoms):
    topology = get_topology(atoms)
    traj = traj.reshape(-1, *traj.shape[-2:])
    traj = md.Trajectory(traj, topology)
    return traj


def get_simple_mdtraj(traj):
    atoms = [6] * traj.shape[-2]
    return get_mdtraj(traj, atoms)


def filter_trajs(trajs):
    mask = abs(trajs).max(axis=(1, 2, 3)) > 10
    mask += np.isnan(trajs).any(axis=(1, 2, 3))
    return trajs[~mask]


def add_feature_to_batch(batch, name, features):
    batch[name] = features[batch.batch]
    return batch


#def wasserstein_distance_circle(d1, d2):
#    try:
#        normalized_d1 = (d1 + np.pi) / (2 * np.pi)
#        normalized_d2 = (d2 + np.pi) / (2 * np.pi)
#        normalized_d = ot.wasserstein_circle(normalized_d1, normalized_d2)[0]
#        return normalized_d * 2 * np.pi
#    except ZeroDivisionError:
#        return None


def to_device(batch, device=None):
    if device is None:
        device = DEVICE
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        if isinstance(batch[key], dict):
            to_device(batch[key], device)


def get_atoms_from_rdkit(mol):
    return get_atoms_from_mol(mol)


def get_bonds_from_rdkit(mol):
    return get_bonds_from_mol(mol)


def get_atoms_from_mol(mol):
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def get_bonds_from_mol(mol):
    return get_bond_index_and_bonds(mol)


def pdb_to_mol(pdb, xyz=None):
    if xyz:
        pdb.xyz = xyz
    pdb.save("/tmp/traj.pdb")
    mol = Chem.MolFromPDBFile("/tmp/traj.pdb", removeHs=False, sanitize=True)

    return mol

def rdkit_to_mdtraj_topology(rdkit_mol): #move to utils
    top = md.Topology()
    atom_map = {}

    # Create one chain and residue (single molecule)
    chain = top.add_chain()
    residue = top.add_residue(name="MOL", chain=chain)

    # Add atoms
    for idx, atom in enumerate(rdkit_mol.GetAtoms()):
        element = atom.GetSymbol()
        md_atom = top.add_atom(name=element + str(idx), element=md.element.get_by_symbol(element), residue=residue)
        atom_map[atom.GetIdx()] = md_atom

    # Add bonds
    for bond in rdkit_mol.GetBonds():
        a1 = atom_map[bond.GetBeginAtomIdx()]
        a2 = atom_map[bond.GetEndAtomIdx()]
        top.add_bond(a1, a2)

    return top

def create_rdkit_mol(node_type, bond_index, bond_type):
    """
    Creates an RDKit molecule from NumPy arrays, automatically handling duplicate
    and direction-swapped bonds (e.g., [0, 1] and [1, 0]).

    Args:
        node_type (np.array): A 1D NumPy array of atomic numbers (e.g., [6, 6, 8])
                              or RDKit Atom objects.
        bond_index (np.array): A 2D NumPy array of shape (2, num_bonds) where the first
                               row specifies the indices of the first atoms in each bond
                               and the second row specifies the indices of the second atoms.
                               Example: np.array([[0, 1], [1, 2]]) for bonds 0-1 and 1-2.
        bond_type (np.array): A 1D NumPy array of RDKit bond types (Chem.BondType)
                              or integer representations of bond types.

    Returns:
        rdkit.Chem.rdchem.Mol: The created RDKit molecule object, or None if an error occurs.
    """
    if not (isinstance(node_type, np.ndarray) and
            isinstance(bond_index, np.ndarray) and
            isinstance(bond_type, np.ndarray)):
        print("Error: All inputs must be NumPy arrays.")
        return None

    if bond_index.shape[0] != 2:
        print("Error: bond_index array must have shape (2, num_bonds).")
        return None

    num_bonds = bond_index.shape[1]
    if num_bonds != bond_type.shape[0]:
        print("Error: The number of bonds in bond_index must match the number of bond types.")
        return None

    mol = Chem.RWMol()
    num_atoms = 0

    # 1. Add Atoms to the molecule
    try:
        for atom_info in node_type:
            if isinstance(atom_info, (int, np.integer)):
                mol.AddAtom(Chem.Atom(int(atom_info)))
            elif isinstance(atom_info, Chem.Atom):
                mol.AddAtom(atom_info)
            else:
                print(f"Error: Invalid node type in array: {atom_info}")
                return None
        num_atoms = mol.GetNumAtoms()
    except Exception as e:
        print(f"Error adding atoms: {e}")
        return None

    # 2. De-duplicate and standardize bonds before adding them
    unique_bonds = {}
    for i in range(num_bonds):
        atom1_idx = int(bond_index[0, i])
        atom2_idx = int(bond_index[1, i])
        b_type = bond_type[i]

        # Standardize the bond tuple to ensure consistency (e.g., (1,0) becomes (0,1))
        standard_bond_tuple = tuple(sorted((atom1_idx, atom2_idx)))

        # Handle both RDKit bond types and integer representations
        if isinstance(b_type, (int, np.integer)):
            b_type = AllChem.BondType.values.get(b_type, AllChem.BondType.UNSPECIFIED)
        elif not isinstance(b_type, Chem.BondType):
            print(f"Error: Invalid bond type at index {i}: {b_type}")
            return None

        # Store the unique bond and its type. If a duplicate exists, the last one wins.
        unique_bonds[standard_bond_tuple] = b_type
        
    # 3. Add Bonds to the molecule from the cleaned list
    try:
        for (atom1_idx, atom2_idx), b_type in unique_bonds.items():
            # Ensure indices are within a valid range
            if not (0 <= atom1_idx < num_atoms and 0 <= atom2_idx < num_atoms):
                print(f"Error: Bond indices {atom1_idx}-{atom2_idx} are out of range.")
                return None

            mol.AddBond(atom1_idx, atom2_idx, b_type)
    except Exception as e:
        print(f"Error adding bonds: {e}")
        return None

    # 4. Sanitize and return the molecule
    # try:
    #     Chem.SanitizeMol(mol)
    return mol
    # except Exception as e:
    #     print(f"Sanitization failed: {e}")
    #     return None


def kl_divergence(p, q):
    mask = p != 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def jensen_shannon_divergence(x, y):
    p, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=100, density=True)
    q, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=[x_edges, y_edges], density=True)

    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


def get_x_from_mol(mol):
    conformer = mol.GetConformer()
    coordinates = []
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(atom_idx)
        coordinates.append([pos.x, pos.y, pos.z])

    return torch.tensor(coordinates)


def mdtraj_to_mol(traj):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdb").name
    traj.save(tmp_file)
    mol = Chem.MolFromPDBFile(tmp_file, removeHs=False, sanitize=True)
    return mol


def get_bond_index_and_bonds_from_calpha_topology(top):
    assert len(list(top.residues)) == len(
        list(top.atoms)
    ), "there are more atoms than residues, so it doesn't look like a calpha topology"

    bond_index = np.array([[i, i + 1] for i in range(len(list(top.residues)) - 1)]).T  # type:ignore
    bond_type = np.array([1] * len(bond_index.T))

    bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
    bond_type = np.concatenate([bond_type, bond_type])

    return bond_index, bond_type


def get_ca_bonds(n):
    bonds = [[i, i + 1] for i in range(n - 1)]
    bond_index = torch.cat([torch.tensor(bonds).T, torch.tensor(bonds[::-1]).T], dim=1)
    bond_types = torch.ones_like(bond_index[0])
    return bond_index, bond_types


def load_binary_topology(topology):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as tmp:
        tmp.write(topology.decode())
        tmp.flush()
        return md.load_topology(tmp.name)


def get_bond_index_and_bonds(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromPDBFile(mol)

    bond_matrix = torch.tensor(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in mol.GetBonds()],
        dtype=torch.long,
    )
    if len(bond_matrix) == 0:
        bond_matrix = torch.zeros(0, 3, dtype=torch.long)

    bonds = bond_matrix[:, 2]
    edges = bond_matrix[:, :2].t().contiguous()
    src = torch.cat([edges[0], edges[1]], dim=0)
    dst = torch.cat([edges[1], edges[0]], dim=0)
    edge_index = torch.stack([src, dst], dim=0)

    bond_index = torch.cat((bonds, bonds))

    return edge_index, bond_index


def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_ref_path(path):
    model = path.split("results/")[-1].split("/")[0]
    ref_path = path.replace(model, "md")
    return ref_path


def get_out_path(path, name):
    path = path.replace(name, f"figures/{name}").replace(".pkl", ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def filter_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    clean_data = data[z_scores < threshold]
    return clean_data


# def setup_geometric_dataloader(dataset, batch_size=64, num_workers=8):
#     """Setup GeometricDataLoader with efficient collate function"""
#     collate_fn = create_threaded_collate_fn(dataset, threads=num_workers)
    
#     return GeometricDataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True,
#         persistent_workers=True
#     )

# data_set, epochs, learning_rate, n_features, n_model_layers,
#                      n_embedding_layers, max_lag, length_scale, n_reduced_features, uniform_lag, batch_size,  seed, multigpu, num_workers, no_ot

#best_loss = 1e10
# for epoch in range(1000):
#     for batch in dataloader:
#         t = torch.rand(len(batch['cond'])).type_as(batch['cond'].x)
#         loss = cfm.get_loss(t, batch)
#         cfm.training_step(loss)
#         step += 1
#         wandb.log({"step": step, "epoch": epoch, "loss": loss.item()})
        
#         print(
#             f"epoch: {epoch}, step: {step}, time passed: {(time.time()-start):.2f}, loss: {loss.item():.4f}",
#             end="\r",
#         )

#         if (step + 1) % 200 == 0:
            
#             mlops.save(cfm, f"results/model/model_{step}.pkl")
#             mlops.save(cfm, f"results/model/model_latest.pkl")
#             # mlops.save(samples, f"results/samples/samples_{step}.pkl")
#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             mlops.save(cfm, f"results/model/model_best.pkl")
                
# HINT using my implementation of the equivariant Readout and a painn model with 8 hidden features 
# I was able to get a loss of 0.037 in 10.000 steps within 20 minutes running locally on my laptop. 

#base_density = model.BaseDensity(std=dataset.scaling_factor)