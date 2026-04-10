"""Preprocessing pipeline for molecular graphs.

Converts SMILES strings into PyTorch Geometric Data objects with:
- Node features: atom-level properties (atomic number, degree, etc.)
- Edge features: bond-level properties (bond type, stereo, etc.)
- Graph-level labels: molecular property targets

Also provides scaffold-based and random splitting utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.utils import subgraph

# Suppress noisy RDKit warnings (invalid SMILES, valence errors)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _process_mol_row_simple(args):
    """Worker for simple 2D SMILES-to-graph conversion.
    
    Must be at top-level for multiprocessing pickling.
    """
    idx, smiles, targets = args
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None, None, None
        
    y = np.array(targets, dtype=np.float32)
    # This worker is primarily used for the 2D preprocess_moleculenet function
    data = smiles_to_graph(smiles, y=y)
    
    if data is not None and data.x.shape[0] > 0:
        return data, smiles, idx
    return None, None, None


class MoleculeDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset for faster batching.
    
    Compiles individual Data objects into a contiguous memory block.
    """
    def __init__(self, data_list: list[Data] | None = None):
        super().__init__(None)
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)


# ── Atom (Node) Featurization ────────────────────────────────────────────

# Allowlists for one-hot encoding
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
HYBRIDIZATION_LIST = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
FORMAL_CHARGE_LIST = [-2, -1, 0, 1, 2]
NUM_H_LIST = [0, 1, 2, 3, 4]


def one_hot(value, allowlist: list) -> list[float]:
    """One-hot encode a value against an allowlist, with 'other' bin."""
    encoding = [0.0] * (len(allowlist) + 1)
    if value in allowlist:
        encoding[allowlist.index(value)] = 1.0
    else:
        encoding[-1] = 1.0  # 'other' category
    return encoding


def atom_features(atom) -> list[float]:
    """Extract a feature vector for a single atom.

    Features (total ~39 dimensions):
    - Atomic number (one-hot, 10 dims)
    - Degree (one-hot, 7 dims)
    - Formal charge (one-hot, 6 dims)
    - Hybridization (one-hot, 6 dims)
    - Num hydrogens (one-hot, 6 dims)
    - Aromaticity (1 dim)
    - In ring (1 dim)
    - Atomic mass (1 dim, scaled)
    """
    from rdkit.Chem import Atom, HybridizationType

    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_LIST))
    features.extend(one_hot(atom.GetTotalDegree(), DEGREE_LIST))
    features.extend(one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_LIST))
    features.extend(one_hot(str(atom.GetHybridization()), HYBRIDIZATION_LIST))
    features.extend(one_hot(atom.GetTotalNumHs(), NUM_H_LIST))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    features.append(atom.GetMass() / 100.0)  # Scaled atomic mass

    return features


def get_node_feature_dim() -> int:
    """Return the dimensionality of atom features."""
    return (
        len(ATOM_LIST) + 1      # atomic number
        + len(DEGREE_LIST) + 1  # degree
        + len(FORMAL_CHARGE_LIST) + 1  # formal charge
        + len(HYBRIDIZATION_LIST) + 1  # hybridization
        + len(NUM_H_LIST) + 1   # num H
        + 1  # aromaticity
        + 1  # in ring
        + 1  # mass
    )


# ── Bond (Edge) Featurization ────────────────────────────────────────────

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
STEREO_LIST = ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]


def bond_features(bond) -> list[float]:
    """Extract a feature vector for a single bond.

    Features (total ~12 dimensions):
    - Bond type (one-hot, 5 dims)
    - Stereo (one-hot, 6 dims)
    - Conjugated (1 dim)
    - In ring (1 dim)
    """
    features = []
    features.extend(one_hot(str(bond.GetBondType()).split(".")[-1], BOND_TYPES))
    features.extend(one_hot(str(bond.GetStereo()), STEREO_LIST))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)

    return features


def get_edge_feature_dim() -> int:
    """Return the dimensionality of bond features."""
    return (
        len(BOND_TYPES) + 1     # bond type
        + len(STEREO_LIST) + 1  # stereo
        + 1  # conjugated
        + 1  # in ring
    )


# ── Bond Type to Relation Index (for RGCN) ──────────────────────────────

BOND_TYPE_TO_REL = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
}


def get_bond_relation(bond) -> int:
    """Map a bond to a relation index for relational GNNs."""
    bt = str(bond.GetBondType()).split(".")[-1]
    return BOND_TYPE_TO_REL.get(bt, 0)


# ── SMILES → Graph Conversion ───────────────────────────────────────────

def smiles_to_graph(
    smiles: str,
    y: np.ndarray | None = None,
) -> Data | None:
    """Convert a SMILES string to a PyG Data object."""
    raw = smiles_to_graph_dict(smiles, y)
    if raw is None:
        return None
    
    data = Data(
        x=torch.from_numpy(raw["x"]),
        edge_index=torch.from_numpy(raw["edge_index"]),
        edge_attr=torch.from_numpy(raw["edge_attr"]),
    )
    data.edge_type = torch.from_numpy(raw["edge_type"])
    data.smiles = raw["smiles"]
    if raw["y"] is not None:
        data.y = torch.from_numpy(raw["y"]).unsqueeze(0)
    
    return data


def smiles_to_graph_dict(
    smiles: str,
    y: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Convert a SMILES string to a raw dictionary of NumPy arrays.
    
    This avoids PyTorch's shared memory (mmap) overhead during multiprocessing,
    as NumPy objects are pickled and passed more reliably for massive datasets.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    node_feats = np.array(node_feats, dtype=np.float32)

    # Edge index and features (undirected: add both directions)
    if mol.GetNumBonds() == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, get_edge_feature_dim()), dtype=np.float32)
        edge_type = np.zeros(0, dtype=np.int64)
    else:
        src_list, dst_list = [], []
        edge_feats = []
        edge_types = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_features(bond)
            rel = get_bond_relation(bond)

            # Add both directions for undirected graph
            src_list.extend([i, j])
            dst_list.extend([j, i])
            edge_feats.extend([bf, bf])
            edge_types.extend([rel, rel])

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(edge_feats, dtype=np.float32)
        edge_type = np.array(edge_types, dtype=np.int64)

    return {
        "x": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_type": edge_type,
        "smiles": smiles,
        "y": y
    }




# ── Molecular Fingerprints (for tabular baselines) ──────────────────────

def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Compute Morgan (ECFP) fingerprint for a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation.
    radius : int
        Morgan fingerprint radius (2 = ECFP4).
    n_bits : int
        Length of the bit vector.

    Returns
    -------
    np.ndarray or None
        Binary fingerprint vector, or None if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def compute_descriptors(smiles: str) -> np.ndarray | None:
    """Compute RDKit 2D molecular descriptors.

    Returns a vector of common physicochemical descriptors:
    [MolWt, LogP, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds,
     NumAromaticRings, FractionCSP3, HeavyAtomCount, RingCount]
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Lipinski.FractionCSP3(mol),
        Lipinski.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
    ]
    return np.array(desc, dtype=np.float32)


# ── Scaffold Splitting ──────────────────────────────────────────────────

def generate_scaffold(smiles: str) -> str:
    """Generate the Murcko scaffold for a molecule."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    return scaffold


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    **kwargs
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic split by Murcko scaffold.
    
    1. Group molecules by scaffold.
    2. Sort scaffold groups by size in descending order (largest first).
    3. Sequentially fill splits until quotas are met.
    
    This ensures that common scaffolds stay in the training set and the
    test set contains the most 'unique' structural variations.
    """
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        try:
            scaffold = generate_scaffold(smi)
        except Exception:
            scaffold = smi  # Fallback for complex errors
        scaffolds[scaffold].append(idx)

    # Deterministic Sort: Sort by scaffold group size descending
    # Then sort by scaffold string to break ties deterministically
    scaffold_sets = sorted(
        scaffolds.values(), 
        key=lambda x: (len(x), smiles_list[x[0]]), 
        reverse=True
    )

    n_total = len(smiles_list)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    train_indices, val_indices, test_indices = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= n_train:
            train_indices.extend(scaffold_set)
        elif len(val_indices) + len(scaffold_set) <= n_val:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # Validate output - no silent fallback
    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Strict scaffold split failed to produce three splits. "
            f"Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}. "
            "The dataset may be too small or contains too few unique scaffolds."
        )

    logger.info(
        "Deterministic Scaffold split: train=%d, val=%d, test=%d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    return train_indices, val_indices, test_indices


def random_split(
    n: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    **kwargs
) -> tuple[list[int], list[int], list[int]]:
    """Standard random split with strict seed-based reproducibility."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()

    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Validate output
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Random split produced empty split(s). Dataset too small.")

    logger.info(
        "Random split: train=%d, val=%d, test=%d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    return train_indices, val_indices, test_indices


def stratified_scaffold_split(
    smiles_list: list[str],
    y: np.ndarray,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    **kwargs
) -> tuple[list[int], list[int], list[int]]:
    """Balanced scaffold split for imbalanced datasets (Tox21/HIV).
    
    1. Group by scaffold.
    2. Calculate positive ratio for each scaffold bucket.
    3. Use a greedy iterative approach to place buckets into splits, 
       maintaining the global active-label ratio across segments.
    """
    from collections import defaultdict
    
    # 1. Group by scaffold
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        try:
            scaffold = generate_scaffold(smi)
        except Exception:
            scaffold = smi
        scaffolds[scaffold].append(idx)
        
    # 2. Calculate stats for each bucket
    # For multi-task, we use the mean positive ratio as the balancing heuristic
    bucket_stats = []
    global_pos_ratio = np.nanmean(y)
    
    for scaffold, indices in scaffolds.items():
        bucket_y = y[indices]
        pos_ratio = np.nanmean(bucket_y) if not np.all(np.isnan(bucket_y)) else global_pos_ratio
        bucket_stats.append({
            "indices": indices,
            "pos_ratio": pos_ratio,
            "size": len(indices)
        })
        
    # Sort buckets by size descending to handle large ones first
    bucket_stats.sort(key=lambda x: x["size"], reverse=True)
    
    # 3. Greedy allocation
    train_indices, val_indices, test_indices = [], [], []
    train_pos, val_pos, test_pos = 0.0, 0.0, 0.0
    
    n_total = len(smiles_list)
    target_counts = {
        "train": int(n_total * frac_train),
        "val": int(n_total * frac_val),
        "test": n_total - int(n_total * frac_train) - int(n_total * frac_val)
    }

    def get_pos_count(indices):
        bucket_y = y[indices]
        return np.nansum(bucket_y)
        
    def get_ratio(current_size, current_pos):
        return current_pos / current_size if current_size > 0 else 0.0

    for bucket in bucket_stats:
        indices = bucket["indices"]
        pos_count = get_pos_count(indices)
        size = len(indices)
        
        s_train = target_counts["train"] - len(train_indices)
        s_val = target_counts["val"] - len(val_indices)
        s_test = target_counts["test"] - len(test_indices)

        can_train = s_train >= size
        can_val = s_val >= size
        can_test = s_test >= size

        if not (can_train or can_val or can_test):
            rem_space = {"train": s_train, "val": s_val, "test": s_test}
            best_split = max(rem_space, key=rem_space.get)
            if best_split == "train": can_train = True
            elif best_split == "val": can_val = True
            else: can_test = True

        cands = []
        if can_train: cands.append(("train", get_ratio(len(train_indices), train_pos), target_counts["train"]))
        if can_val: cands.append(("val", get_ratio(len(val_indices), val_pos), target_counts["val"]))
        if can_test: cands.append(("test", get_ratio(len(test_indices), test_pos), target_counts["test"]))
        
        if pos_count > 0:
            best_split = min(cands, key=lambda x: (x[1], -x[2]))[0]
        else:
            best_split = max(cands, key=lambda x: (x[1], x[2]))[0]

        if best_split == "train":
            train_indices.extend(indices); train_pos += pos_count
        elif best_split == "val":
            val_indices.extend(indices); val_pos += pos_count
        else:
            test_indices.extend(indices); test_pos += pos_count

    logger.info(
        "Stratified Scaffold split (Pos Ratios - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%)",
        (train_pos / len(train_indices)) * 100,
        (val_pos / len(val_indices)) * 100,
        (test_pos / len(test_indices)) * 100
    )
    
    return train_indices, val_indices, test_indices


def _fast_sparse_butina(fps: list, sim_cutoff: float = 0.4) -> list[list[int]]:
    """Memory-efficient, highly optimized Butina clustering.
    
    Bypasses the dense O(N^2) distance matrix memory leak by using an 
    adjacency list and utilizes RDKit's C++ BulkTanimoto for speed.
    """
    from rdkit import DataStructs
    from tqdm import tqdm
    
    n = len(fps)
    neighbors = [None] * n
    
    # 1. Build Sparse Adjacency List (Fast C++ Comparisons)
    for i in tqdm(range(n), desc="Building Sparse Neighbors"):
        # BulkTanimoto compares 1 fingerprint against ALL others instantly
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        
        # Only store indices where similarity >= cutoff (exclude self)
        # Using list comprehension for speed, filter only on what we need
        neighbors[i] = [j for j, sim in enumerate(sims) if sim >= sim_cutoff and j != i]
        
    # 2. Sort by neighbor density (descending) 
    # Tie-breaker: sort by index to ensure deterministic output (lower index first)
    nodes = sorted(
        [(len(nbrs), i) for i, nbrs in enumerate(neighbors)], 
        key=lambda x: (x[0], -x[1]), 
        reverse=True
    )
    
    # 3. Greedy Clustering (The Butina Algorithm)
    clusters = []
    assigned = set()
    
    for _, i in nodes:
        if i in assigned:
            continue
            
        # Initialize new cluster with the centroid node
        cluster = [i]
        assigned.add(i)
        
        # Absorb unassigned neighbors
        for nbr in neighbors[i]:
            if nbr not in assigned:
                cluster.append(nbr)
                assigned.add(nbr)
                
        clusters.append(cluster)
        
    return clusters


def butina_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    similarity_cutoff: float = 0.4,
    **kwargs
) -> tuple[list[int], list[int], list[int]]:
    """Clustering-based split using Butina algorithm (Structural Exclusion).
    
    Guarantees that molecules in different splits are structurally distinct.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    logger.info("Starting Fast Butina Clustering (N=%d, cutoff=%.1f)...", len(smiles_list), similarity_cutoff)
    
    # 1. Compute fingerprints
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    
    # 2. Parallel Neighbor Clustering
    cluster_indices = _fast_sparse_butina(fps, similarity_cutoff)
    
    # Sort clusters by size descending (largest clusters are 'hubs')
    cluster_indices.sort(key=len, reverse=True)
    
    # 3. Assign clusters to splits
    train_indices, val_indices, test_indices = [], [], []
    n_mols = len(fps)
    n_train = int(n_mols * frac_train)
    n_val = int(n_mols * frac_val)
    
    for cluster in cluster_indices:
        if len(train_indices) + len(cluster) <= n_train:
            train_indices.extend(cluster)
        elif len(val_indices) + len(cluster) <= n_val:
            val_indices.extend(cluster)
        else:
            test_indices.extend(cluster)
            
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Butina split failed to produce three splits. Similarity cutoff might be too high.")
            
    logger.info(
        "Butina split: train=%d, val=%d, test=%d (%d clusters total)",
        len(train_indices), len(val_indices), len(test_indices), len(cluster_indices)
    )
    
    return train_indices, val_indices, test_indices


def stratified_butina_split(
    smiles_list: list[str],
    y: np.ndarray,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    similarity_cutoff: float = 0.4,
    **kwargs
) -> tuple[list[int], list[int], list[int]]:
    """Master split: Structural Exclusion + Stratified Label Distribution.
    
    1. Clusters molecules via Butina (Tanimoto).
    2. Calculates size and positive ratio for each cluster.
    3. Greedily fills splits by prioritizing the bucket with the largest 
       deficit in positive labels while respecting size quotas.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    from tqdm import tqdm
    
    logger.info("Starting Stratified Butina Split (N=%d, cutoff=%.1f)...", len(smiles_list), similarity_cutoff)
    
    # 1. Cluster Generation
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in tqdm(mols, desc="Computing Fingerprints")]
    cluster_indices = _fast_sparse_butina(fps, similarity_cutoff)
    
    # 2. Cluster Profiling
    global_pos_ratio = np.nanmean(y)
    cluster_stats = []
    for cluster in cluster_indices:
        cluster_indices = list(cluster)
        cluster_y = y[cluster_indices]
        
        # Calculate positive count across tasks (mean as heuristic)
        pos_count = np.nansum(cluster_y)
        cluster_stats.append({
            "indices": cluster_indices,
            "size": len(cluster_indices),
            "pos_count": pos_count,
            "pos_ratio": np.nanmean(cluster_y) if not np.all(np.isnan(cluster_y)) else global_pos_ratio
        })
        
    # Sort clusters by size descending (largest pieces first)
    cluster_stats.sort(key=lambda x: x["size"], reverse=True)
    
    # 3. Greedy allocation with deficit tracking
    train_indices, val_indices, test_indices = [], [], []
    train_pos, val_pos, test_pos = 0.0, 0.0, 0.0
    
    n_mols = len(fps)
    target_counts = {
        "train": int(n_mols * frac_train),
        "val": int(n_mols * frac_val),
        "test": n_mols - int(n_mols * frac_train) - int(n_mols * frac_val)
    }

    def get_ratio(current_size, current_pos):
        return current_pos / current_size if current_size > 0 else 0.0

    for cluster in tqdm(cluster_stats, desc="Allocating Clusters"):
        indices = cluster["indices"]
        pos_count = cluster["pos_count"]
        size = cluster["size"]
        
        s_train = target_counts["train"] - len(train_indices)
        s_val = target_counts["val"] - len(val_indices)
        s_test = target_counts["test"] - len(test_indices)

        can_train = s_train >= size
        can_val = s_val >= size
        can_test = s_test >= size

        if not (can_train or can_val or can_test):
            rem_space = {"train": s_train, "val": s_val, "test": s_test}
            best_split = max(rem_space, key=rem_space.get)
            if best_split == "train": can_train = True
            elif best_split == "val": can_val = True
            else: can_test = True

        cands = []
        if can_train: cands.append(("train", get_ratio(len(train_indices), train_pos), target_counts["train"]))
        if can_val: cands.append(("val", get_ratio(len(val_indices), val_pos), target_counts["val"]))
        if can_test: cands.append(("test", get_ratio(len(test_indices), test_pos), target_counts["test"]))
        
        if pos_count > 0:
            best_split = min(cands, key=lambda x: (x[1], -x[2]))[0]
        else:
            best_split = max(cands, key=lambda x: (x[1], x[2]))[0]

        if best_split == "train":
            train_indices.extend(indices); train_pos += pos_count
        elif best_split == "val":
            val_indices.extend(indices); val_pos += pos_count
        else:
            test_indices.extend(indices); test_pos += pos_count

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Stratified Butina split failed. Try lower similarity_cutoff.")

    logger.info(
        "Stratified Butina Split (Pos Ratios - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%)",
        (train_pos / len(train_indices)) * 100,
        (val_pos / len(val_indices)) * 100,
        (test_pos / len(test_indices)) * 100
    )
    
    return train_indices, val_indices, test_indices


# ── Full Preprocessing Pipeline ─────────────────────────────────────────

def preprocess_moleculenet(
    csv_path: str | Path,
    config: dict[str, Any],
    cache_path: str | Path | None = None,
) -> tuple[MoleculeDataset, list[int], list[int], list[int]]:
    """Full preprocessing pipeline: load CSV → convert to graphs → split.

    Parameters
    ----------
    csv_path : str or Path
        Path to a MoleculeNet CSV file.
    config : dict
        Configuration dictionary with data settings.
    cache_path : str or Path, optional
        Path to save/load processed data.

    Returns
    -------
    (dataset, train_idx, val_idx, test_idx)
    """
    from tqdm import tqdm

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached processed data from %s", cache_path)
            # weights_only=False is safe for our internal Data objects
            return torch.load(cache_path, weights_only=False)

    data_cfg = config.get("data", config)
    dataset_name = data_cfg.get("dataset_name", "bbbp").lower()

    # Resolve column names from metadata
    from mol_prop_gnn.data.download import get_dataset_info
    info = get_dataset_info(dataset_name)
    smiles_col = info["smiles_col"]
    target_cols = info["target_cols"]
    task_type = info["task_type"]

    logger.info("Loading %s from %s ...", dataset_name.upper(), csv_path)
    df = pd.read_csv(csv_path)

    # Validate columns exist
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found. Columns: {list(df.columns)}")

    available_targets = [c for c in target_cols if c in df.columns]
    if not available_targets:
        raise ValueError(f"No target columns found. Expected: {target_cols}")

    logger.info(
        "Dataset: %d molecules, %d tasks (%s)",
        len(df), len(available_targets), task_type,
    )

    # Convert SMILES to graphs
    graphs = []
    valid_smiles = []
    valid_indices = []

    # 1. Prepare tasks using fast zip iteration
    tasks = []
    for idx, (smiles, *target_vals) in enumerate(zip(df[smiles_col], *[df[c] for c in available_targets])):
        targets = [float("nan") if pd.isna(v) else float(v) for v in target_vals]
        tasks.append((idx, smiles, targets))

    # 2. Parallel Processing
    n_workers = min(multiprocessing.cpu_count(), len(tasks))
    if n_workers > 1:
        logger.info("Starting parallel graph conversion across %d cores (chunksize=100)...", n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # chunksize=100 batches jobs for reduced IPC overhead
            results = list(tqdm(
                executor.map(_process_mol_row_simple, tasks, chunksize=100),
                total=len(tasks),
                desc="Converting SMILES"
            ))
            
            for data, sm, i in results:
                if data is not None:
                    graphs.append(data)
                    valid_smiles.append(sm)
                    valid_indices.append(i)
    else:
        # Fallback
        for task in tqdm(tasks, desc="Converting SMILES"):
            data, sm, i = _process_mol_row_simple(task)
            if data is not None:
                graphs.append(data)
                valid_smiles.append(sm)
                valid_indices.append(idx)

    logger.info(
        "Successfully converted %d / %d molecules to graphs",
        len(graphs), len(df),
    )

    # Split
    split_type = data_cfg.get("split_type", "scaffold").lower()
    seed = config.get("training", {}).get("seed", 42)
    frac_train = data_cfg.get("frac_train", 0.8)
    frac_val = data_cfg.get("frac_val", 0.1)
    frac_test = data_cfg.get("frac_test", 0.1)
    
    # Extract labels for stratified splitting (needed for imbalance handling)
    all_y = np.array([g.y.numpy().flatten() for g in graphs])

    split_fns = {
        "random": random_split,
        "scaffold": scaffold_split,
        "stratified_scaffold": stratified_scaffold_split,
        "butina": butina_split,
        "stratified_butina": stratified_butina_split
    }
    
    if split_type not in split_fns:
        raise ValueError(f"Unknown split_type: {split_type}. Choices: {list(split_fns.keys())}")
        
    split_fn = split_fns[split_type]
    train_idx, val_idx, test_idx = split_fn(
        smiles_list=valid_smiles,
        y=all_y,
        frac_train=frac_train,
        frac_val=frac_val,
        frac_test=frac_test,
        seed=seed,
        similarity_cutoff=data_cfg.get("similarity_cutoff", 0.4)
    )

    # Convert to contiguous InMemoryDataset
    dataset = MoleculeDataset(graphs)
    output = (dataset, train_idx, val_idx, test_idx)

    # Cache if path provided
    if cache_path is not None:
        logger.info("Saving processed data to cache: %s", cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, cache_path)

    return output
