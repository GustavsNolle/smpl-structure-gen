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
from torch_geometric.data import Data

# Suppress noisy RDKit warnings (invalid SMILES, valence errors)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


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
    """Convert a SMILES string to a PyG Data object.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    y : np.ndarray, optional
        Target label(s) for the molecule.

    Returns
    -------
    Data or None
        PyG Data object, or None if the SMILES is invalid.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Edge index and features (undirected: add both directions)
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_edge_feature_dim()), dtype=torch.float32)
        edge_type = torch.zeros(0, dtype=torch.long)
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

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    data.smiles = smiles

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    return data


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
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split molecule indices by Bemis-Murcko scaffold.

    Ensures that molecules with the same scaffold are in the same split,
    providing a more realistic evaluation of generalization.

    Uses randomized scaffold ordering (standard DeepChem approach) to
    avoid creating splits with extreme class imbalance. Falls back to
    random split if any split ends up empty (can happen with small datasets).

    Returns
    -------
    (train_indices, val_indices, test_indices)
    """
    from collections import defaultdict

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        try:
            scaffold = generate_scaffold(smi)
        except Exception:
            scaffold = smi  # Fallback: treat as unique scaffold
        scaffolds[scaffold].append(idx)

    # Randomize scaffold ordering to avoid systematic class imbalance
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

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

    # Fallback: if any split is empty, use random split instead
    if not train_indices or not val_indices or not test_indices:
        logger.warning(
            "Scaffold split produced empty split(s) (train=%d, val=%d, test=%d). "
            "Falling back to random split.",
            len(train_indices), len(val_indices), len(test_indices),
        )
        return random_split(n_total, frac_train, frac_val, frac_test, seed)

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    logger.info(
        "Scaffold split: train=%d, val=%d, test=%d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    return train_indices, val_indices, test_indices


def random_split(
    n: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Simple random split of indices."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()

    n_train = int(n * frac_train)
    n_val = int(n * frac_val)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    logger.info(
        "Random split: train=%d, val=%d, test=%d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    return train_indices, val_indices, test_indices


# ── Full Preprocessing Pipeline ─────────────────────────────────────────

def preprocess_moleculenet(
    csv_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[Data], list[int], list[int], list[int]]:
    """Full preprocessing pipeline: load CSV → convert to graphs → split.

    Parameters
    ----------
    csv_path : str or Path
        Path to a MoleculeNet CSV file.
    config : dict
        Configuration dictionary with data settings.

    Returns
    -------
    (graphs, train_idx, val_idx, test_idx)
    """
    from tqdm import tqdm

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

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting SMILES"):
        smiles = row[smiles_col]
        if not isinstance(smiles, str) or len(smiles) == 0:
            continue

        targets = []
        for col in available_targets:
            val = row[col]
            if pd.isna(val):
                targets.append(float("nan"))
            else:
                targets.append(float(val))
        y = np.array(targets, dtype=np.float32)

        data = smiles_to_graph(smiles, y=y)
        if data is not None and data.x.shape[0] > 0:
            graphs.append(data)
            valid_smiles.append(smiles)
            valid_indices.append(idx)

    logger.info(
        "Successfully converted %d / %d molecules to graphs",
        len(graphs), len(df),
    )

    # Split
    split_type = data_cfg.get("split_type", "scaffold")
    seed = config.get("training", {}).get("seed", 42)
    frac_train = data_cfg.get("frac_train", 0.8)
    frac_val = data_cfg.get("frac_val", 0.1)
    frac_test = data_cfg.get("frac_test", 0.1)

    if split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            valid_smiles, frac_train, frac_val, frac_test, seed,
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(graphs), frac_train, frac_val, frac_test, seed,
        )

    return graphs, train_idx, val_idx, test_idx
