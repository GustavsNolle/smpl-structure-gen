"""Unified dataset preprocessing for multi-task semi-supervised training.

Merges multiple MoleculeNet datasets using exact SMILES matches.
Missing labels are filled with NaNs to be masked during training.
Regression targets are optionally standard-scaled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mol_prop_gnn.data.download import download_moleculenet, get_dataset_info
from mol_prop_gnn.data.preprocessing import smiles_to_graph, scaffold_split
from mol_prop_gnn.data.dataset import MoleculeDataModule

logger = logging.getLogger(__name__)

# The fixed set of datasets we are combining for our 5-dimensional map
UNIFIED_DATASETS = ["bbbp", "esol", "bace", "freesolv", "lipophilicity"]


def build_unified_dataframe(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, dict[str, dict]]:
    """Download and merge multiple datasets into a single DataFrame.

    Returns
    -------
    merged_df : pd.DataFrame
        DataFrame with a 'smiles' column and one column per dataset target.
    scaling_stats : dict
        Mean and std for each regression dataset (to unscale predictions later).
    """
    raw_dir = Path(raw_dir)
    merged_df = None
    scaling_stats = {}

    for ds_name in tqdm(UNIFIED_DATASETS, desc="Building unified dataset"):
        csv_path = download_moleculenet(ds_name, raw_dir=raw_dir)
        info = get_dataset_info(ds_name)
        
        df = pd.read_csv(csv_path)
        smiles_col = info["smiles_col"]
        target_col = info["target_cols"][0]  # We assume single-task for these 5 datasets
        task_type = info["task_type"]

        # Drop rows with missing SMILES or target
        df = df.dropna(subset=[smiles_col, target_col]).copy()
        
        # Rename columns to standard names
        df = df.rename(columns={smiles_col: "smiles", target_col: ds_name})
        df = df[["smiles", ds_name]]
        
        # Keep first duplicate SMILES if any
        df = df.drop_duplicates(subset=["smiles"], keep="first")

        # Standard scale regression targets
        if task_type == "regression":
            vals = df[ds_name].values
            mean, std = vals.mean(), vals.std()
            scaling_stats[ds_name] = {"mean": mean, "std": std}
            df[ds_name] = (df[ds_name] - mean) / (std + 1e-8)
            logger.info("Scaled %s: mean=%.3f, std=%.3f", ds_name, mean, std)
        
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="smiles", how="outer")

    # The resulting merged_df has 'smiles' and one column for each of the 5 datasets
    # Missing tasks for a given molecule will naturally be NaN in Pandas
    logger.info(
        "Built unified dataset: %d total unique SMILES across %d datasets",
        len(merged_df), len(UNIFIED_DATASETS)
    )
    
    return merged_df, scaling_stats


def preprocess_unified_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> tuple[list[Any], list[int], list[int], list[int]]:
    """Convert the unified DataFrame into PyTorch Geometric Data objects.

    Targets are aligned to the order of UNIFIED_DATASETS.
    """
    graphs = []
    valid_smiles = []
    
    # We enforce a strict order of targets in the y-vector
    target_cols = UNIFIED_DATASETS

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting unified SMILES"):
        smiles = row["smiles"]
        if not isinstance(smiles, str) or len(smiles) == 0:
            continue
            
        targets = []
        for col in target_cols:
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

    logger.info("Successfully converted %d / %d molecules", len(graphs), len(df))
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        valid_smiles, frac_train, frac_val, frac_test, seed
    )
    
    return graphs, train_idx, val_idx, test_idx


def get_task_types() -> list[str]:
    """Return the task type ('classification' or 'regression') for each dimension."""
    task_types = []
    for ds_name in UNIFIED_DATASETS:
        info = get_dataset_info(ds_name)
        task_types.append(info["task_type"])
    return task_types
