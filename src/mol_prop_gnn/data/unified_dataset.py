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
import hashlib
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from mol_prop_gnn.data.download import download_moleculenet, get_dataset_info
from mol_prop_gnn.data.preprocessing import (
    smiles_to_graph, scaffold_split, random_split, 
    stratified_scaffold_split, butina_split, 
    stratified_butina_split, MoleculeDataset
)
from mol_prop_gnn.data.dataset import MoleculeDataModule

logger = logging.getLogger(__name__)

# Default set of datasets for the 5-dimensional map
DEFAULT_DATASETS = ["bbbp", "esol", "bace", "freesolv", "lipophilicity"]


def _process_mol_task(args):
    """Worker function for multiprocessing graph conversion using NumPy dicts."""
    idx, smiles, y = args
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None, None, None
        
    from mol_prop_gnn.data.preprocessing import smiles_to_graph_dict
    raw = smiles_to_graph_dict(smiles, y=y)
        
    if raw is not None:
        return raw, smiles, idx
    return None, None, None


def build_unified_dataframe(
    dataset_names: list[str] | None = None,
    raw_dir: str | Path = "data/raw"
) -> tuple[pd.DataFrame, dict[str, dict], list[str], list[str], dict[str, str]]:
    """Download and merge multiple datasets into a single DataFrame.

    Returns
    -------
    merged_df : pd.DataFrame
        DataFrame with a 'smiles' column and one column per dataset target.
    scaling_stats : dict
        Mean and std for each regression dataset.
    target_names : list[str]
        Ordered list of target column names in the merged DataFrame.
    task_types : list[str]
        Ordered list of task types ('classification' or 'regression') for each target.
    """
    dataset_names = dataset_names or DEFAULT_DATASETS
    raw_dir = Path(raw_dir)
    merged_df = None
    scaling_stats = {}
    target_names = []
    task_types = []
    target_to_ds = {}

    for ds_name in tqdm(dataset_names, desc="Building unified dataset"):
        csv_path = download_moleculenet(ds_name, raw_dir=raw_dir)
        info = get_dataset_info(ds_name)
        
        df = pd.read_csv(csv_path)
        smiles_col = info["smiles_col"]
        ds_targets = info["target_cols"]
        ds_task_type = info["task_type"]

        # Drop rows with missing SMILES
        df = df.dropna(subset=[smiles_col]).copy()
        
        # Rename SMILES column to standard name
        df = df.rename(columns={smiles_col: "smiles"})
        
        # Keep only SMILES and original targets
        subset_cols = ["smiles"] + ds_targets
        df = df[subset_cols]
        
        # Keep first duplicate SMILES if any
        df = df.drop_duplicates(subset=["smiles"], keep="first")

        # Standard scale regression targets and track target names/types
        for target in ds_targets:
            # For single-task datasets, we use the dataset name for clarity in metrics
            clean_name = ds_name if len(ds_targets) == 1 else target
            
            if ds_task_type == "regression":
                vals = df[target].values
                # We only scale if there are valid values
                valid_vals = vals[~pd.isna(vals)]
                if len(valid_vals) > 0:
                    mean, std = valid_vals.mean(), valid_vals.std()
                    scaling_stats[clean_name] = {"mean": mean, "std": std}
                    df[target] = (df[target] - mean) / (std + 1e-8)
                    logger.info("Scaled %s (as %s): mean=%.3f, std=%.3f", target, clean_name, mean, std)
            
            # Map the original column name to our clean name in the dataframe
            if target != clean_name:
                df = df.rename(columns={target: clean_name})
            
            target_names.append(clean_name)
            task_types.append(ds_task_type)
            target_to_ds[clean_name] = ds_name
        
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="smiles", how="outer")

    # The resulting merged_df has 'smiles' and one column for each of the 5 datasets
    # Missing tasks for a given molecule will naturally be NaN in Pandas
    logger.info(
        "Built unified dataset: %d total unique SMILES across %d tasks (%d datasets)",
        len(merged_df), len(target_names), len(dataset_names)
    )
    
    return merged_df, scaling_stats, target_names, task_types, target_to_ds


def preprocess_unified_dataset(
    df: pd.DataFrame,
    target_names: list[str],
    seed: int = 42,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    split_type: str = "stratified_butina",
    similarity_cutoff: float = 0.4,
    cache_dir: str | Path | None = "data/processed",
) -> tuple[MoleculeDataset, list[int], list[int], list[int]]:
    """Convert the unified DataFrame into PyTorch Geometric Data objects.

    Targets are aligned to the order of target_names.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        # Create a stable hash based on the target names and split type to avoid cache collisions
        task_hash = hashlib.md5("".join(sorted(target_names)).encode()).hexdigest()[:10]
        cache_path = cache_dir / f"unified_{len(df)}_{task_hash}_{split_type}_cut{similarity_cutoff}.pt"
        
        if cache_path.exists():
            logger.info("Loading cached unified dataset from %s", cache_path)
            # weights_only=False is safe for our internal objects
            return torch.load(cache_path, weights_only=False)

    graphs = []
    valid_smiles = []
    
    # 1. Prepare tasks using fast zip iteration
    target_cols = target_names

    # 1. Prepare tasks using fast zip iteration
    tasks = []
    for idx, (smiles, *target_vals) in enumerate(zip(df["smiles"], *[df[c] for c in target_cols])):
        y = np.array([float(v) if not pd.isna(v) else float("nan") for v in target_vals], dtype=np.float32)
        tasks.append((idx, smiles, y))

    # 2. Parallel Processing
    n_workers = min(multiprocessing.cpu_count(), len(tasks))
    if n_workers > 1:
        logger.info("Starting parallel graph conversion across %d cores (chunksize=100)...", n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # batching jobs for reduced IPC overhead
            results = list(tqdm(
                executor.map(_process_mol_task, tasks, chunksize=100),
                total=len(tasks),
                desc="Converting unified SMILES (NumPy)"
            ))
            
            for raw, sm, i in results:
                if raw is not None:
                    # Convert raw NumPy dict back to PyG Data object in main process
                    # This avoids the mmap shared memory limit during parallel return
                    from torch_geometric.data import Data
                    data = Data(
                        x=torch.from_numpy(raw["x"]),
                        edge_index=torch.from_numpy(raw["edge_index"]),
                        edge_attr=torch.from_numpy(raw["edge_attr"]),
                    )
                    data.edge_type = torch.from_numpy(raw["edge_type"])
                    data.smiles = raw["smiles"]
                    if raw["y"] is not None:
                        data.y = torch.from_numpy(raw["y"]).unsqueeze(0)
                    
                    graphs.append(data)
                    valid_smiles.append(sm)
    else:
        # Fallback for single-core or very small datasets
        for task in tqdm(tasks, desc="Converting unified SMILES"):
            raw, sm, i = _process_mol_task(task)
            if raw is not None:
                # Re-use smiles_to_graph logic
                from mol_prop_gnn.data.preprocessing import smiles_to_graph
                data = smiles_to_graph(sm, y=raw["y"])
                if data is not None:
                    graphs.append(data)
                    valid_smiles.append(sm)

    logger.info("Successfully converted %d / %d molecules", len(graphs), len(df))
    
    # 3. Enhanced Splitting
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
        similarity_cutoff=similarity_cutoff
    )
    
    # Convert to contiguous InMemoryDataset
    dataset = MoleculeDataset(graphs)
    output = (dataset, train_idx, val_idx, test_idx)
    
    # Save to cache if enabled
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving processed unified dataset to cache: %s", cache_path)
        torch.save(output, cache_path)
    
    return output


# Removed hardcoded UNIFIED_DATASETS logic as build_unified_dataframe now returns task info dynamically.
