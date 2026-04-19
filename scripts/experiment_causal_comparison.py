"""Benchmarking Causal GNNs versus Vanilla GNNs and Tabular Baselines.

Evaluates performance on BBBP, BACE (Classification) and FreeSolv (Regression)
under standardized splitting.

Usage:
    python scripts/experiment_causal_comparison.py --config configs/causal_comparison.yaml
"""

from __future__ import annotations
import argparse
import logging
import sys
import copy
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.data.download import get_dataset_info, download_moleculenet
from mol_prop_gnn.data.unified_dataset import preprocess_unified_dataset
from mol_prop_gnn.data.preprocessing import (
    compute_fingerprint,
    get_node_feature_dim,
    get_edge_feature_dim,
)
from mol_prop_gnn.models.factory import build_causal_model, build_backbone
from mol_prop_gnn.training.causal_semi_sup_module import CausalSemiSupModule
from mol_prop_gnn.training.supervised_module import MolPropertyModule
from mol_prop_gnn.models.xgboost_baseline import XGBoostBaseline
from mol_prop_gnn.models.lightgbm_baseline import LightGBMBaseline
from mol_prop_gnn.models.rdkit_baseline import RDKitBaseline
from mol_prop_gnn.utils.config import apply_config_to_parser, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def extract_fingerprints(dataset, n_bits=2048):
    fps, labels = [], []
    for g in dataset:
        if hasattr(g, "smiles") and getattr(g, "smiles", None) is not None:
            fp = compute_fingerprint(g.smiles, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(fps), np.array(labels)

def evaluate_torch_model(lit_module, datamodule, task_type):
    """Evaluate a PyTorch model natively on the test dataset."""
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.test(lit_module, datamodule=datamodule, verbose=False)
    
    # Robust extraction from lit_module state if available (used by Causal module)
    res = {}
    if hasattr(lit_module, "latest_test_results") and lit_module.latest_test_results:
        raw_res = lit_module.latest_test_results
        if task_type == "classification":
            # Find any key ending in _auroc
            auroc_vals = [v for k, v in raw_res.items() if k.endswith("_auroc")]
            res["auroc"] = float(np.mean(auroc_vals)) if auroc_vals else 0.0
            acc_vals = [v for k, v in raw_res.items() if k.endswith("_acc")]
            res["accuracy"] = float(np.mean(acc_vals)) if acc_vals else 0.0
            res.update({"rmse": np.nan, "mae": np.nan, "r2": np.nan})
        else:
            rmse_vals = [v for k, v in raw_res.items() if k.endswith("_rmse")]
            res["rmse"] = float(np.mean(rmse_vals)) if rmse_vals else 0.0
            mae_vals = [v for k, v in raw_res.items() if k.endswith("_mae")]
            res["mae"] = float(np.mean(mae_vals)) if mae_vals else 0.0
            r2_vals = [v for k, v in raw_res.items() if k.endswith("_r2")]
            res["r2"] = float(np.mean(r2_vals)) if r2_vals else 0.0
            res.update({"accuracy": np.nan, "auroc": np.nan})
    else:
        # Fallback to standard trainer metrics
        metrics_dict = trainer.callback_metrics
        if task_type == "classification":
            res["accuracy"] = float(metrics_dict.get("test_acc", 0.0))
            res["auroc"] = float(metrics_dict.get("test_auroc", 0.0))
            res.update({"rmse": np.nan, "mae": np.nan, "r2": np.nan})
        else:
            res["rmse"] = float(metrics_dict.get("test_rmse", 0.0))
            res["mae"] = float(metrics_dict.get("test_mae", 0.0))
            res["r2"] = float(metrics_dict.get("test_r2", 0.0))
            res.update({"accuracy": np.nan, "auroc": np.nan})
            
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    
    args, _ = parser.parse_known_args()
    if args.config:
        apply_config_to_parser(parser, args.config)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Overrides
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 128)
    
    dataset_names = config.get("datasets", ["bbbp", "bace", "freesolv"])
    backbone_names = config.get("models", ["gcn", "gin"])
    tabular_names = config.get("tabular_models", ["xgboost", "lightgbm", "rdkit_rf"])
    
    results = []
    
    for ds_name in dataset_names:
        logger.info(f"\n{'='*60}\nProcessing Dataset: {ds_name.upper()}\n{'='*60}")
        csv_path = download_moleculenet(ds_name)
        info = get_dataset_info(ds_name)
        task_type = info["task_type"]
        df = pd.read_csv(csv_path)
        
        # Standardize smiles column
        smiles_col = info["smiles_col"]
        if smiles_col != "smiles":
            df = df.rename(columns={smiles_col: "smiles"})
        
        target_names = info["target_cols"]
        
        graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
            df, 
            target_names=target_names,
            split_type=config.get("split_type", "stratified_butina"),
            similarity_cutoff=config.get("similarity_cutoff", 0.4),
        )
        
        dm = MoleculeDataModule(
            graphs=graphs,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=batch_size,
        )
        dm.setup()
        
        node_dim = get_node_feature_dim()
        edge_dim = get_edge_feature_dim()
        
        # --- TABULAR BASELINES ---
        X_train_fp, y_train = extract_fingerprints(dm.train_dataset)
        X_test_fp, y_test = extract_fingerprints(dm.test_dataset)
        
        for t_name in tabular_names:
            logger.info(f"--- Evaluatiing Tabular Baseline: {t_name.upper()} ---")
            if t_name == "xgboost":
                model = XGBoostBaseline(task_type=task_type)
            elif t_name == "lightgbm":
                model = LightGBMBaseline(task_type=task_type)
            else:
                model = RDKitBaseline(task_type=task_type)
                
            model.fit(X_train_fp, y_train)
            res = model.evaluate(X_test_fp, y_test)
            # Standardize for final df
            metrics = {
                "accuracy": res.get("accuracy", np.nan),
                "auroc": res.get("auroc", np.nan),
                "rmse": res.get("rmse", np.nan),
                "mae": res.get("mae", np.nan),
                "r2": res.get("r2", np.nan),
            }
            results.append({"Dataset": ds_name.upper(), "Type": "TABULAR", "Model": t_name.upper(), **metrics})
            
        # --- GNN BACKBONES ---
        for bb_name in backbone_names:
            # 1. Vanilla Supervised
            logger.info(f"--- Training Vanilla GNN: {bb_name.upper()} ---")
            vanilla_net = build_backbone(
                name=bb_name,
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=config.get("hidden_dim", 128),
                layers=config.get("num_layers", 3),
            )
            
            vanilla_module = MolPropertyModule(
                model=vanilla_net,
                task_type=task_type,
                learning_rate=config.get("lr", 1e-3),
            )
            
            trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=epochs, logger=False, enable_progress_bar=False)
            trainer.fit(vanilla_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=[dm.val_dataloader(), dm.test_dataloader()])
            res = evaluate_torch_model(vanilla_module, dm, task_type)
            results.append({"Dataset": ds_name.upper(), "Type": "VANILLA", "Model": bb_name.upper(), **res})
            
            # 2. Causal SSL (CIB)
            logger.info(f"--- Training Causal GNN: {bb_name.upper()} ---")
            causal_net = build_causal_model(
                backbone_name=bb_name,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_tasks=len(target_names),
                bottleneck_dim=config.get("bottleneck_dim", 64),
                hidden_dim=config.get("hidden_dim", 128),
                num_layers=config.get("num_layers", 3),
                dropout=config.get("dropout", 0.2),
            )
            causal_module = CausalSemiSupModule(
                model=causal_net,
                task_types=[task_type],
                dataset_names=[ds_name],
                learning_rate=config.get("lr", 1e-3),
                sparsity_beta=config.get("sparsity_beta", 1.0),
                env_beta=config.get("env_beta", 0.5),
            )
            
            trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=epochs, logger=False, enable_progress_bar=False)
            trainer.fit(causal_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=[dm.val_dataloader(), dm.test_dataloader()])
            res = evaluate_torch_model(causal_module, dm, task_type)
            results.append({"Dataset": ds_name.upper(), "Type": "CAUSAL", "Model": bb_name.upper(), **res})

    summary_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "=" * 100)
    print("    CAUSAL VS VANILLA GNN COMPARISON RESULTS")
    print("=" * 100)
    print(summary_df.to_string(index=False, float_format="%.3f"))
    print("=" * 100)

if __name__ == "__main__":
    main()
