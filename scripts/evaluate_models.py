"""CLI script for comprehensive model evaluation and comparison."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd

from mol_prop_gnn.utils.config import load_config
from mol_prop_gnn.data.download import download_moleculenet, get_dataset_info
from mol_prop_gnn.data.preprocessing import (
    preprocess_moleculenet,
    compute_fingerprint,
    compute_descriptors,
    get_node_feature_dim,
    get_edge_feature_dim,
)
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.rgcn import MolRGCN
from mol_prop_gnn.models.mlp_baseline import MLPBaseline
from mol_prop_gnn.models.rdkit_baseline import RDKitBaseline
from mol_prop_gnn.models.xgboost_baseline import XGBoostBaseline
from mol_prop_gnn.models.lightgbm_baseline import LightGBMBaseline
from mol_prop_gnn.models.sage import MolGraphSAGE
from mol_prop_gnn.models.transformer import MolTransformerGNN
from mol_prop_gnn.training.supervised_module import MolPropertyModule
from mol_prop_gnn.evaluation.metrics import compute_all_metrics
import torch_geometric

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_fingerprints(graphs, n_bits=2048):
    """Extract Morgan fingerprints and labels from graph list."""
    fps, labels = [], []
    for g in graphs:
        if hasattr(g, "smiles"):
            fp = compute_fingerprint(g.smiles, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(fps), np.array(labels)


def extract_descriptors(graphs):
    """Extract RDKit descriptors and labels from graph list."""
    descs, labels = [], []
    for g in graphs:
        if hasattr(g, "smiles"):
            desc = compute_descriptors(g.smiles)
            if desc is not None:
                descs.append(desc)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(descs), np.array(labels)


def evaluate_torch_model(module, graphs, task_type, batch_size=32):
    """Evaluate a PyTorch Lightning model on a set of graphs."""
    device = next(module.parameters()).device
    loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size)
    preds, targets = [], []
    module.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = module(batch)
            preds.append(out.cpu())
            targets.append(batch.y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return compute_all_metrics(preds, targets, task_type=task_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    task_type = data_cfg.get("task_type", "classification")

    # 1. Load Data
    dataset_name = data_cfg.get("dataset_name", "bbbp")
    csv_path = download_moleculenet(dataset_name, raw_dir=data_cfg.get("raw_dir", "data/raw"))
    graphs, train_idx, val_idx, test_idx = preprocess_moleculenet(csv_path, config)

    dm = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=config.get("training", {}).get("batch_size", 32),
    )
    dm.setup()

    results = []

    # 2. Evaluate Non-DL Baselines
    logger.info("Evaluating tabular baselines...")

    # RDKit Descriptors + Random Forest
    X_train_desc, y_train = extract_descriptors(dm.train_dataset)
    X_test_desc, y_test = extract_descriptors(dm.test_dataset)
    rdkit_model = RDKitBaseline(task_type=task_type)
    rdkit_model.fit(X_train_desc, y_train)
    results.append({"Model": "RDKit (RF)", **rdkit_model.evaluate(X_test_desc, y_test)})

    # Morgan fingerprint baselines
    X_train_fp, y_train_fp = extract_fingerprints(dm.train_dataset)
    X_val_fp, y_val_fp = extract_fingerprints(dm.val_dataset)
    X_test_fp, y_test_fp = extract_fingerprints(dm.test_dataset)

    # XGBoost
    xgb_model = XGBoostBaseline(task_type=task_type)
    xgb_model.fit(X_train_fp, y_train_fp, eval_set=[(X_val_fp, y_val_fp)])
    results.append({"Model": "XGBoost (FP)", **xgb_model.evaluate(X_test_fp, y_test_fp)})

    # LightGBM
    lgb_model = LightGBMBaseline(task_type=task_type)
    lgb_model.fit(X_train_fp, y_train_fp, eval_set=[(X_val_fp, y_val_fp)])
    results.append({"Model": "LightGBM (FP)", **lgb_model.evaluate(X_test_fp, y_test_fp)})

    # 3. Evaluate DL Models
    logger.info("Evaluating Deep Learning models (from checkpoints if available)...")
    log_dir = Path(args.log_dir)

    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    output_dim = data_cfg.get("num_tasks", 1)

    dl_models = [
    ("gcn", MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, output_dim=output_dim)),
    ("rgcn", MolRGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, output_dim=output_dim)),
    ("sage", MolGraphSAGE(node_input_dim=node_dim, edge_input_dim=edge_dim, output_dim=output_dim)),
    ("transformer", MolTransformerGNN(node_input_dim=node_dim, edge_input_dim=edge_dim, output_dim=output_dim)),
    ("mlp_baseline", MLPBaseline(input_dim=node_dim, output_dim=output_dim)),
    ]

    for m_name, net in dl_models:
        ckpt_files = list(log_dir.glob(f"semi_supervised_{m_name}/**/checkpoints/*.ckpt"))
        if not ckpt_files:
            logger.warning("No checkpoint for %s, skipping.", m_name)
            continue

        latest_ckpt = sorted(ckpt_files, key=lambda p: p.stat().st_mtime)[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = MolPropertyModule.load_from_checkpoint(
            latest_ckpt, model=net, task_type=task_type,
        ).to(device)
        logger.info("Loaded %s from %s", m_name, latest_ckpt.name)

        metrics = evaluate_torch_model(module, dm.test_dataset, task_type)
        pretty = m_name.replace("_", " ").upper()
        results.append({"Model": f"Mol{pretty}", **metrics})

    # 4. Show Results
    summary_df = pd.DataFrame(results)
    sort_col = "auroc" if task_type == "classification" else "rmse"
    ascending = task_type != "classification"
    summary_df = summary_df.sort_values(sort_col, ascending=ascending)

    print("\n" + "=" * 60)
    print("            MODEL COMPARISON RESULTS")
    print(f"      Dataset: {dataset_name.upper()} ({task_type})")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
