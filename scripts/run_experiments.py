"""Multi-dataset benchmark orchestrator for MolPropGNN.

Benchmarks all model architectures across multiple MoleculeNet datasets
(BBBP, ESOL, etc.) and generates a comparison table.
"""

import copy
import logging
import shutil
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mol_prop_gnn.utils.config import load_config
from mol_prop_gnn.data.download import download_moleculenet, get_dataset_info
from mol_prop_gnn.data.preprocessing import (
    preprocess_moleculenet,
    compute_fingerprint,
    get_node_feature_dim,
    get_edge_feature_dim,
)
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.training.supervised_module import MolPropertyModule
from mol_prop_gnn.models.xgboost_baseline import XGBoostBaseline
from mol_prop_gnn.evaluation.metrics import compute_all_metrics
from train import build_model

import torch

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Datasets to benchmark
BENCHMARK_DATASETS = {
    "bbbp": {"task_type": "classification", "num_tasks": 1, "metric": "auroc"},
    "esol": {"task_type": "regression", "num_tasks": 1, "metric": "rmse"},
    "bace": {"task_type": "classification", "num_tasks": 1, "metric": "auroc"},
    "lipophilicity": {"task_type": "regression", "num_tasks": 1, "metric": "rmse"},
    "freesolv": {"task_type": "regression", "num_tasks": 1, "metric": "rmse"},
}

DL_MODEL_NAMES = ["mlp_baseline", "gcn", "sage", "transformer"]


def extract_fingerprints(graphs, n_bits=2048):
    """Extract Morgan fingerprints and labels."""
    fps, labels = [], []
    for g in graphs:
        if hasattr(g, "smiles"):
            fp = compute_fingerprint(g.smiles, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(fps), np.array(labels)


def train_and_eval_dl(model_name, config, datamodule, experiment, task_type):
    """Train a DL model and return test metrics."""
    logger.info("   Training %s ...", model_name.upper())

    model = build_model(config, model_name)
    lit_module = MolPropertyModule(
        model=model,
        task_type=task_type,
        learning_rate=0.005,
    )

    ckpt_dir = Path(f"lightning_logs/{experiment}/{model_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(lit_module, datamodule=datamodule)
    metrics = trainer.test(lit_module, datamodule=datamodule, ckpt_path="best", verbose=False)

    if metrics and len(metrics) > 0:
        return metrics[0]
    return {}


def run_dataset_benchmark(base_config, dataset_name, dataset_info):
    """Run full benchmark on a single dataset."""
    config = copy.deepcopy(base_config)
    config["data"]["dataset_name"] = dataset_name
    config["data"]["task_type"] = dataset_info["task_type"]
    config["data"]["num_tasks"] = dataset_info["num_tasks"]
    config["training"]["max_epochs"] = 10
    config["training"]["batch_size"] = 64

    task_type = dataset_info["task_type"]
    metric_key = dataset_info["metric"]

    # Download and preprocess
    csv_path = download_moleculenet(dataset_name, raw_dir=config["data"].get("raw_dir", "data/raw"))
    graphs, train_idx, val_idx, test_idx = preprocess_moleculenet(csv_path, config)

    dm = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=32,
    )
    dm.setup()

    results = {}

    # XGBoost baseline (fingerprints)
    logger.info("   Training XGBOOST ...")
    X_train, y_train = extract_fingerprints(dm.train_dataset)
    X_val, y_val = extract_fingerprints(dm.val_dataset)
    X_test, y_test = extract_fingerprints(dm.test_dataset)

    xgb = XGBoostBaseline(task_type=task_type, n_estimators=200, early_stopping_rounds=10)
    eval_set = [(X_val, y_val)] if len(X_val) > 0 else None
    xgb.fit(X_train, y_train, eval_set=eval_set)
    xgb_metrics = xgb.evaluate(X_test, y_test)
    results["XGBoost"] = xgb_metrics.get(metric_key, 0.0)

    # DL models
    for model_name in DL_MODEL_NAMES:
        m = train_and_eval_dl(
            model_name, config, dm,
            experiment=f"{dataset_name}_{model_name}",
            task_type=task_type,
        )
        test_key = f"test_{metric_key}"
        results[model_name.upper().replace("_BASELINE", "")] = m.get(test_key, 0.0)

    return results


def main():
    log_dir = Path("lightning_logs")
    if log_dir.exists():
        logger.info("Wiping legacy lightning_logs/ for clean benchmark...")
        shutil.rmtree(log_dir)

    config = load_config("configs/default.yaml")

    all_results = {}

    for ds_name, ds_info in BENCHMARK_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {ds_name.upper()} ({ds_info['task_type']})")
        print(f"{'='*70}")
        all_results[ds_name] = run_dataset_benchmark(config, ds_name, ds_info)

    # Print master comparison
    print(f"\n\n{'='*70}")
    print("     FINAL BENCHMARK RESULTS")
    print(f"{'='*70}")

    for ds_name, ds_info in BENCHMARK_DATASETS.items():
        metric = ds_info["metric"]
        results = all_results[ds_name]
        records = [
            {"Model": model, f"{metric.upper()} ({ds_name.upper()})": f"{val:.4f}"}
            for model, val in sorted(results.items(), key=lambda x: x[1], reverse=(metric == "auroc"))
        ]
        df = pd.DataFrame(records)
        print(f"\n  {ds_name.upper()} — {ds_info['task_type']} (metric: {metric})")
        print("-" * 50)
        print(df.to_string(index=False))

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
