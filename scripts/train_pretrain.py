"""Pretraining script for large MoleculeNet datasets (HIV, Tox21).

Pretrains the GNN backbone using a combination of supervised multi-task 
learning on 13 targets (1 HIV + 12 Tox21 assays) and self-supervised 
GraphCL contrastive learning.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from clearml import Task

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mol_prop_gnn.data.unified_dataset import (
    build_unified_dataframe, 
    preprocess_unified_dataset,
)
from mol_prop_gnn.data.preprocessing import (
    get_node_feature_dim,
    get_edge_feature_dim
)
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.factory import build_joint_model
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')

DEFAULT_PRETRAIN_DATASETS = ["hiv", "tox21"]

from pytorch_lightning.callbacks import Callback

class MetricSpy(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print("\n" + "="*60)
        print("🕵️ METRIC SPY: WHAT LIGHTNING ACTUALLY SEES")
        print("Keys available to Checkpoint:", list(trainer.callback_metrics.keys()))
        if "val_loss" in trainer.callback_metrics:
            print("Value of val_loss:", trainer.callback_metrics["val_loss"])
        print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-Scale Pretraining on HIV and Tox21")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin", "pna"], help="Backbone model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Semantic bottleneck dimension")
    parser.add_argument("--contrastive_beta", type=float, default=0.1, help="Contrastive loss weight")
    parser.add_argument("--balanced", action="store_true", default=True, help="Use balanced sampler (defaults to True)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_PRETRAIN_DATASETS, help="Datasets to pretrain on")
    parser.add_argument("--split_type", type=str, default="scaffold", choices=["random", "scaffold", "butina", "stratified_butina"], help="Data splitting methodology")
    parser.add_argument("--similarity_cutoff", type=float, default=0.4, help="Similarity cutoff for Butina clustering")
    args = parser.parse_args()
    
    # Initialize ClearML
    task = Task.init(
        project_name="MoleculeNet-Pretrain", 
        task_name=f"{args.model}_pretrain_hiv_tox21",
        output_uri=True  # Automatically upload checkpoints to ClearML cloud
    )
    task.connect(args)
    task.add_tags(["pretrain", args.model])
    
    # 1. Prepare Data
    df, scaling_stats, target_names, task_types, target_to_ds = build_unified_dataframe(dataset_names=args.datasets)
    graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
        df, 
        target_names=target_names,
        split_type=args.split_type,
        similarity_cutoff=args.similarity_cutoff
    )
    
    datamodule = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampler=args.balanced
    )
    
    # 2. Build Model
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    
    # Ensure datamodule is setup for PNA degree calculation
    datamodule.setup()
    
    # Store model configuration for reconstruction during inference
    model_config = {
        "backbone_name": args.model,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_tasks": len(target_names),
        "bottleneck_dim": args.bottleneck_dim,
        "dropout": args.dropout,
        "deg": datamodule.get_degree_histogram() if args.model == "pna" else None,
    }
    
    model = build_joint_model(**model_config)
    
    lit_module = JointSemiSupModule(
        model=model,
        task_types=task_types,
        dataset_names=target_names,
        learning_rate=args.lr,
        ortho_beta=0.05,
        contrastive_beta=args.contrastive_beta,
        target_to_ds=target_to_ds,
        model_config=model_config
    )
    
    # 3. Training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best_model",
        auto_insert_metric_name=False
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback, MetricSpy()],
        logger=pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name="pretraining"),
    )
    
    logger.info("Starting Large-Scale Pretraining Stage...")
    trainer.fit(lit_module, datamodule=datamodule)
    
    logger.info("Running Test Evaluation...")
    trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")
    
    logger.info("Pretraining completed. Best checkpoint: %s", checkpoint_callback.best_model_path)
    task.close()

if __name__ == "__main__":
    main()
