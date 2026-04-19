"""Causal Learning runner for Semi-Supervised Multi-Task Prediction.

Splits molecular graphs into Causal and Scaffold subgraphs dynamically during
training to isolate mechanisms of action and prevent spurious correlations.

Usage:
    uv run python scripts/train_causal.py --config configs/causal.yaml
    uv run python scripts/train_causal.py --config configs/engineering.yaml
    uv run python scripts/train_causal.py --config configs/causal.yaml --model pna
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from clearml import Task

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mol_prop_gnn.data.unified_dataset import DEFAULT_DATASETS, build_unified_dataframe, preprocess_unified_dataset
from mol_prop_gnn.data.preprocessing import get_node_feature_dim, get_edge_feature_dim
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.factory import build_causal_model
from mol_prop_gnn.training.causal_semi_sup_module import CausalSemiSupModule
from mol_prop_gnn.utils.config import apply_config_to_parser
from mol_prop_gnn.visualization.causal_mask import CausalVisualizationCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Semi-Supervised Training via Graph Causal Learning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (CLI args override config values)")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin", "pna", "rgcn", "hybrid_causal"],
                        help="GNN backbone architecture (default: gcn)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GNN backbone")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN message-passing layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Semantic bottleneck dimension")
    parser.add_argument("--sparsity_beta", type=float, default=1.0, help="Sparsity constraint weight")
    parser.add_argument("--env_beta", type=float, default=0.5, help="Environment penalty weight")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--split_type", type=str, default="stratified_butina", choices=["random", "scaffold", "butina", "stratified_butina"], help="Data splitting methodology")
    parser.add_argument("--similarity_cutoff", type=float, default=0.4, help="Similarity cutoff for Butina clustering")
    parser.add_argument("--accelerator", type=str, default="auto", help="Hardware accelerator (auto, cpu, gpu)")
    parser.add_argument("--datasets", nargs="+", default=["bbbp", "esol", "freesolv", "lipophilicity", "bace", "hiv", "tox21"], help="Datasets to run Causal Learning on")
    
    # Two-pass parsing: load config defaults first, then CLI overrides
    args, _ = parser.parse_known_args()
    if args.config:
        apply_config_to_parser(parser, args.config)
    args = parser.parse_args()
    
    task = Task.init(
        project_name="MoleculeNet-Causal", 
        task_name=f"{args.model}_causal_ssl_{args.split_type}",
        output_uri=True
    )
    task.connect(args)
    task.add_tags(["causal-ssl", args.model])
    
    # 1. Prepare Data
    logger.info("Initializing multi-dataset assembly...")
    datasets = args.datasets
    df, scaling_stats, target_names, task_types, target_to_ds = build_unified_dataframe(dataset_names=datasets)
    
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
        use_balanced_sampler=True
    )
    datamodule.setup()
    
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    
    model_config = {
        "backbone_name": args.model,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_tasks": len(target_names),
        "bottleneck_dim": args.bottleneck_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "deg": datamodule.get_degree_histogram() if args.model == "pna" else None,
    }
    
    model = build_causal_model(**model_config)
    
    logger.info("Causal model instantiated. Backbone: %s", args.model)
    
    lit_module = CausalSemiSupModule(
        model=model,
        task_types=task_types,
        dataset_names=target_names,
        learning_rate=args.lr,
        sparsity_beta=args.sparsity_beta,
        env_beta=args.env_beta,
        target_to_ds=target_to_ds,
        model_config=model_config
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=10
    )
    
    # Causal mask visualization callback (reports atom importance heatmaps to ClearML)
    viz_callback = CausalVisualizationCallback(
        sample_graphs=list(datamodule.test_dataset),
        task_names=target_names,
        task_types=task_types,
        num_samples=6,
        every_n_val=1,
    )
    
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"causal_ssl_{args.model}"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback, viz_callback],
        logger=tb_logger,
        enable_progress_bar=True,
    )
    
    logger.info("Training Causal Subgraph SSL Map...")
    trainer.fit(
        lit_module, 
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
        ckpt_path=args.checkpoint
    )
    
    logger.info("Evaluating optimal bottleneck states...")
    ckpt_to_test = "best"
    if not checkpoint_callback.best_model_path or not Path(checkpoint_callback.best_model_path).exists():
        logger.warning("No 'best' checkpoint found. Falling back to the 'last' saved state.")
        ckpt_to_test = "last"
        
    trainer.test(lit_module, datamodule=datamodule, ckpt_path=ckpt_to_test, verbose=False)


if __name__ == "__main__":
    main()
