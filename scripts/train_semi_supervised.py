"""Script to train the semi-supervised joint embedding model across 5 MoleculeNet datasets.

Creates an interpretable latent "Multi-Dimensional Map" where each dataset is isolated
to a single dimension.
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
    DEFAULT_DATASETS, 
    build_unified_dataframe, 
    preprocess_unified_dataset,
)
from mol_prop_gnn.data.preprocessing import (
    get_node_feature_dim,
    get_edge_feature_dim
)
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.joint_embedder import JointMolEmbedder
from mol_prop_gnn.models.factory import build_joint_model
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule

# Model imports for the factory
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.gin import MolGIN
from mol_prop_gnn.models.pna import MolPNA
from mol_prop_gnn.models.gat import MolGAT
from mol_prop_gnn.models.egnn import MolEGNN
from mol_prop_gnn.models.gine import MolGINE
from mol_prop_gnn.models.rgcn import MolRGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def build_backbone(name: str, node_dim: int, edge_dim: int, datamodule: MoleculeDataModule = None) -> nn.Module:
    """Factory to build the GNN encoder backbone."""
    # We use slightly larger defaults for the multi-task backbone 
    # to ensure it has enough capacity for 5 datasets.
    hidden_dim = 256
    layers = 5

    if name == "gcn":
        return MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "gin":
        return MolGIN(node_input_dim=node_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "pna":
        deg = datamodule.get_degree_histogram()
        return MolPNA(deg=deg, node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "gat":
        return MolGAT(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=64, heads=4, num_gnn_layers=layers)
    elif name == "egnn":
        return MolEGNN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=layers)
    elif name == "gine":
        return MolGINE(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
    elif name == "rgcn":
        return MolRGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=layers)
    else:
        raise ValueError(f"Unknown backbone model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Dimensional Map Semi-Supervised Training")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin", "pna", "gat", "egnn", "rgcn", "gine"],
                        help="GNN backbone architecture (default: gcn)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="Semantic bottleneck dimension")
    parser.add_argument("--ortho_beta", type=float, default=0.01, help="Orthogonal constraint weight")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--contrastive_beta", type=float, default=0.1, help="Weight for GraphCL contrastive loss")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--backbone_path", type=str, default=None, help="Path to pretrained backbone checkpoint")
    parser.add_argument("--split_type", type=str, default="stratified_butina", 
                        choices=["random", "scaffold", "stratified_scaffold", "butina", "stratified_butina"],
                        help="Data splitting methodology (default: stratified_butina)")
    parser.add_argument("--similarity_cutoff", type=float, default=0.4, help="Butina similarity cutoff")
    parser.add_argument("--balanced", action="store_true", default=False, help="Use balanced sampler for training")
    # Defaulting to all 7 datasets: 5 original + HIV + Tox21
    parser.add_argument("--datasets", nargs="+", default=["bbbp", "esol", "bace", "freesolv", "lipophilicity", "hiv", "tox21"], help="Datasets to use for training")
    args = parser.parse_args()

    # Initialize ClearML for experiment tracking
    task_name = f"train_{args.model}" if not args.backbone_path else f"finetune_{args.model}"
    task = Task.init(
        project_name="MoleculeNet-SemiSupervised", 
        task_name=task_name,
        output_uri=True  # Automatically upload checkpoints to ClearML cloud
    )
    task.connect(args)

    logger.info("Initializing multi-dimensional map setup across: %s", args.datasets)
    logger.info("Backbone Architecture: %s", args.model.upper())
    
    # 1. Prepare Unified Dataset
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
    
    # Ensure datamodule is setup for PNA
    datamodule.setup()
    
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()

    datamodule.setup()
    
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

    # Handle Backbone Loading
    if args.backbone_path:
        logger.info("Loading pretrained backbone from: %s", args.backbone_path)
        # weights_only=False is required for loading custom Lightning objects/datasets
        checkpoint = torch.load(args.backbone_path, map_location="cpu", weights_only=False)
        
        if not isinstance(checkpoint, dict):
            type_name = type(checkpoint).__name__
            raise TypeError(
                f"Expected a dictionary checkpoint (Lightning .ckpt), but got a {type_name}. "
                "This usually happens if you provide a path to a DATASET instead of a MODEL. "
                "Please check your path and ensure it points to a .ckpt file."
            )
        
        state_dict = checkpoint["state_dict"]
        # Correctly strip 'model.backbone.' prefix (15 chars) to load into the backbone directly
        backbone_dict = {k[15:]: v for k, v in state_dict.items() if k.startswith("model.backbone.")}
        model.backbone.load_state_dict(backbone_dict)
        logger.info("Transfer learning initiated: Backbone weights loaded.")
    
    # 2. Setup Lightning Module
    lit_module = JointSemiSupModule(
        model=model,
        task_types=task_types,
        dataset_names=target_names,
        learning_rate=args.lr,
        ortho_beta=0.01,
        contrastive_beta=args.contrastive_beta,
        target_to_ds=target_to_ds,
        model_config=model_config
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    
    # Initialize TensorBoard for ClearML to capture
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"semi_supervised_{args.model}"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    logger.info("Training Constraint-Based Joint Evaluator...")
    trainer.fit(lit_module, datamodule=datamodule, ckpt_path=args.checkpoint)
    
    # 4. Evaluate (try best, fallback to last)
    logger.info("Evaluating optimal bottleneck states...")
    ckpt_to_test = "best"
    if not checkpoint_callback.best_model_path or not Path(checkpoint_callback.best_model_path).exists():
        logger.warning("No 'best' checkpoint found. Falling back to the 'last' saved state.")
        ckpt_to_test = "last"
        
    results_list = trainer.test(lit_module, datamodule=datamodule, ckpt_path=ckpt_to_test, verbose=False)
    
    # 5. Robust Extraction: prefer state-passing over trainer return value
    if hasattr(lit_module, "latest_test_results") and lit_module.latest_test_results:
        res = lit_module.latest_test_results
        logger.info("Retrieved results via model state-passing.")
    elif results_list:
        res = results_list[0]
        logger.info("Retrieved results via trainer return value.")
    else:
        logger.error("Failed to retrieve test results from any source.")
        res = {}

    logger.info("=========================================")
    logger.info(f" Semi-Supervised Map ({args.model.upper()})")
    logger.info("=========================================")
    
    for ds_name in target_names:
        # Check both prefixed and unprefixed keys for robustness
        idx = target_names.index(ds_name)
        tt = task_types[idx]
        if tt == "classification":
            val = res.get(f"test_{ds_name}_auroc", res.get(f"{ds_name}_auroc", 0))
            logger.info(" => %s AUROC: %.4f", ds_name.upper(), val)
        else:
            val = res.get(f"test_{ds_name}_rmse", res.get(f"{ds_name}_rmse", 0))
            logger.info(" => %s RMSE (scaled): %.4f", ds_name.upper(), val)
    
    logger.info("-----------------------------------------")
    logger.info(" => OVERALL TEST SCORE: %.4f", res.get("test_overall_score", res.get("overall_score", 0.0)))
    logger.info("=========================================")


if __name__ == "__main__":
    main()
