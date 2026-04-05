"""Script to train the semi-supervised joint embedding model across 5 MoleculeNet datasets.

Creates an interpretable latent "Multi-Dimensional Map" where each dataset is isolated
to a single dimension.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mol_prop_gnn.data.unified_dataset import (
    UNIFIED_DATASETS, 
    build_unified_dataframe, 
    preprocess_unified_dataset,
    get_task_types
)
from mol_prop_gnn.data.preprocessing import (
    get_node_feature_dim,
    get_edge_feature_dim
)
from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.models.joint_embedder import JointMolEmbedder
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule

# Model imports for the factory
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.gat import MolGAT
from mol_prop_gnn.models.egnn import MolEGNN
from mol_prop_gnn.models.gine import MolGINE
from mol_prop_gnn.models.rgcn import MolRGCN

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_backbone(name: str, node_dim: int, edge_dim: int) -> nn.Module:
    """Factory to build the GNN encoder backbone."""
    # We use slightly larger defaults for the multi-task backbone 
    # to ensure it has enough capacity for 5 datasets.
    hidden_dim = 256
    layers = 5

    if name == "gcn":
        return MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=layers)
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
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "egnn", "rgcn", "gine"],
                        help="GNN backbone architecture (default: gcn)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--contrastive_beta", type=float, default=0.1, help="Weight for GraphCL contrastive loss")
    args = parser.parse_args()

    logger.info("Initializing multi-dimensional map setup across: %s", UNIFIED_DATASETS)
    logger.info("Backbone Architecture: %s", args.model.upper())
    
    # 1. Prepare Unified Dataset
    df, scaling_stats = build_unified_dataframe(raw_dir="data/raw")
    graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
        df,
        seed=42,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1
    )
    
    datamodule = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,  
        num_workers=0
    )
    
    task_types = get_task_types()
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()

    # 2. Build the Model
    logger.info("Building Backbone: %s", args.model)
    backbone = build_backbone(args.model, node_dim, edge_dim)
    
    logger.info("Building Joint Embedder (Constraint Bottleneck = %d)", len(UNIFIED_DATASETS))
    model = JointMolEmbedder(
        backbone=backbone,
        backbone_out_dim=backbone.out_channels,
        num_datasets=len(UNIFIED_DATASETS),
        dropout=0.2
    )
    
    lit_module = JointSemiSupModule(
        model=model,
        task_types=task_types,
        dataset_names=UNIFIED_DATASETS,
        learning_rate=0.001,
        ortho_beta=0.5, 
        contrastive_beta=args.contrastive_beta,
    )
    
    # 3. Train
    ckpt_dir = Path(f"lightning_logs/semi_supervised_{args.model}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_overall_score",
        mode="max",
        save_top_k=1,
        filename="best-epoch-{epoch:02d}",
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    logger.info("Training Constraint-Based Joint Evaluator...")
    trainer.fit(lit_module, datamodule=datamodule)
    
    # 4. Evaluate
    logger.info("Evaluating optimal bottleneck states...")
    results = trainer.test(lit_module, datamodule=datamodule, ckpt_path="best", verbose=False)
    
    logger.info("=========================================")
    logger.info(f" Semi-Supervised Map ({args.model.upper()})")
    logger.info("=========================================")
    if results:
        res = results[0]
        for ds_name in UNIFIED_DATASETS:
            tt = task_types[UNIFIED_DATASETS.index(ds_name)]
            if tt == "classification":
                val = res.get(f"test_{ds_name}_auroc", 0)
                logger.info(" => %s AUROC: %.4f", ds_name.upper(), val)
            else:
                val = res.get(f"test_{ds_name}_rmse", 0)
                logger.info(" => %s RMSE (scaled): %.4f", ds_name.upper(), val)
    logger.info("=========================================")


if __name__ == "__main__":
    main()
