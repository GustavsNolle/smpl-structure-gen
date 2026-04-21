"""Phase 3: Counterfactual Molecule Generation (Causal DiGress)

Trains a Discrete Denoising Diffusion Model guided by a frozen Causal Judge.
This script sets up a ClearML workflow and executes the training loop.

Usage:
    uv run python scripts/train_digress.py --config configs/digress.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from clearml import Task, Model

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mol_prop_gnn.utils.config import load_config, apply_config_to_parser
from mol_prop_gnn.training.digress_module import DiGressModule
# We re-use the RDKitZINCDataModule from Phase 1 for pre-training distribution
from train_pretrain_masked import RDKitZINCDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Causal DiGress Training")
    parser.add_argument("--config", type=str, default="configs/digress.yaml", help="Path to YAML config file")
    
    # Model Architecture Overrides
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of DiGress layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of Attention heads")
    
    # Diffusion Overrides
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Total diffusion steps")
    parser.add_argument("--p_uncond", type=float, default=0.15, help="CFG dropout rate")
    
    # Training Overrides
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    # Infrastructure
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--accelerator", type=str, default="auto", help="Hardware accelerator (auto, cpu, gpu)")
    parser.add_argument("--resume_id", type=str, default=None, help="ClearML model ID to resume training from")
    
    args, _ = parser.parse_known_args()
    if args.config:
        apply_config_to_parser(parser, args.config)
    args = parser.parse_args()

    # 1. ClearML Workflow Setup
    task = Task.init(
        project_name="MoleculeNet-Phase3-CausalDiGress", 
        task_name="train_digress_ada_ln_cfg",
        output_uri=True
    )
    task.connect(args)
    task.add_tags(["phase3", "digress", "causal_guidance", "cfg", "ada_ln_zero"])
    
    logger.info("Initializing Phase 3: Causal DiGress")

    # 2. Data Preparation
    # We use ZINC for learning the unconditional molecule distribution
    # The datamodule from train_pretrain_masked automatically downloads/prepares it.
    config = load_config(args.config) if args.config else {}
    data_cfg = config.get("data", {})
    
    datamodule = RDKitZINCDataModule(
        root=data_cfg.get("root", "data/ZINC"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_rate=0.0, # No masking for diffusion, we need clean graphs
    )
    
    # 3. Model Setup
    causal_judge_model_id = config.get("causal_judge_model_id", "94f148c657ed4b7b8fdaa54b4ad2bdd3")
    
    lit_module = DiGressModule(
        num_node_classes=config.get("num_node_classes", 11),
        num_edge_classes=config.get("num_edge_classes", 6),
        hidden_dim=args.hidden_dim,
        causal_cond_dim=config.get("causal_cond_dim", 128),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_timesteps=args.num_timesteps,
        learning_rate=args.lr,
        causal_judge_model_id=causal_judge_model_id,
        p_uncond=args.p_uncond
    )
    
    # 4. Trainer Setup
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/phase3_digress",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=5
    )
    
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name="causal_digress"
    )

    # 4b. High-performance Extraction (Bypassing PyG __getitem__ bottleneck)
    logger.info("Extracting train SMILES and Node Counts for evaluation...")
    datamodule.setup()
    dataset = datamodule.train_dataset

    # 1. Fast SMILES extraction
    # The dataset might be a Subset, so we need to handle it
    if hasattr(dataset, 'dataset'):
        full_dataset = dataset.dataset
        indices = dataset.indices
    else:
        full_dataset = dataset
        indices = list(range(len(dataset)))

    # RDKitZINC stores 'smiles' as an attribute of the flat 'data' object in InMemoryDataset
    if hasattr(full_dataset, 'data') and hasattr(full_dataset.data, 'smiles'):
        all_smiles = full_dataset.data.smiles
        train_smiles = [all_smiles[i] for i in indices]
    else:
        # Fallback: Extract from raw file if not in memory
        raw_path = Path(datamodule.root) / "raw" / "zinc_250k.txt"
        if raw_path.exists():
            with open(raw_path, 'r') as f:
                raw_smiles = [line.strip() for line in f.readlines()]
                if "smiles" in raw_smiles[0].lower():
                    raw_smiles = raw_smiles[1:]
                # We need to assume the indices match the raw file order (which they should for ZINC)
                train_smiles = [raw_smiles[i] for i in indices]
        else:
            logger.warning("Could not find raw SMILES for novelty check. Novelty will be 100%.")
            train_smiles = []

    # 2. Fast Node Count extraction via slices
    if hasattr(full_dataset, 'slices') and 'x' in full_dataset.slices:
        x_slices = full_dataset.slices['x']
        # Node count of graph i is slices[i+1] - slices[i]
        all_node_counts = (x_slices[1:] - x_slices[:-1]).tolist()
        node_counts = [all_node_counts[i] for i in indices]
    else:
        # Emergency slow fallback
        node_counts = [data.x.shape[0] for data in dataset]
    
    from mol_prop_gnn.training.digress_callbacks import DiGressEvaluationCallback
    
    eval_callback = DiGressEvaluationCallback(
        train_smiles=train_smiles,
        train_node_counts=node_counts,
        num_samples=1000,
        guidance_scale=7.0,
        evaluate_every_n_epochs=1  # Run often for monitoring
    )
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, eval_callback],
        logger=tb_logger,
        gradient_clip_val=0.8,
        enable_progress_bar=True,
    )
    
    # 5. Execute Training
    ckpt_path = None
    if args.resume_id:
        logger.info(f"Resuming from ClearML Model ID: {args.resume_id}...")
        resume_model = Model(model_id=args.resume_id)
        ckpt_path = resume_model.get_local_copy()
        logger.info(f"Local checkpoint path: {ckpt_path}")

    logger.info("Starting DiGress Training with Causal Guidance...")
    trainer.fit(
        lit_module, 
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )
    
    logger.info("✓ Phase 3 Training Complete!")


if __name__ == "__main__":
    main()
