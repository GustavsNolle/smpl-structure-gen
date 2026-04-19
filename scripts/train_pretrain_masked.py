"""Phase 1: Self-supervised pre-training via Continuous Masked Node Prediction on ZINC.

Trains the exact MolGINE continuous encoder to predict 38-dim RDKit masked atom features 
from graph topology. Saves MolGINE state_dict (.pt) for Phase 2 fine-tuning.

Usage:
    python scripts/train_pretrain_masked.py --config configs/pretrain_masked.yaml
    python scripts/train_pretrain_masked.py --config configs/pretrain_masked.yaml --subset false
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mol_prop_gnn.data.rdkit_zinc import RDKitZINC
from mol_prop_gnn.models.gine_sixseeven import MolGINE
from mol_prop_gnn.training.pretrain_masked_module import (
    MaskedNodePredModule,
    ContinuousMaskTransform,
)
from mol_prop_gnn.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── ZINC DataModule ──────────────────────────────────────────────────────

class RDKitZINCDataModule(pl.LightningDataModule):
    """Lightning DataModule for custom RDKit ZINC with continuous masking."""

    def __init__(
        self,
        root: str = "data/ZINC",
        batch_size: int = 128,  # Lowered from 256 for memory constraints
        num_workers: int = 4,
        mask_rate: float = 0.15,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_transform = ContinuousMaskTransform(mask_rate=mask_rate)

    def setup(self, stage=None):
        # The RDKitZINC dataset processes the 250k ZINC subset downloaded.
        # Here we do a simple random split. Real setups might use scaffold.
        full_dataset = RDKitZINC(self.root, transform=self.mask_transform)
        
        torch.manual_seed(42)
        total_size = len(full_dataset)
        val_size = int(0.1 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        logger.info(
            "RDKit ZINC loaded — train: %d, val: %d, test: %d",
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )


# ── Encoder Checkpoint Callback ─────────────────────────────────────────

class SaveEncoderCallback(Callback):
    """Saves encoder state_dict (.pt) when validation loss improves."""

    def __init__(self, save_path: str | Path):
        super().__init__()
        self.save_path = Path(save_path)
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return

        val_loss = val_loss.item()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            encoder_state = pl_module.get_encoder_state_dict()
            torch.save(encoder_state, self.save_path)

            logger.info(
                "💾 Saved MolGINE backbone setup (val_loss=%.4f) → %s",
                val_loss, self.save_path,
            )


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Continuous Masked Node Prediction Pre-training on ZINC"
    )
    parser.add_argument(
        "--config", type=str, default="configs/pretrain_masked.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    mask_cfg = config.get("masking", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    # ── ClearML ───────────────────────────────────────────────────────
    task = Task.init(
        project_name="MoleculeNet-Pretrain",
        task_name="continuous_masked_node_zinc",
        output_uri=True,
    )
    task.connect({
        "data": data_cfg,
        "masking": mask_cfg,
        "model": model_cfg,
        "training": train_cfg,
    })
    task.add_tags(["pretrain", "masked_node_continuous", "zinc"])

    # ── Seed ─────────────────────────────────────────────────────────
    seed = train_cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # ── Data ─────────────────────────────────────────────────────────
    mask_rate = mask_cfg.get("mask_rate", 0.15)
    batch_size = train_cfg.get("batch_size", 128)  # Defaulting low per user warning

    datamodule = RDKitZINCDataModule(
        root=data_cfg.get("root", "data/ZINC"),
        batch_size=batch_size,
        num_workers=config.get("num_workers", 4),
        mask_rate=mask_rate,
    )

    # ── Model (The Exact MolGINE used downstream!) ──────────────────
    # Note: Global features for GINE are currently set to 0. 
    # Can adjust output_dim to 1 as it doesn't matter (we intercept .encode())
    model = MolGINE(
        node_input_dim=38,
        edge_input_dim=13,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_gnn_layers=model_cfg.get("num_gnn_layers", 5),
        dropout=model_cfg.get("dropout", 0.3),
        global_features_dim=0, # Pretraining focuses on topology only
        output_dim=1, # Ignored, we use the prediction head on the backbone
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("MolGINE Backbone — %d parameters (%.2f M)", total_params, total_params / 1e6)

    # ── Lightning Module ─────────────────────────────────────────────
    sched_cfg = train_cfg.get("scheduler", {})
    lit_module = MaskedNodePredModule(
        backbone=model,
        node_dim=38,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        scheduler_patience=sched_cfg.get("patience", 10),
        scheduler_factor=sched_cfg.get("factor", 0.5),
    )

    # ── Callbacks ────────────────────────────────────────────────────
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/pretrain_masked"))

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        SaveEncoderCallback(save_path=ckpt_dir / "molgine_encoder_best.pt"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator=config.get("accelerator", "auto"),
        devices=1,
        max_epochs=train_cfg.get("epochs", 100),
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="lightning_logs", name="pretrain_masked"
        ),
        deterministic=True,
    )

    # ── Train ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Phase 1: Continuous Masked Node Prediction Pre-training")
    logger.info("  Mask rate: %.0f%%", mask_rate * 100)
    logger.info("=" * 60)

    trainer.fit(lit_module, datamodule=datamodule)

    # ── Test ─────────────────────────────────────────────────────────
    logger.info("Running test evaluation ...")
    trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")

    logger.info("✓ Continuous Pre-training complete!")
    logger.info("  Encoder weights ready for Phase 2: %s", ckpt_dir / "molgine_encoder_best.pt")


if __name__ == "__main__":
    main()
