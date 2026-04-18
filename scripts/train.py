"""Train a molecular property prediction model.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
    uv run python scripts/train.py --config configs/default.yaml --model gcn
    uv run python scripts/train.py --config configs/default.yaml --model mlp_baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from clearml import Task

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.data.download import download_moleculenet
from mol_prop_gnn.data.preprocessing import (
    get_node_feature_dim,
    get_edge_feature_dim,
    preprocess_moleculenet,
)
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.rgcn import MolRGCN
from mol_prop_gnn.models.mlp_baseline import MLPBaseline
from mol_prop_gnn.training.supervised_module import MolPropertyModule
from mol_prop_gnn.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(config: dict, model_name: str | None = None):
    """Instantiate a model from config.

    Parameters
    ----------
    config : dict
        Full config dictionary.
    model_name : str, optional
        Override model name.

    Returns
    -------
    nn.Module
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    name = model_name or model_cfg.get("name", "gcn")

    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    output_dim = data_cfg.get("num_tasks", 1)

    if name == "gcn":
        gcn_cfg = model_cfg.get("gcn", {})
        return MolGCN(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=gcn_cfg.get("hidden_dim", 64),
            num_gnn_layers=gcn_cfg.get("num_gnn_layers", 3),
            decoder_hidden_dim=gcn_cfg.get("decoder_hidden_dim", 64),
            dropout=gcn_cfg.get("dropout", 0.2),
            output_dim=output_dim,
        )
    elif name == "rgcn":
        rgcn_cfg = model_cfg.get("rgcn", {})
        return MolRGCN(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=rgcn_cfg.get("hidden_dim", 64),
            num_layers=rgcn_cfg.get("num_gnn_layers", 3),
            num_relations=rgcn_cfg.get("num_relations", 4),
            decoder_hidden_dim=rgcn_cfg.get("decoder_hidden_dim", 64),
            dropout=rgcn_cfg.get("dropout", 0.2),
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MolPropGNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (gcn, rgcn)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["training"]
    # ── ClearML ───────────────────────────────────────────────────────
    # Initialize ClearML task to capture TensorBoard metrics.
    dataset_name = data_cfg.get("dataset_name", "bbbp")
    Task.init(project_name="MoleculeNet-GCN", task_name=f"train_{dataset_name}")

    # ── Seed ──────────────────────────────────────────────────────────
    pl.seed_everything(train_cfg.get("seed", 42), workers=True)

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Downloading dataset: %s ...", dataset_name)
    csv_path = download_moleculenet(
        dataset_name,
        raw_dir=data_cfg.get("raw_dir", "data/raw"),
    )

    logger.info("Preprocessing molecular graphs ...")
    graphs, train_idx, val_idx, test_idx = preprocess_moleculenet(csv_path, config)

    datamodule = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(config, args.model)
    logger.info("Model: %s", model.__class__.__name__)

    task_type = data_cfg.get("task_type", "classification")
    lit_module = MolPropertyModule(
        model=model,
        task_type=task_type,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        scheduler_config=train_cfg.get("scheduler", {}),
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    # Always monitor val_loss for checkpointing (reliable across all tasks).
    # AUROC/RMSE are logged at epoch-end for tracking but val_loss drives
    # checkpoint selection to avoid sanity-check edge cases.
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    es_cfg = train_cfg.get("early_stopping", {})
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 20),
                mode=es_cfg.get("mode", "min"),
            )
        )

    # ── Trainer ───────────────────────────────────────────────────────
    model_name = args.model or config.get("model", {}).get("name", "gcn")
    logger.info("Initializing trainer for model: %s", model_name)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_cfg.get("save_dir", "lightning_logs"),
        name=model_name,
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 100),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 1),
        deterministic=True,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    trainer.fit(lit_module, datamodule=datamodule, ckpt_path=args.checkpoint)

    # ── Test ──────────────────────────────────────────────────────────
    logger.info("Running test evaluation ...")
    trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")

    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
