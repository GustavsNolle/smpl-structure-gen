#!/usr/bin/env python3
"""Train lightweight Mini Judges for molecular property prediction.

Trains three separate compact GIN models (with residual connections + virtual node)
to predict:
  - SA Score (Synthetic Accessibility, 1-10 scale)
  - QED (Quantitative Estimation of Drug-likeness, 0-1 scale)
  - LogP (Lipophilicity, continuous)

Data source: ZINC250K (~250K drug-like molecules with pre-computed properties)

Usage:
  # Train all three judges:
  python scripts/train_mini_judge.py --property all

  # Train a specific judge:
  python scripts/train_mini_judge.py --property sascore
  python scripts/train_mini_judge.py --property qed
  python scripts/train_mini_judge.py --property logp
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from clearml import Task

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig, QED as QED_module

# RDKit SA Score calculator
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from mol_prop_gnn.data.preprocessing import smiles_to_graph, MoleculeDataset
from mol_prop_gnn.models.mini_judge_gin import MiniJudgeGIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')

# ZINC250K: 250K drug-like molecules from the Grammar VAE paper
# Pre-computed columns: smiles, logP, qed, SAS
ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

# Property column mapping in the ZINC250K CSV
ZINC250K_COLS = {
    "sascore": "SAS",
    "qed": "qed",
    "logp": "logP",
}


# ---------------------------------------------------------------------------
# Data: Download ZINC250K and prepare PyG datasets
# ---------------------------------------------------------------------------

def download_zinc250k(raw_dir: str = "data/raw") -> Path:
    """Download ZINC250K CSV if not already present."""
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "zinc250k.csv"

    if out_path.exists():
        logger.info(f"ZINC250K already exists at {out_path}")
        return out_path

    logger.info(f"Downloading ZINC250K from GitHub...")
    resp = requests.get(ZINC250K_URL, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    logger.info(f"Saved {len(resp.content) / 1e6:.1f} MB to {out_path}")
    return out_path


def compute_property_fallback(smiles: str, prop: str) -> float | None:
    """Compute a property from SMILES using RDKit (fallback if CSV column is missing)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if prop == "sascore":
            return sascorer.calculateScore(mol)
        elif prop == "qed":
            return QED_module.qed(mol)
        elif prop == "logp":
            return Descriptors.MolLogP(mol)
    except Exception:
        return None
    return None


def build_property_dataset(
    prop: str,
    raw_dir: str = "data/raw",
    max_molecules: int = 250_000,
    cache_dir: str = "data/processed",
) -> tuple[list, list]:
    """Load ZINC250K, extract target property, convert to PyG graphs.

    Uses cached graphs if available. Returns (graphs, values).
    """
    cache_path = Path(cache_dir) / f"mini_judge_{prop}_{max_molecules}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["graphs"], cached["values"]

    # Download ZINC250K
    csv_path = download_zinc250k(raw_dir)
    df = pd.read_csv(csv_path)
    logger.info(f"ZINC250K loaded: {len(df)} molecules")

    # Load DiGress-generated molecules if available
    gen_path = Path(raw_dir) / "digress_generated.csv"
    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
        logger.info(f"Loading {len(gen_df)} DiGress-generated molecules from {gen_path}")
        # Standardize column names if necessary (they already match ZINC250K_COLS)
        df = pd.concat([df, gen_df], ignore_index=True)
        # Deduplicate SMILES
        df = df.drop_duplicates(subset=["smiles"])
        logger.info(f"Total dataset size after merging: {len(df)}")

    # Get target column
    target_col = ZINC250K_COLS.get(prop)

    if target_col and target_col in df.columns:
        logger.info(f"Using pre-computed '{target_col}' column from ZINC250K")
        df = df.dropna(subset=["smiles", target_col])
    else:
        logger.info(f"Column '{target_col}' not found, computing {prop} from SMILES...")
        target_col = prop
        df["smiles"] = df["smiles"].astype(str)
        df = df.dropna(subset=["smiles"])
        vals = []
        for smi in tqdm(df["smiles"], desc=f"Computing {prop}"):
            vals.append(compute_property_fallback(smi, prop))
        df[target_col] = vals
        df = df.dropna(subset=[target_col])

    # Subsample if requested
    if len(df) > max_molecules:
        df = df.sample(n=max_molecules, random_state=42)
        logger.info(f"Subsampled to {max_molecules} molecules")

    # Convert to PyG graphs
    graphs = []
    values = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SMILES → PyG graphs"):
        smi = row["smiles"]
        val = float(row[target_col])
        data = smiles_to_graph(smi, y=np.array([val], dtype=np.float32))
        if data is not None:
            graphs.append(data)
            values.append(val)

    logger.info(f"Converted {len(graphs)}/{len(df)} molecules successfully")
    logger.info(f"  {prop} stats: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
                f"min={np.min(values):.3f}, max={np.max(values):.3f}")

    # Cache
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "values": values}, cache_path)
    logger.info(f"Cached to {cache_path}")

    return graphs, values


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class MiniJudgeModule(pl.LightningModule):
    """Lightweight GIN+VirtualNode wrapper for single-property regression."""

    def __init__(
        self,
        prop_name: str,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.prop_name = prop_name
        self.learning_rate = learning_rate

        self.model = MiniJudgeGIN(
            node_input_dim=38,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, batch):
        return self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

    def _shared_step(self, batch, stage: str):
        pred = self(batch)
        target = batch.y.squeeze(-1)

        # Huber loss: robust to outliers (LogP can have extreme values)
        loss = F.huber_loss(pred, target, delta=2.0)

        with torch.no_grad():
            mae = (pred - target).abs().mean()
            ss_res = ((pred - target) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        self.log(f"{stage}_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log(f"{stage}_mae", mae, batch_size=batch.num_graphs, prog_bar=(stage == "val"))
        self.log(f"{stage}_r2", r2, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_single_judge(prop: str, args: argparse.Namespace) -> None:
    """Train a single mini judge for the given property."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Mini Judge for: {prop.upper()}")
    logger.info(f"  Model: MiniJudgeGIN (hidden={args.hidden_dim}, layers={args.num_layers}, VirtualNode=ON)")
    logger.info(f"  Data: ZINC250K ({args.max_molecules} max molecules)")
    logger.info(f"{'='*60}")

    # ClearML tracking
    task = Task.init(
        project_name="MoleculeNet-MiniJudge",
        task_name=f"mini_judge_{prop}",
        output_uri=True,
    )
    task.connect(vars(args))

    # 1. Build dataset
    graphs, values = build_property_dataset(
        prop, raw_dir=args.raw_dir, max_molecules=args.max_molecules
    )

    if len(graphs) < 100:
        logger.error(f"Not enough molecules ({len(graphs)}). Need at least 100.")
        task.close()
        return

    # 2. Split: 80/10/10
    n = len(graphs)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.9 * n)

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_val]]
    test_graphs = [graphs[i] for i in indices[n_val:]]

    logger.info(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    train_loader = DataLoader(MoleculeDataset(train_graphs), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    val_loader = DataLoader(MoleculeDataset(val_graphs), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, persistent_workers=True)
    test_loader = DataLoader(MoleculeDataset(test_graphs), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 3. Model
    module = MiniJudgeModule(
        prop_name=prop,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
    )

    # Log model size
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # 4. Callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        filename=f"mini_judge_{prop}_best",
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_mae",
        patience=15,
        mode="min",
    )

    # 5. Train
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 6. Test
    trainer.test(module, dataloaders=test_loader, ckpt_path="best")

    logger.info(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    logger.info(f"Best val MAE: {checkpoint_cb.best_model_score:.4f}")

    task.close()


def main():
    parser = argparse.ArgumentParser(description="Train Mini Judges (ZINC250K)")
    parser.add_argument("--property", type=str, default="all",
                        choices=["sascore", "qed", "logp", "all"],
                        help="Which property judge to train")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="GIN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of GIN layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--max_molecules", type=int, default=250000,
                        help="Max molecules to use from ZINC250K")
    parser.add_argument("--accelerator", type=str, default="auto")

    args = parser.parse_args()

    properties = ["sascore", "qed", "logp"] if args.property == "all" else [args.property]

    for prop in properties:
        train_single_judge(prop, args)


if __name__ == "__main__":
    main()
