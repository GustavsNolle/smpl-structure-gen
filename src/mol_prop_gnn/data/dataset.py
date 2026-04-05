"""PyTorch Geometric dataset and Lightning DataModule for molecular graphs.

Each molecule is a ``torch_geometric.data.Data`` object with:
- ``x``: atom features ``(N_atoms, F_atom)``
- ``edge_index``: bonds ``(2, N_bonds*2)`` (undirected)
- ``edge_attr``: bond features ``(N_bonds*2, F_bond)``
- ``edge_type``: bond type indices for RGCN ``(N_bonds*2,)``
- ``y``: molecular property target ``(1, num_tasks)``
- ``smiles``: original SMILES string
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class MoleculeDataModule(pl.LightningDataModule):
    """Lightning DataModule for molecular graph datasets.

    Parameters
    ----------
    graphs : list[Data]
        All molecular graph Data objects.
    train_idx : list[int]
        Indices into ``graphs`` for training set.
    val_idx : list[int]
        Indices for validation set.
    test_idx : list[int]
        Indices for test set.
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of data loading workers.
    """

    def __init__(
        self,
        graphs: list[Data],
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.graphs = graphs
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: list[Data] = []
        self.val_dataset: list[Data] = []
        self.test_dataset: list[Data] = []

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = [self.graphs[i] for i in self.train_idx]
        self.val_dataset = [self.graphs[i] for i in self.val_idx]
        self.test_dataset = [self.graphs[i] for i in self.test_idx]

        logger.info(
            "Data split — train: %d, val: %d, test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        # Log class distribution per split (critical for scaffold splits)
        for name, dataset in [
            ("train", self.train_dataset),
            ("val", self.val_dataset),
            ("test", self.test_dataset),
        ]:
            if dataset:
                import torch
                labels = torch.cat([g.y for g in dataset]).numpy().flatten()
                valid = labels[~__import__("numpy").isnan(labels)]
                unique, counts = __import__("numpy").unique(valid, return_counts=True)
                dist = {f"{int(u)}": int(c) for u, c in zip(unique, counts)}
                logger.info("  %s class dist: %s (n=%d)", name, dist, len(valid))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
