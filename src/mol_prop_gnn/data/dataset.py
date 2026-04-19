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
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

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
        num_workers: int = 8,
        use_balanced_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.graphs = graphs
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_balanced_sampler = use_balanced_sampler

        self.train_dataset: list[Data] = []
        self.val_dataset: list[Data] = []
        self.test_dataset: list[Data] = []
        self.train_sampler: WeightedRandomSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        if isinstance(self.graphs, torch.utils.data.Dataset) and not isinstance(self.graphs, list):
            # Efficient slicing for PyG/Torch datasets
            self.train_dataset = self.graphs[self.train_idx]
            self.val_dataset = self.graphs[self.val_idx]
            self.test_dataset = self.graphs[self.test_idx]
        else:
            # Fallback for standard lists
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
                all_y = torch.cat([g.y for g in dataset]).numpy()
                num_tasks = all_y.shape[1]
                
                if num_tasks == 1:
                    labels = all_y.flatten()
                    valid = labels[~__import__("numpy").isnan(labels)]
                    unique, counts = __import__("numpy").unique(valid, return_counts=True)
                    dist = {f"{int(u)}": int(c) for u, c in zip(unique, counts)}
                    logger.info("  %s class dist: %s (n=%d)", name, dist, len(valid))
                else:
                    # For multi-task, log a summary of total labeled data points
                    valid_mask = ~__import__("numpy").isnan(all_y)
                    total_labeled = valid_mask.sum()
                    logger.info("  %s: %d total labeled data points across %d tasks", name, total_labeled, num_tasks)

        # 1. Compute Balanced Weights for Training Set
        if self.use_balanced_sampler and self.train_dataset:
            logger.info("Computing balanced weights for training set...")
            # y is (num_samples, num_tasks)
            all_y = torch.cat([g.y for g in self.train_dataset])
            # mask of valid labels (not NaN)
            mask = ~torch.isnan(all_y)
            # number of samples per task
            samples_per_task = mask.sum(dim=0).float()
            # weight per task (inverse frequency)
            task_weights = 1.0 / (samples_per_task + 1e-6)
            
            # For each sample, compute its weight as the sum of weights of its labeled tasks
            # This ensures that samples from rare tasks (small datasets) are seen more often.
            sample_weights = (mask.float() * task_weights).sum(dim=1)
            
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            logger.info("Balanced sampler initialized.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def get_degree_histogram(self) -> torch.Tensor:
        """Calculate node degree distribution from the training set.
        Required for PNAConv initialization.
        """
        if not self.train_dataset:
            raise ValueError("Training dataset not setup. Call setup() first.")
            
        # Collect degrees from all training graphs
        degrees = []
        for data in self.train_dataset:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            degrees.append(d)
        
        all_degrees = torch.cat(degrees)
        max_degree = all_degrees.max().item()
        
        # Create histogram
        hist = torch.zeros(max_degree + 1, dtype=torch.long)
        for d in all_degrees:
            hist[d] += 1
            
        return hist
