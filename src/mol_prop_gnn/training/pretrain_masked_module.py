"""Lightning module for Phase 1: Continuous Masked Node Prediction pre-training.

Wraps the standard MolGINE encoder, but operates on node-level embeddings.
- Computes MSE loss between predicted 38-dim RDKit features and true features.
- Loss is ONLY computed on the 15% masked nodes.
- Exposes `get_encoder_state_dict()` to extract the MolGINE backbone.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContinuousMaskTransform:
    """Masks 38-dimensional continuous/one-hot RDKit atom features.

    Replaces selected nodes with an all -1.0 mask vector to easily distinguish
    from valid RDKit features (which are typically 0.0 or 1.0).
    """
    def __init__(self, mask_rate: float = 0.15):
        self.mask_rate = mask_rate
        self.mask_value = -1.0

    def __call__(self, data):
        num_nodes = data.x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_rate))

        # Store true continuous features BEFORE masking
        data.node_labels = data.x.clone()

        # Boolean mask
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        perm = torch.randperm(num_nodes)
        node_mask[perm[:num_mask]] = True
        data.node_mask = node_mask

        # Apply [MASK] vector to selected nodes
        data.x = data.x.clone()
        data.x[node_mask] = self.mask_value

        return data


class MaskedNodePredModule(pl.LightningModule):
    """Lightning module for continuous self-supervised pretraining."""

    def __init__(
        self,
        backbone,
        node_dim: int = 38,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.pos_weight = pos_weight
        
        # We need the jk_dim to build the prediction head.
        # MolGINE out_channels property returns the exact jk_dim.
        jk_dim = self.backbone.out_channels
        
        self.prediction_head = nn.Sequential(
            nn.Linear(jk_dim, self.backbone.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3 if not hasattr(self.backbone, 'dropout') else 0.3),
            nn.Linear(self.backbone.hidden_dim, node_dim),
        )

        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, batch):
        # Encode returns node embeddings (N, jk_dim) BEFORE pooling
        h = self.backbone.encode(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        return self.prediction_head(h)

    def _shared_step(self, batch, stage: str):
        """Compute MSE loss on masked nodes only."""
        preds = self(batch)              # (N_total, node_dim)
        mask = batch.node_mask           # (N_total,) boolean
        labels = batch.node_labels       # (N_total, node_dim) float
        
        # Extract predictions and labels for masked nodes only
        masked_preds = preds[mask]
        masked_labels = labels[mask]

        loss = F.binary_cross_entropy_with_logits(
            masked_preds, 
            masked_labels.float(),
            pos_weight=torch.tensor([self.pos_weight], device=self.device)
        )

        self.log(f"{stage}_loss", loss, batch_size=mask.sum(), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        
    def get_encoder_state_dict(self) -> dict:
        """Extract the MolGINE backbone state dict for Phase 2 fine-tuning."""
        return self.backbone.state_dict()
