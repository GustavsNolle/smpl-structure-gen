"""PyTorch Lightning module for trade flow prediction.

Wraps any model (GCN or MLP) with a standard training loop,
validation, test evaluation, optimizer, and LR scheduling.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from trade_flow_gcn.evaluation.metrics import compute_all_metrics


class TradeFlowModule(pl.LightningModule):
    """Lightning wrapper for trade flow models.

    Parameters
    ----------
    model : nn.Module
        Any model that accepts ``(x, edge_index, edge_attr, [edge_type])``
        and returns ``(E,)`` predictions.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization.
    scheduler_config : dict, optional
        LR scheduler settings.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        if hasattr(data, 'edge_type'):
            try:
                return self.model(data.x, data.edge_index, data.edge_attr, edge_type=data.edge_type)
            except TypeError:
                return self.model(data.x, data.edge_index, data.edge_attr)
        return self.model(data.x, data.edge_index, data.edge_attr)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        y_pred = self.forward(batch)
        y_true = batch.y
        loss = self.loss_fn(y_pred, y_true)

        metrics = compute_all_metrics(
            y_pred.detach(), 
            y_true.detach()
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_rmse", metrics["rmse"], on_epoch=True)
        self.log(f"{stage}_mae", metrics["mae"], on_epoch=True)
        self.log(f"{stage}_r2", metrics["r2"], on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        config: dict[str, Any] = {"optimizer": optimizer}

        sched_name = self.scheduler_config.get("name", "")
        if sched_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.scheduler_config.get("patience", 10),
                factor=self.scheduler_config.get("factor", 0.5),
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get("T_max", 100),
            )
            config["lr_scheduler"] = {"scheduler": scheduler}

        return config
