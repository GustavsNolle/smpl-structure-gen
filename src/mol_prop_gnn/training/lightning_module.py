"""PyTorch Lightning module for molecular property prediction.

Supports both classification (BCE loss + AUROC) and regression
(MSE loss + RMSE/MAE) tasks, auto-detected from config.

AUROC and accuracy are computed at the epoch level (aggregated across
all batches) rather than per-batch to avoid averaging issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class MolPropertyModule(pl.LightningModule):
    """Lightning wrapper for molecular property models.

    Parameters
    ----------
    model : nn.Module
        Any model that accepts ``(x, edge_index, edge_attr, batch)``
        and returns ``(B, num_tasks)`` predictions.
    task_type : str
        'classification' or 'regression'.
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
        task_type: str = "classification",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}

        if task_type == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

        # Accumulate predictions for epoch-level metrics
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        kwargs = {"batch": data.batch} if hasattr(data, "batch") else {}
        if hasattr(data, "edge_type"):
            try:
                return self.model(
                    data.x, data.edge_index, data.edge_attr,
                    edge_type=data.edge_type, **kwargs,
                )
            except TypeError:
                return self.model(data.x, data.edge_index, data.edge_attr, **kwargs)
        return self.model(data.x, data.edge_index, data.edge_attr, **kwargs)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute masked loss (handles NaN targets for multi-task)."""
        valid_mask = ~torch.isnan(y_true)
        if valid_mask.any():
            return self.loss_fn(y_pred[valid_mask], y_true[valid_mask])
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    @staticmethod
    def _compute_epoch_auroc(all_preds: list[torch.Tensor], all_targets: list[torch.Tensor]) -> float:
        """Compute AUROC over all accumulated predictions."""
        from sklearn.metrics import roc_auc_score

        if not all_preds:
            return 0.0
        preds = torch.cat(all_preds).cpu().numpy().flatten()
        targets = torch.cat(all_targets).cpu().numpy().flatten()

        # Remove NaN targets
        valid = ~np.isnan(targets)
        if not valid.any():
            return 0.0
        preds, targets = preds[valid], targets[valid]

        # Apply sigmoid to logits
        preds = 1.0 / (1.0 + np.exp(-preds))

        # Need both classes
        if len(np.unique(targets)) < 2:
            return 0.0

        try:
            return float(roc_auc_score(targets, preds))
        except ValueError:
            return 0.0

    @staticmethod
    def _compute_epoch_accuracy(all_preds: list[torch.Tensor], all_targets: list[torch.Tensor]) -> float:
        """Compute accuracy over all accumulated predictions."""
        if not all_preds:
            return 0.0
        preds = torch.cat(all_preds).cpu()
        targets = torch.cat(all_targets).cpu()

        valid = ~torch.isnan(targets)
        if not valid.any():
            return 0.0

        pred_labels = (torch.sigmoid(preds[valid]) > 0.5).float()
        return float((pred_labels == targets[valid]).float().mean().item())

    # ── Training ─────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        y_pred = self.forward(batch)
        y_true = batch.y
        loss = self._compute_loss(y_pred, y_true)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=y_pred.shape[0])
        return loss

    # ── Validation ───────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def validation_step(self, batch, batch_idx: int) -> None:
        y_pred = self.forward(batch)
        y_true = batch.y
        loss = self._compute_loss(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=y_pred.shape[0])

        self._val_preds.append(y_pred.detach())
        self._val_targets.append(y_true.detach())

    def on_validation_epoch_end(self) -> None:
        if self.task_type == "classification":
            auroc = self._compute_epoch_auroc(self._val_preds, self._val_targets)
            acc = self._compute_epoch_accuracy(self._val_preds, self._val_targets)
            self.log("val_auroc", auroc, prog_bar=True)
            self.log("val_accuracy", acc)
        else:
            preds = torch.cat(self._val_preds).cpu()
            targets = torch.cat(self._val_targets).cpu()
            valid = ~torch.isnan(targets)
            if valid.any():
                rmse = torch.sqrt(torch.mean((preds[valid] - targets[valid]) ** 2)).item()
                mae_val = torch.mean(torch.abs(preds[valid] - targets[valid])).item()
                self.log("val_rmse", rmse, prog_bar=True)
                self.log("val_mae", mae_val)

    # ── Test ─────────────────────────────────────────────────────────

    def on_test_epoch_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()

    def test_step(self, batch, batch_idx: int) -> None:
        y_pred = self.forward(batch)
        y_true = batch.y
        loss = self._compute_loss(y_pred, y_true)
        self.log("test_loss", loss, on_epoch=True, batch_size=y_pred.shape[0])

        self._test_preds.append(y_pred.detach())
        self._test_targets.append(y_true.detach())

    def on_test_epoch_end(self) -> None:
        if self.task_type == "classification":
            auroc = self._compute_epoch_auroc(self._test_preds, self._test_targets)
            acc = self._compute_epoch_accuracy(self._test_preds, self._test_targets)
            self.log("test_auroc", auroc, prog_bar=True)
            self.log("test_accuracy", acc)
        else:
            preds = torch.cat(self._test_preds).cpu()
            targets = torch.cat(self._test_targets).cpu()
            valid = ~torch.isnan(targets)
            if valid.any():
                rmse = torch.sqrt(torch.mean((preds[valid] - targets[valid]) ** 2)).item()
                mae_val = torch.mean(torch.abs(preds[valid] - targets[valid])).item()
                r2 = 1.0 - (
                    torch.sum((targets[valid] - preds[valid]) ** 2)
                    / (torch.sum((targets[valid] - torch.mean(targets[valid])) ** 2) + 1e-8)
                ).item()
                self.log("test_rmse", rmse, prog_bar=True)
                self.log("test_mae", mae_val)
                self.log("test_r2", r2)

    # ── Optimizer ────────────────────────────────────────────────────

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
