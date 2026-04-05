"""Evaluation metrics for molecular property prediction.

Supports both classification (AUROC, accuracy) and regression (RMSE, MAE, R²).
"""

from __future__ import annotations

import torch
import numpy as np


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    valid = ~torch.isnan(y_true)
    if not valid.any():
        return torch.tensor(0.0)
    return torch.sqrt(torch.mean((y_pred[valid] - y_true[valid]) ** 2))


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    valid = ~torch.isnan(y_true)
    if not valid.any():
        return torch.tensor(0.0)
    return torch.mean(torch.abs(y_pred[valid] - y_true[valid]))


def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination (R²)."""
    valid = ~torch.isnan(y_true)
    if not valid.any():
        return torch.tensor(0.0)
    y_p, y_t = y_pred[valid], y_true[valid]
    ss_res = torch.sum((y_t - y_p) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def auroc(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Area Under the ROC Curve.

    Applies sigmoid to logits before computing AUROC.
    """
    from sklearn.metrics import roc_auc_score

    valid = ~torch.isnan(y_true)
    if not valid.any():
        return 0.0

    y_p = torch.sigmoid(y_pred[valid]).cpu().numpy().flatten()
    y_t = y_true[valid].cpu().numpy().flatten()

    # Need both classes present
    if len(np.unique(y_t)) < 2:
        return 0.0

    try:
        return float(roc_auc_score(y_t, y_p))
    except ValueError:
        return 0.0


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Binary accuracy (threshold at 0.5 on sigmoid output)."""
    valid = ~torch.isnan(y_true)
    if not valid.any():
        return 0.0

    y_p = (torch.sigmoid(y_pred[valid]) > 0.5).float()
    y_t = y_true[valid]

    return float((y_p == y_t).float().mean().item())


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    task_type: str = "classification",
) -> dict[str, float]:
    """Compute all metrics and return as a dictionary.

    Parameters
    ----------
    y_pred : torch.Tensor
        Model predictions (logits for classification).
    y_true : torch.Tensor
        Ground truth labels.
    task_type : str
        'classification' or 'regression'.
    """
    if task_type == "classification":
        return {
            "auroc": auroc(y_pred, y_true),
            "accuracy": accuracy(y_pred, y_true),
        }
    else:
        return {
            "rmse": rmse(y_pred, y_true).item(),
            "mae": mae(y_pred, y_true).item(),
            "r2": r_squared(y_pred, y_true).item(),
        }
