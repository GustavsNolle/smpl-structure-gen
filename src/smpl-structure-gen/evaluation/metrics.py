"""Evaluation metrics for trade flow prediction.

All metrics operate on log-trade predictions vs ground truth.
"""

from __future__ import annotations

import torch
import numpy as np


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_pred - y_true))


def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error.

    Note: since y_true is log(1+trade), values are always positive.
    """
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))) * 100.0


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> dict[str, float]:
    """Compute all metrics and return as a dictionary."""
    return {
        "rmse": rmse(y_pred, y_true).item(),
        "mae": mae(y_pred, y_true).item(),
        "r2": r_squared(y_pred, y_true).item(),
        "mape": mape(y_pred, y_true).item(),
    }
