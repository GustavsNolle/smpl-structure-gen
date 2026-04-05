"""LightGBM baseline for molecular property prediction.

Operates on molecular fingerprints without using graph structure.
Supports both classification and regression tasks.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


class LightGBMBaseline:
    """LightGBM model for molecular property prediction.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    **params
        Additional LightGBM parameters.
    """

    def __init__(self, task_type: str = "classification", **params) -> None:
        self.task_type = task_type

        if task_type == "classification":
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "binary",
                "verbosity": -1,
            }
        else:
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "regression",
                "verbosity": -1,
            }

        if params:
            default_params.update(params)
        self.params = default_params

        if task_type == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        """Fit the LightGBM model."""
        valid = ~np.isnan(y)
        if eval_set:
            self.model.fit(X[valid], y[valid], eval_set=eval_set)
        else:
            self.model.fit(X[valid], y[valid])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict molecular properties."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate model performance."""
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        if self.task_type == "classification":
            preds = self.predict(X_valid)
            proba = self.predict_proba(X_valid)
            return {
                "accuracy": float(accuracy_score(y_valid, preds)),
                "auroc": float(roc_auc_score(y_valid, proba))
                if len(np.unique(y_valid)) > 1 else 0.0,
            }
        else:
            preds = self.predict(X_valid)
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
                "mae": float(mean_absolute_error(y_valid, preds)),
                "r2": float(r2_score(y_valid, preds)),
            }
