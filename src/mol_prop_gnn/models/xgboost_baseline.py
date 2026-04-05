"""XGBoost baseline for molecular property prediction.

Operates on molecular fingerprints (Morgan/ECFP) without using
graph structure. Supports both classification and regression tasks.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


class XGBoostBaseline:
    """XGBoost model for molecular property prediction.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    **params
        Additional XGBoost parameters.
    """

    def __init__(self, task_type: str = "classification", **params) -> None:
        self.task_type = task_type

        if task_type == "classification":
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "early_stopping_rounds": 50,
            }
        else:
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "reg:squarederror",
                "early_stopping_rounds": 50,
            }

        if params:
            default_params.update(params)
        self.params = default_params

        if task_type == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        """Fit the XGBoost model."""
        valid = ~np.isnan(y)
        if eval_set:
            self.model.fit(X[valid], y[valid], eval_set=eval_set, verbose=False)
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
