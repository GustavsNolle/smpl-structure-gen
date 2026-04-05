"""Hybrid model combining GAE molecular embeddings with XGBoost."""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Optional

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.nn import global_mean_pool


class HybridGAEXGBoost:
    """Hybrid model: GAE-based molecular embeddings + XGBoost.

    Decouples graph structural learning (GAE) from the
    property prediction (XGBoost).

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    **xgb_params
        Additional XGBoost parameters.
    """

    def __init__(self, task_type: str = "classification", **xgb_params) -> None:
        self.task_type = task_type

        if task_type == "classification":
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "binary:logistic",
                "early_stopping_rounds": 50,
            }
        else:
            default_params = {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "reg:squarederror",
                "early_stopping_rounds": 50,
            }

        if xgb_params:
            default_params.update(xgb_params)

        if task_type == "classification":
            self.model = xgb.XGBClassifier(**default_params)
        else:
            self.model = xgb.XGBRegressor(**default_params)

        self.gae_model = None

    def set_gae_model(self, gae_model) -> None:
        """Set the pre-trained GAE model for embedding extraction."""
        self.gae_model = gae_model

    def _extract_features(self, graphs: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract hybrid features: raw atom features + GAE embeddings."""
        if self.gae_model is None:
            raise ValueError("GAE model must be set via set_gae_model().")

        X_all, Y_all = [], []
        self.gae_model.eval()

        with torch.no_grad():
            for graph in graphs:
                # GAE structural embeddings
                z = self.gae_model.encode(graph.x, graph.edge_index)

                # Pool to graph level
                batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                x_pooled = global_mean_pool(graph.x, batch).numpy()
                z_pooled = global_mean_pool(z, batch).numpy()

                feat = np.concatenate([x_pooled, z_pooled], axis=1)
                X_all.append(feat)
                Y_all.append(graph.y.numpy().flatten())

        return np.concatenate(X_all), np.concatenate(Y_all)

    def fit(
        self,
        train_graphs: list,
        val_graphs: list,
    ) -> None:
        """Fit XGBoost on hybrid features."""
        X_train, y_train = self._extract_features(train_graphs)
        X_val, y_val = self._extract_features(val_graphs)

        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)

        self.model.fit(
            X_train[valid_train], y_train[valid_train],
            eval_set=[(X_val[valid_val], y_val[valid_val])],
            verbose=False,
        )

    def predict(self, graphs: list) -> np.ndarray:
        """Predict molecular properties."""
        X, _ = self._extract_features(graphs)
        return self.model.predict(X)

    def evaluate(self, graphs: list) -> dict[str, float]:
        """Evaluate performance."""
        X, y = self._extract_features(graphs)
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        if self.task_type == "classification":
            preds = self.model.predict(X_valid)
            proba = self.model.predict_proba(X_valid)[:, 1]
            return {
                "accuracy": float(accuracy_score(y_valid, preds)),
                "auroc": float(roc_auc_score(y_valid, proba))
                if len(np.unique(y_valid)) > 1 else 0.0,
            }
        else:
            preds = self.model.predict(X_valid)
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
                "mae": float(mean_absolute_error(y_valid, preds)),
                "r2": float(r2_score(y_valid, preds)),
            }
