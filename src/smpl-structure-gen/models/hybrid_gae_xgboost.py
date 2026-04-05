"""Hybrid model combining GAE node embeddings with XGBoost tabular regression."""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, List
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HybridGAEXGBoost:
    """Hybrid model: GAE-based structural embeddings + XGBoost regression.
    
    This model decouples the graph structural learning (GAE) from the 
    tabular interaction learning (XGBoost).
    """

    def __init__(self, **xgb_params) -> None:
        default_params = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50
        }
        if xgb_params:
            default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.embeddings_dict = None

    def set_embeddings(self, embeddings_dict: Dict[int, np.ndarray]) -> None:
        """Load pre-computed node embeddings (mapped by graph/year index)."""
        self.embeddings_dict = embeddings_dict

    def _prepare_hybrid_data(
        self,
        graphs: List,
        start_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine raw features with structural embeddings."""
        if self.embeddings_dict is None:
            raise ValueError("Embeddings must be set via set_embeddings() before preparation.")

        X_hybrid, Y = [], []
        for i, graph in enumerate(graphs):
            idx = start_idx + i
            z = self.embeddings_dict.get(idx)
            if z is None:
                continue
            
            src, dst = graph.edge_index
            h_src = z[src]
            h_dst = z[dst]
            
            # Combine: [node_src, z_src, node_dst, z_dst, edge_attr]
            feat = np.concatenate([
                graph.x[src].numpy(), h_src,
                graph.x[dst].numpy(), h_dst,
                graph.edge_attr.numpy()
            ], axis=1)
            X_hybrid.append(feat)
            Y.append(graph.y.numpy())
            
        return np.concatenate(X_hybrid), np.concatenate(Y)

    def fit(
        self,
        train_graphs: List,
        val_graphs: List,
        train_start_idx: int = 0,
        val_start_idx: int = 0
    ) -> None:
        """Fit XGBoost on the enriched graph data."""
        X_train, y_train = self._prepare_hybrid_data(train_graphs, train_start_idx)
        X_val, y_val = self._prepare_hybrid_data(val_graphs, val_start_idx)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict(self, graphs: List, start_idx: int) -> np.ndarray:
        """Predict trade flows for a list of graphs."""
        X, _ = self._prepare_hybrid_data(graphs, start_idx)
        return self.model.predict(X)

    def evaluate(self, graphs: List, start_idx: int) -> Dict[str, float]:
        """Evaluate performance on test graphs."""
        X, y = self._prepare_hybrid_data(graphs, start_idx)
        preds = self.model.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
