"""Classical gravity model baseline (log-linear OLS).

The gravity model of trade predicts:

    log(trade_{ij}) ≈ β₀ + β₁ log(GDP_i) + β₂ log(GDP_j)
                     - β₃ log(dist_{ij}) + β₄ contig_{ij} + ...

This serves as the economics baseline that the GNN should outperform.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class GravityBaseline:
    """OLS gravity model baseline.

    Fits a log-linear regression on concatenated node + edge features.

    Parameters
    ----------
    feature_names : list[str], optional
        Readable names for logging purposes.
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names
        self.model = LinearRegression()
        self._is_fitted = False

    def _prepare_features(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
    ) -> np.ndarray:
        """Concatenate source node, dest node, and edge features.

        Parameters
        ----------
        x_src : (E, F_node) source node features
        x_dst : (E, F_node) destination node features
        edge_attr : (E, F_edge) edge features

        Returns
        -------
        (E, 2*F_node + F_edge) combined feature matrix
        """
        return np.hstack([x_src, x_dst, edge_attr])

    def fit(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
    ) -> "GravityBaseline":
        """Fit the OLS gravity model.

        Parameters
        ----------
        x_src, x_dst : (E, F_node)
        edge_attr : (E, F_edge)
        y : (E,) log-trade targets

        Returns
        -------
        self
        """
        X = self._prepare_features(x_src, x_dst, edge_attr)
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info("Gravity baseline fitted. R² (train) = %.4f", self.model.score(X, y))
        return self

    def predict(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
    ) -> np.ndarray:
        """Predict log-trade values.

        Parameters
        ----------
        x_src, x_dst : (E, F_node)
        edge_attr : (E, F_edge)

        Returns
        -------
        (E,) predicted log-trade values
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = self._prepare_features(x_src, x_dst, edge_attr)
        return self.model.predict(X)

    def evaluate(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate predictions against ground truth.

        Returns
        -------
        dict with RMSE, MAE, R² metrics.
        """
        y_pred = self.predict(x_src, x_dst, edge_attr)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }
