"""RDKit descriptor-based baseline (Random Forest on molecular fingerprints).

This serves as the cheminformatics baseline — a non-DL approach using
hand-crafted molecular descriptors and fingerprints, analogous to how
the gravity model baseline worked for trade flows.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class RDKitBaseline:
    """Random Forest baseline operating on RDKit molecular descriptors.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    n_estimators : int
        Number of trees.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 500,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        if task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RDKitBaseline":
        """Fit the Random Forest model.

        Parameters
        ----------
        X : (N, D) molecular descriptors or fingerprints
        y : (N,) targets
        """
        # Remove NaN targets
        valid = ~np.isnan(y)
        self.model.fit(X[valid], y[valid])
        self._is_fitted = True
        logger.info("RDKit baseline fitted on %d molecules.", valid.sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict molecular properties."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate predictions against ground truth."""
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        if self.task_type == "classification":
            y_pred = self.predict(X_valid)
            y_proba = self.predict_proba(X_valid)
            metrics = {
                "accuracy": float(accuracy_score(y_valid, y_pred)),
                "auroc": float(roc_auc_score(y_valid, y_proba))
                if len(np.unique(y_valid)) > 1 else 0.0,
            }
        else:
            y_pred = self.predict(X_valid)
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
                "mae": float(mean_absolute_error(y_valid, y_pred)),
                "r2": float(r2_score(y_valid, y_pred)),
            }

        return metrics
