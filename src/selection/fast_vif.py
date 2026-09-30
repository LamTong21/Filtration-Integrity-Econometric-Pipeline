"""Fast Linear Orthogonalization via Inverse-Correlation VIF Module."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FastVIFPruner(BaseEstimator, TransformerMixin):
    """Accelerated O(K^3) Variance Inflation Factor (VIF) collinearity pruner.

    Computes VIF values from the diagonal of the pseudo-inverse of the feature
    correlation matrix:
        VIF_j = diag(R^{-1})_j

    Iteratively drops the feature with the highest VIF until max(VIF) <= threshold.
    """

    def __init__(self, vif_threshold: float = 5.0, min_features: int = 2) -> None:
        """Initialize the fast VIF filter.

        Args:
            vif_threshold: Maximum allowable variance inflation factor (default is 5.0).
            min_features: Minimum floor of features to preserve.
        """
        self.vif_threshold = vif_threshold
        self.min_features = min_features
        self.selected_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FastVIFPruner:
        """Identify collinear features on the input block.

        Args:
            X: Candidate feature matrix (DataFrame).
            y: Ignored target parameter.

        Returns:
            FastVIFPruner: Fitted instance with pruned feature set.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        cols = list(X.columns)
        if len(cols) <= self.min_features:
            self.selected_features_ = cols
            return self

        # Clean constant columns or zero-variance columns first
        variances = X[cols].var(axis=0)
        valid_cols = variances[variances > 1e-12].index.tolist()
        if len(valid_cols) <= self.min_features:
            self.selected_features_ = valid_cols or cols
            return self

        current_cols = valid_cols.copy()

        while len(current_cols) > self.min_features:
            sub_x = X[current_cols].to_numpy(dtype=np.float64)

            # Compute pairwise Pearson correlation matrix
            corr = np.corrcoef(sub_x, rowvar=False)

            # Invert correlation matrix (Moore-Penrose pseudo-inverse for near-singular matrices)
            try:
                corr_inv = np.linalg.pinv(corr)
                vifs = np.diag(corr_inv)
            except np.linalg.LinAlgError:
                logger.warning("VIF Matrix inversion failure. Falling back to SVD pseudo-inverse.")
                corr_inv = np.linalg.pinv(corr, rcond=1e-5)
                vifs = np.diag(corr_inv)

            # Replace NaNs or negative VIFs caused by numerical inaccuracies
            vifs = np.nan_to_num(vifs, nan=1.0, posinf=100.0, neginf=1.0)
            max_idx = int(np.argmax(vifs))
            max_vif = float(vifs[max_idx])

            if max_vif > self.vif_threshold:
                dropped_col = current_cols.pop(max_idx)
                logger.debug("FastVIF: Pruned '%s' with VIF=%.2f", dropped_col, max_vif)
            else:
                break

        self.selected_features_ = current_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter columns by retained orthogonal features.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Column-reduced DataFrame.
        """
        if not self.selected_features_:
            return X
        return X[self.selected_features_]