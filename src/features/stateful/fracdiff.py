"""Optimal Fractional Differentiation (d*) with Memory Preservation."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class OptimalFractionalDifferentiator(BaseEstimator, TransformerMixin):
    """Finds minimal real-valued differencing degree d* that achieves stationarity.

    Uses the binomial expansion:
        (1 - L)^d = sum_{k=0}^infty omega_k^{(d)} L^k
        omega_k^{(d)} = -omega_{k-1}^{(d)} * (d - k + 1) / k
    """

    def __init__(
        self,
        price_col: str = "close_adj",
        d_candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        adf_pvalue_threshold: float = 0.05,
        memory_depth: int = 100,
        weight_threshold: float = 1e-4,
    ) -> None:
        """Initialize fractional differentiator.

        Args:
            price_col: Adjusted price series to differentiate.
            d_candidates: Grid of fractional orders to scan.
            adf_pvalue_threshold: Significance level for ADF stationary null rejection.
            memory_depth: Truncation boundary K_max for binomial expansion.
            weight_threshold: Minimum absolute weight threshold for truncation.
        """
        self.price_col = price_col
        self.d_candidates = d_candidates
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.memory_depth = memory_depth
        self.weight_threshold = weight_threshold

        self.optimal_d_: float = 1.0
        self.frozen_weights_: np.ndarray | None = None

    def _get_weights(self, d: float) -> np.ndarray:
        """Compute memory-truncated fractional binomial weights vector."""
        w = [1.0]
        for k in range(1, self.memory_depth):
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < self.weight_threshold:
                break
            w.append(w_k)
        return np.array(w[::-1], dtype=np.float64)

    def _fractional_convolution(self, series: pd.Series, weights: np.ndarray) -> pd.Series:
        """Convolve series forward with frozen lag weights."""
        return series.rolling(window=len(weights)).apply(
            lambda x: np.dot(x, weights), raw=True
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> OptimalFractionalDifferentiator:
        """Estimate optimal order d* on the training fold strictly.

        Args:
            X: Training DataFrame containing price series.
            y: Ignored target parameter.

        Returns:
            OptimalFractionalDifferentiator: Fitted instance with frozen weights.
        """
        if self.price_col not in X.columns:
            raise KeyError(f"Price column '{self.price_col}' missing.")

        p_train = X[self.price_col].dropna()
        self.optimal_d_ = 1.0  # Fallback to integer first differencing

        for d in sorted(self.d_candidates):
            w = self._get_weights(d)
            diff_series = self._fractional_convolution(p_train, w).dropna()
            if len(diff_series) < 30:
                continue

            # Run Augmented Dickey-Fuller (ADF) test
            adf_res = adfuller(diff_series, maxlag=5, autolag="AIC")
            p_val = adf_res[1]

            if p_val < self.adf_pvalue_threshold:
                self.optimal_d_ = d
                logger.info("Optimal FracDiff: Found stationary d* = %.2f (ADF p-val: %.4f)", d, p_val)
                break

        # Freeze learned weight vector
        self.frozen_weights_ = self._get_weights(self.optimal_d_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen convolution weights to out-of-sample partitions.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame augmented with stationary fracdiff price series.
        """
        if self.frozen_weights_ is None:
            raise RuntimeError("OptimalFractionalDifferentiator must be fitted before transforming.")

        df = X.copy()
        col_out = f"fracdiff_d{int(self.optimal_d_ * 100):02d}"
        diff_series = self._fractional_convolution(df[self.price_col], self.frozen_weights_)
        df[col_out] = diff_series.fillna(0.0)
        return df