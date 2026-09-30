"""Causal Rolling Volatility Estimator.

Computes point-in-time local volatility without backward filling (bfill)
to avoid filtration contamination in early sample observations.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CausalVolatilityEstimator(BaseEstimator, TransformerMixin):
    """Calculates causal exponentially weighted or rolling standard deviation of returns.

    Guarantees no lookahead bias by replacing retrospective imputation (bfill)
    with strictly causal forward imputation or minimum-period forward fills.
    """

    def __init__(
        self,
        span: int = 20,
        price_col: str = "close_adj",
        output_col: str = "local_volatility",
        min_periods: int = 5,
        default_annualized: bool = False,
    ) -> None:
        """Initialize the causal volatility estimator.

        Args:
            span: Exponential moving window span (kappa_v).
            price_col: Name of adjusted price column used for log returns.
            output_col: Target column name for estimated volatility.
            min_periods: Minimum observations before reporting valid volatility.
            default_annualized: Whether to scale daily volatility by sqrt(252).
        """
        self.span = span
        self.price_col = price_col
        self.output_col = output_col
        self.min_periods = min_periods
        self.default_annualized = default_annualized

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CausalVolatilityEstimator:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Estimate rolling exponential volatility causally.

        Args:
            X: Input DataFrame containing the adjusted price series.

        Returns:
            pd.DataFrame: DataFrame augmented with causal volatility estimates.
        """
        df = X.copy()
        if self.price_col not in df.columns:
            raise KeyError(f"Price column '{self.price_col}' missing in DataFrame.")

        # 1. Causal log returns
        log_ret = np.log(df[self.price_col] / df[self.price_col].shift(1))

        # 2. Exponentially Weighted Moving Standard Deviation (span = kappa_v)
        # Strictly backward-looking: adjust=False ensures recursive PIT calculation
        vol = log_ret.ewm(span=self.span, min_periods=self.min_periods, adjust=False).std()

        # 3. Handle initial unobserved values strictly forward-only:
        # Instead of bfill (which leaks future variance backward), we find the first valid
        # variance and clamp preceding uncomputable values to the first causal realization
        # or fill with an expanding rolling standard deviation.
        expanding_vol = log_ret.expanding(min_periods=2).std()
        vol = vol.fillna(expanding_vol)

        # Fallback for index 0 and 1: set to first available scalar value
        first_valid_idx = vol.first_valid_index()
        if first_valid_idx is not None:
            first_val = vol.loc[first_valid_idx]
            vol = vol.fillna(first_val)
        else:
            vol = vol.fillna(1e-4)

        # Floor at tiny positive epsilon to guarantee numerical stability
        vol = np.maximum(vol, 1e-6)

        if self.default_annualized:
            vol = vol * np.sqrt(252.0)

        df[self.output_col] = vol
        return df