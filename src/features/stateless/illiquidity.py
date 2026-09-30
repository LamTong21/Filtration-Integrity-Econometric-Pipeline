"""Amihud Illiquidity Proxy and Friction Kinematics Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmihudIlliquidityEngine(BaseEstimator, TransformerMixin):
    """Calculates standardized Amihud price impact friction and its kinematic derivatives."""

    def __init__(
        self,
        window: int = 20,
        price_col: str = "close_adj",
        volume_col: str = "volume",
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize illiquidity engine.

        Args:
            window: Rolling normalization window.
            price_col: Price column for return calculation.
            volume_col: Volume column representing turnover impact.
            epsilon: Regularizer preventing division by zero.
        """
        self.window = window
        self.price_col = price_col
        self.volume_col = volume_col
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> AmihudIlliquidityEngine:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract Amihud price impact z-scores and higher-order derivatives.

        Args:
            X: Input DataFrame with price and volume.

        Returns:
            pd.DataFrame: Augmented DataFrame with friction derivatives.
        """
        df = X.copy()
        c = df[self.price_col]
        v = df[self.volume_col].astype(np.float64)

        abs_ret = np.abs(np.log(c / c.shift(1))).fillna(0.0)
        dollar_volume_proxy = (c * v) + self.epsilon
        amihud_raw = (abs_ret / dollar_volume_proxy) * 1e6

        # Rolling standardization
        r_mean = amihud_raw.rolling(window=self.window, min_periods=5).mean()
        r_std = amihud_raw.rolling(window=self.window, min_periods=5).std().replace(0.0, np.nan)
        z_amihud = ((amihud_raw - r_mean) / (r_std + self.epsilon)).fillna(0.0)

        # Kinematic derivatives of illiquidity shocks
        df["liq_amihud_z"] = z_amihud
        df["liq_amihud_z_momentum"] = (z_amihud - z_amihud.shift(1)).fillna(0.0)
        df["liq_amihud_z_acceleration"] = (
            z_amihud - 2.0 * z_amihud.shift(1) + z_amihud.shift(2)
        ).fillna(0.0)

        return df