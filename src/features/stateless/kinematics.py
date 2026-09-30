"""Kinematic Price Derivatives Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class KinematicDerivativeEngine(BaseEstimator, TransformerMixin):
    """Computes higher-order physical kinematic derivatives of price movements."""

    def __init__(
        self,
        price_col: str = "close_adj",
        horizons: tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        """Initialize kinematic derivative engine.

        Args:
            price_col: Adjusted price column name.
            horizons: Multi-horizon step lengths for return momentum and velocity.
        """
        self.price_col = price_col
        self.horizons = horizons

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> KinematicDerivativeEngine:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive velocity, acceleration, and jerk across horizons.

        Args:
            X: Input DataFrame with adjusted price series.

        Returns:
            pd.DataFrame: DataFrame augmented with kinematic derivatives.
        """
        df = X.copy()
        if self.price_col not in df.columns:
            raise KeyError(f"Price column '{self.price_col}' missing.")

        p = df[self.price_col].astype(np.float64)

        # Base log returns: v_t = r_t
        log_ret_1 = np.log(p / p.shift(1)).fillna(0.0)
        df["log_return_momentum"] = log_ret_1  # Velocity v_t
        df["log_return_acceleration"] = (log_ret_1 - log_ret_1.shift(1)).fillna(0.0)  # Acceleration a_t
        df["log_return_jerk"] = (
            log_ret_1 - 2.0 * log_ret_1.shift(1) + log_ret_1.shift(2)
        ).fillna(0.0)  # Jerk j_t

        # Multi-horizon return kinematics
        for h in self.horizons:
            if h == 1:
                continue
            ret_h = np.log(p / p.shift(h)).fillna(0.0)
            df[f"ret_{h}_momentum"] = ret_h
            df[f"ret_{h}_acceleration"] = (ret_h - ret_h.shift(1)).fillna(0.0)

        return df