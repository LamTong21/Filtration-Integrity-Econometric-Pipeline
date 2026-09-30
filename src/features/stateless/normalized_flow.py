"""Normalized and Finite-Window Volume Flow Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NormalizedVolumeFlowEngine(BaseEstimator, TransformerMixin):
    """Transforms unbounded accumulators into stationary finite-window flow features."""

    def __init__(
        self,
        window: int = 20,
        price_col: str = "close_adj",
        high_col: str = "high_adj",
        low_col: str = "low_adj",
        volume_col: str = "volume",
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize flow engine.

        Args:
            window: Rolling observation window (W).
            price_col: Adjusted close column.
            high_col: Adjusted high column.
            low_col: Adjusted low column.
            volume_col: Trading volume column.
            epsilon: Regularizer against zero division.
        """
        self.window = window
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NormalizedVolumeFlowEngine:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate standardized finite rolling OBV, CMF, and VPT.

        Args:
            X: Input DataFrame with OHLCV columns.

        Returns:
            pd.DataFrame: DataFrame augmented with stationary flow features.
        """
        df = X.copy()
        c = df[self.price_col]
        h = df[self.high_col]
        l = df[self.low_col]
        v = df[self.volume_col].astype(np.float64)

        # 1. Rolling Standardized On-Balance Volume (Z_t^OBV)
        ret_1 = c.diff().fillna(0.0)
        direction = np.sign(ret_1)
        directed_vol = direction * v
        rolling_obv = directed_vol.rolling(window=self.window, min_periods=5).sum()
        obv_mean = rolling_obv.rolling(window=self.window, min_periods=5).mean()
        obv_std = rolling_obv.rolling(window=self.window, min_periods=5).std().replace(0.0, np.nan)
        df["z_obv_flow"] = ((rolling_obv - obv_mean) / (obv_std + self.epsilon)).fillna(0.0)

        # 2. Chaikin Money Flow (CMF) over window W
        clv = ((c - l) - (h - c)) / ((h - l) + self.epsilon)
        vol_money = clv * v
        cmf = vol_money.rolling(window=self.window, min_periods=5).sum() / (
            v.rolling(window=self.window, min_periods=5).sum() + self.epsilon
        )
        df["cmf"] = cmf.fillna(0.0)
        df["cmf_slope_5"] = (cmf - cmf.shift(5)).fillna(0.0)

        # 3. Finite-window Volume-Price Trend (VPT)
        pct_chg = c.pct_change().fillna(0.0)
        vpt_rolling = (pct_chg * v).rolling(window=self.window, min_periods=5).sum()
        df["vpt_flow"] = vpt_rolling.fillna(0.0)

        return df