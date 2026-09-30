"""Candlestick Shadow Geometry and Intraday Rejection Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ShadowGeometryEngine(BaseEstimator, TransformerMixin):
    """Measures dynamic force and supply/demand rejection via shadow kinematics."""

    def __init__(
        self,
        open_col: str = "open_adj",
        high_col: str = "high_adj",
        low_col: str = "low_adj",
        close_col: str = "close_adj",
        epsilon: float = 1e-8,
        z_span: int = 20,
    ) -> None:
        """Initialize shadow geometry engine.

        Args:
            open_col: Adjusted Open price column.
            high_col: Adjusted High price column.
            low_col: Adjusted Low price column.
            close_col: Adjusted Close price column.
            epsilon: Regularizer preventing zero division in flat candles.
            z_span: Span for rolling standardization.
        """
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.epsilon = epsilon
        self.z_span = z_span

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ShadowGeometryEngine:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive upper/lower shadow ratios and kinematic derivatives.

        Args:
            X: Input DataFrame with adjusted OHLC bars.

        Returns:
            pd.DataFrame: DataFrame augmented with shadow geometric metrics.
        """
        df = X.copy()
        o = df[self.open_col].to_numpy(dtype=np.float64)
        h = df[self.high_col].to_numpy(dtype=np.float64)
        l = df[self.low_col].to_numpy(dtype=np.float64)
        c = df[self.close_col].to_numpy(dtype=np.float64)

        hl_range = (h - l) + self.epsilon
        max_oc = np.maximum(o, c)
        min_oc = np.minimum(o, c)

        upper_shadow = (h - max_oc) / hl_range
        lower_shadow = (min_oc - l) / hl_range

        s_upper = pd.Series(upper_shadow, index=df.index)
        s_lower = pd.Series(lower_shadow, index=df.index)

        # Upper shadow dynamics (Supply pressure / Intraday rejection)
        df["geo_upper_shadow_momentum"] = s_upper
        df["geo_upper_shadow_acceleration"] = s_upper - s_upper.shift(1).fillna(0.0)

        # Lower shadow dynamics (Demand absorption / Thrust)
        df["geo_lower_shadow_momentum"] = s_lower
        df["geo_lower_shadow_acceleration"] = s_lower - s_lower.shift(1).fillna(0.0)

        # Rolling z-scores of upper shadow (stationary bounded indicator)
        rolling_mean = s_upper.rolling(window=self.z_span, min_periods=5).mean()
        rolling_std = s_upper.rolling(window=self.z_span, min_periods=5).std().replace(0.0, np.nan)
        z_upper = ((s_upper - rolling_mean) / rolling_std).fillna(0.0)
        df["geo_upper_shadow_zscaled_lag3"] = z_upper.shift(3).fillna(0.0)

        return df