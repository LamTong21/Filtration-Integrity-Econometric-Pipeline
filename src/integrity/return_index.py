"""Forward-Compounded Total Return Synthesis Module.

Prevents Radon-Nikodym forward-looking informational leakage (Filtration Contamination)
introduced by standard retrospective (backward) dividend/split adjustments.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ForwardReturnSynthesizer(BaseEstimator, TransformerMixin):
    """Constructs a strictly non-anticipative forward total return index.

    Given raw close P_t, dividend D_t, and split factor S_t:
        R_t = (P_t + D_t) / (P_{t-1} * S_t) - 1.0
        P_t^{adj} = P_0^{adj} * Prod_{i=1}^t (1.0 + R_i)

    High, Low, and Open are adjusted synchronously via the contemporaneous scalar:
        omega_t = P_t^{adj} / P_t
    """

    def __init__(
        self,
        base_index_value: float = 1000.0,
        raw_price_cols: tuple[str, str, str, str] = ("open", "high", "low", "close"),
        dividend_col: str = "Dividends",
        split_col: str = "Stock Splits",
        prefix: str = "",
    ) -> None:
        """Initialize the forward synthesizer.

        Args:
            base_index_value: Arbitrary origin P_0^{adj} for synthetic capitalization index.
            raw_price_cols: Raw OHLC column names.
            dividend_col: Column indicating cash dividend payouts on ex-date t.
            split_col: Column indicating stock split ratio on ex-date t (1.0 = no split).
            prefix: Optional prefix for generated forward-adjusted columns.
        """
        self.base_index_value = base_index_value
        self.raw_open, self.raw_high, self.raw_low, self.raw_close = raw_price_cols
        self.dividend_col = dividend_col
        self.split_col = split_col
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ForwardReturnSynthesizer:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Synthesize forward total return series and point-in-time adjusted OHLC bars.

        Args:
            X: Input DataFrame with raw prices and corporate action event vectors.

        Returns:
            pd.DataFrame: Augmented DataFrame including strictly adapted adjusted OHLC.
        """
        df = X.copy()

        # Fill missing corporate actions with neutral values if columns are absent
        if self.dividend_col not in df.columns:
            df[self.dividend_col] = 0.0
        else:
            df[self.dividend_col] = df[self.dividend_col].fillna(0.0)

        if self.split_col not in df.columns:
            df[self.split_col] = 1.0
        else:
            # Splits of 0.0 are invalid (neutral is 1.0)
            df[self.split_col] = df[self.split_col].replace(0.0, 1.0).fillna(1.0)

        p_close = df[self.raw_close].to_numpy(dtype=np.float64)
        div = df[self.dividend_col].to_numpy(dtype=np.float64)
        split = df[self.split_col].to_numpy(dtype=np.float64)

        n = len(df)
        if n == 0:
            return df

        # Vectorized calculation of point-in-time discrete return R_t
        r_t = np.zeros(n, dtype=np.float64)
        prev_close_scaled = p_close[:-1] * split[1:]

        # Handle potential zero-division safely
        valid_denom = prev_close_scaled > 0.0
        r_t[1:][valid_denom] = (p_close[1:][valid_denom] + div[1:][valid_denom]) / prev_close_scaled[valid_denom] - 1.0

        # Construct cumulative forward-compounded total return index: P_t^{adj}
        gross_returns = 1.0 + r_t
        gross_returns[0] = 1.0  # Index baseline origin
        p_adj = self.base_index_value * np.cumprod(gross_returns)

        # Scale factor: omega_t = P_t^{adj} / P_t
        with np.errstate(divide="ignore", invalid="ignore"):
            omega_t = np.where(p_close > 0.0, p_adj / p_close, 1.0)

        # Compute point-in-time adapted OHLC
        col_close_adj = f"{self.prefix}close_adj"
        col_open_adj = f"{self.prefix}open_adj"
        col_high_adj = f"{self.prefix}high_adj"
        col_low_adj = f"{self.prefix}low_adj"
        col_pit_return = f"{self.prefix}pit_return"

        df[col_close_adj] = p_adj
        df[col_open_adj] = df[self.raw_open] * omega_t
        df[col_high_adj] = df[self.raw_high] * omega_t
        df[col_low_adj] = df[self.raw_low] * omega_t
        df[col_pit_return] = r_t

        return df