"""Dynamic Asymmetric Triple Barrier Method Labeling.

Constructs institutional-grade meta-labels and barrier timestamps for directional classification.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class DynamicTripleBarrierLabeler(BaseEstimator, TransformerMixin):
    """Labels financial time series using dynamic, volatility-scaled barriers.

    For entry at discrete time t over horizon h:
        Upper Barrier (Take Profit): U_t = P_t^{adj} * (1.0 + kappa_pt * sigma_t)
        Lower Barrier (Stop Loss):   L_t = P_t^{adj} * (1.0 - kappa_sl * sigma_t)
        Expiration Barrier:          tau = t + h

    Binary target Y_t = 1 if upper barrier is touched before lower barrier or timeout; 0 otherwise.
    """

    def __init__(
        self,
        horizon: int = 5,
        kappa_pt: float = 1.0,
        kappa_sl: float = 1.0,
        price_col: str = "close_adj",
        volatility_col: str = "local_volatility",
        high_col: str = "high_adj",
        low_col: str = "low_adj",
    ) -> None:
        """Initialize labeling parameters.

        Args:
            horizon: Maximum settlement bars before temporal expiration (h).
            kappa_pt: Multiplier for upper take-profit barrier.
            kappa_sl: Multiplier for lower stop-loss barrier.
            price_col: Column name representing base entry price.
            volatility_col: Column name representing causal local volatility sigma_t.
            high_col: Column name representing contemporaneous intraday high.
            low_col: Column name representing contemporaneous intraday low.
        """
        self.horizon = horizon
        self.kappa_pt = kappa_pt
        self.kappa_sl = kappa_sl
        self.price_col = price_col
        self.volatility_col = volatility_col
        self.high_col = high_col
        self.low_col = low_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DynamicTripleBarrierLabeler:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute triple barrier metadata and binary class targets.

        Args:
            X: Input DataFrame with adjusted prices and local volatility.

        Returns:
            pd.DataFrame: DataFrame containing metadata columns:
                ['target', 'barrier_return', 'touch_timestamp', 'barrier_type']
        """
        df = X.copy()
        for col in (self.price_col, self.volatility_col):
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing for labeling.")

        has_high_low = (self.high_col in df.columns) and (self.low_col in df.columns)

        p = df[self.price_col].to_numpy(dtype=np.float64)
        sigma = df[self.volatility_col].to_numpy(dtype=np.float64)
        p_high = df[self.high_col].to_numpy(dtype=np.float64) if has_high_low else p
        p_low = df[self.low_col].to_numpy(dtype=np.float64) if has_high_low else p

        n = len(df)
        timestamps = df.index

        target = np.zeros(n, dtype=np.int32)
        barrier_ret = np.zeros(n, dtype=np.float64)
        touch_timestamp = np.empty(n, dtype="object")
        barrier_type = np.empty(n, dtype="object")

        for i in range(n):
            entry_price = p[i]
            local_vol = sigma[i]

            u_barrier = entry_price * (1.0 + self.kappa_pt * local_vol)
            l_barrier = entry_price * (1.0 - self.kappa_sl * local_vol)

            end_idx = min(i + self.horizon, n - 1)

            # If at the very edge of the time-series
            if i == end_idx:
                target[i] = 0
                barrier_ret[i] = 0.0
                touch_timestamp[i] = timestamps[i]
                barrier_type[i] = "timeout"
                continue

            hit = False
            for step, j in enumerate(range(i + 1, end_idx + 1), start=1):
                cur_high = p_high[j]
                cur_low = p_low[j]
                cur_close = p[j]

                upper_touch = cur_high >= u_barrier
                lower_touch = cur_low <= l_barrier

                if upper_touch and not lower_touch:
                    target[i] = 1
                    barrier_ret[i] = (u_barrier / entry_price) - 1.0
                    touch_timestamp[i] = timestamps[j]
                    barrier_type[i] = "take_profit"
                    hit = True
                    break
                elif lower_touch and not upper_touch:
                    target[i] = 0
                    barrier_ret[i] = (l_barrier / entry_price) - 1.0
                    touch_timestamp[i] = timestamps[j]
                    barrier_type[i] = "stop_loss"
                    hit = True
                    break
                elif upper_touch and lower_touch:
                    # Intraday conflict: evaluate closer touch or conservative loss
                    target[i] = 0
                    barrier_ret[i] = (l_barrier / entry_price) - 1.0
                    touch_timestamp[i] = timestamps[j]
                    barrier_type[i] = "stop_loss"
                    hit = True
                    break

            if not hit:
                # Vertical barrier (horizon expiration timeout)
                exit_price = p[end_idx]
                target[i] = 1 if exit_price >= u_barrier else 0
                barrier_ret[i] = (exit_price / entry_price) - 1.0
                touch_timestamp[i] = timestamps[end_idx]
                barrier_type[i] = "timeout"

        result = pd.DataFrame(
            {
                "target": target,
                "barrier_return": barrier_ret,
                "touch_timestamp": touch_timestamp,
                "barrier_type": barrier_type,
            },
            index=df.index,
        )
        return result