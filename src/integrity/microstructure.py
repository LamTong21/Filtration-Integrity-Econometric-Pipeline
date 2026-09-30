"""Microstructure and Depth Topology Engine.

Computes multi-level order book features including Spatially Decayed Order Imbalance (OBI)
and Depth Micro-Price to extract liquidity friction dynamics.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class MicrostructureTopologyEngine(BaseEstimator, TransformerMixin):
    """Calculates order queue dynamics across M depth levels.

    Implements spatial exponential decay weights:
        w_i = exp(-lambda * (i - 1)) for level i in {1, ..., M}

    Metrics extracted:
        - Weighted Bid/Ask Depth: V_t^b, V_t^a
        - Spatially Decayed OBI_t = (V_t^b - V_t^a) / (V_t^b + V_t^a + epsilon)
        - Depth Micro-Price P_t^{micro}
        - Microstructure Directional Deviation: delta_t^{micro} = P_t^{micro} - P_t^{mid}
    """

    def __init__(
        self,
        levels: int = 5,
        decay_lambda: float = 0.5,
        epsilon: float = 1e-8,
        bid_price_prefix: str = "bid_price_",
        bid_vol_prefix: str = "bid_vol_",
        ask_price_prefix: str = "ask_price_",
        ask_vol_prefix: str = "ask_vol_",
    ) -> None:
        """Initialize microstructure depth parameters.

        Args:
            levels: Number of book depth tiers (M).
            decay_lambda: Spatial decay rate lambda.
            epsilon: Regularization parameter to avoid division by zero.
            bid_price_prefix: Prefix for bid price columns (e.g. bid_price_1).
            bid_vol_prefix: Prefix for bid quantity columns (e.g. bid_vol_1).
            ask_price_prefix: Prefix for ask price columns (e.g. ask_price_1).
            ask_vol_prefix: Prefix for ask quantity columns (e.g. ask_vol_1).
        """
        self.levels = levels
        self.decay_lambda = decay_lambda
        self.epsilon = epsilon
        self.bid_price_prefix = bid_price_prefix
        self.bid_vol_prefix = bid_vol_prefix
        self.ask_price_prefix = ask_price_prefix
        self.ask_vol_prefix = ask_vol_prefix

        # Precompute spatial decay weights vector: w_i = exp(-lambda * i) for i in 0..M-1
        self.weights = np.exp(-self.decay_lambda * np.arange(self.levels, dtype=np.float64))

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> MicrostructureTopologyEngine:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def _has_depth_features(self, df: pd.DataFrame) -> bool:
        """Verify whether L2 Limit Order Book depth features exist in DataFrame."""
        required = [
            f"{self.bid_price_prefix}1",
            f"{self.bid_vol_prefix}1",
            f"{self.ask_price_prefix}1",
            f"{self.ask_vol_prefix}1",
        ]
        return all(col in df.columns for col in required)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform L2 depth snapshots into microstructural features.

        Args:
            X: Input DataFrame with order book queues or daily settlement bars.

        Returns:
            pd.DataFrame: DataFrame augmented with OBI, micro-price, and deviations.
        """
        df = X.copy()

        if not self._has_depth_features(df):
            logger.info("Microstructure: Depth columns not found. Skipping L2 queue metrics.")
            return df

        # Matrix tensors of shape (N, M)
        p_b = np.column_stack([df.get(f"{self.bid_price_prefix}{i}", 0.0) for i in range(1, self.levels + 1)])
        v_b = np.column_stack([df.get(f"{self.bid_vol_prefix}{i}", 0.0) for i in range(1, self.levels + 1)])
        p_a = np.column_stack([df.get(f"{self.ask_price_prefix}{i}", 0.0) for i in range(1, self.levels + 1)])
        v_a = np.column_stack([df.get(f"{self.ask_vol_prefix}{i}", 0.0) for i in range(1, self.levels + 1)])

        # Apply spatial exponential weights across depth levels
        w = self.weights  # Shape (M,)
        v_b_weighted = v_b * w
        v_a_weighted = v_a * w

        v_b_agg = np.sum(v_b_weighted, axis=1)
        v_a_agg = np.sum(v_a_weighted, axis=1)

        # 1. Spatially Decayed Order Imbalance (OBI)
        obi = (v_b_agg - v_a_agg) / (v_b_agg + v_a_agg + self.epsilon)

        # 2. Depth Micro-Price
        # Mid-price is based on Top-of-Book (Level 1)
        mid_price = 0.5 * (p_b[:, 0] + p_a[:, 0])

        sum_p_b_w = np.sum(p_b * v_b_weighted, axis=1)
        sum_p_a_w = np.sum(p_a * v_a_weighted, axis=1)

        term_bid = sum_p_b_w * (v_a_agg / (v_b_agg + self.epsilon))
        term_ask = sum_p_a_w * (v_b_agg / (v_a_agg + self.epsilon))
        micro_price = term_bid + term_ask

        # 3. Microstructure Directional Deviation
        delta_micro = micro_price - mid_price

        df["micro_obi"] = obi
        df["micro_price"] = micro_price
        df["micro_delta"] = delta_micro

        return df