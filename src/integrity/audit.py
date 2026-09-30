"""Candle Geometry and Data Hygiene Audit Module."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CandleGeometryAuditor(BaseEstimator, TransformerMixin):
    """Audits and sanitizes raw financial OHLCV time-series data.

    Enforces physical market geometry invariants:
    1. Monotonic strictly increasing temporal indexing (deduplication preserving latest tick).
    2. Envelope bounds consistency: High >= max(Open, Close) and Low <= min(Open, Close).
    3. Positivity constraints: Low > 0 and Volume >= 0.
    4. Fill/repair zero or negative volume anomalies if required.
    """

    def __init__(
        self,
        time_col: str | None = None,
        price_cols: Sequence[str] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
        drop_duplicates: bool = True,
        enforce_strictly_positive: bool = True,
    ) -> None:
        """Initialize the auditor with schema settings.

        Args:
            time_col: Name of datetime column if not already set as Index.
            price_cols: Sequence containing (Open, High, Low, Close) column names.
            volume_col: Name of trading volume column.
            drop_duplicates: Whether to drop duplicate timestamps.
            enforce_strictly_positive: If True, drops records with Low <= 0.
        """
        self.time_col = time_col
        self.price_cols = list(price_cols)
        self.volume_col = volume_col
        self.drop_duplicates = drop_duplicates
        self.enforce_strictly_positive = enforce_strictly_positive

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CandleGeometryAuditor:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Audit and enforce candle geometries.

        Args:
            X: Raw tabular DataFrame with financial bars.

        Returns:
            pd.DataFrame: Sanitized time series with guaranteed physical bounds.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        df = X.copy()

        # 1. Normalize temporal index
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df = df.set_index(self.time_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise ValueError("DataFrame index must be convertible to DatetimeIndex.") from exc

        # 2. Deduplicate temporal points
        if self.drop_duplicates:
            init_len = len(df)
            df = df[~df.index.duplicated(keep="last")]
            dropped_dups = init_len - len(df)
            if dropped_dups > 0:
                logger.info("Auditor: Dropped %d duplicate timestamps.", dropped_dups)

        df = df.sort_index()

        open_col, high_col, low_col, close_col = self.price_cols

        # Check column presence
        missing_cols = [c for c in (*self.price_cols, self.volume_col) if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Required columns missing in DataFrame: {missing_cols}")

        # 3. Geometric envelope enforcement
        # High must be at least max(Open, Close); Low must be at most min(Open, Close)
        max_oc = np.maximum(df[open_col].to_numpy(), df[close_col].to_numpy())
        min_oc = np.minimum(df[open_col].to_numpy(), df[close_col].to_numpy())

        df[high_col] = np.maximum(df[high_col].to_numpy(), max_oc)
        df[low_col] = np.minimum(df[low_col].to_numpy(), min_oc)

        # 4. Physical boundaries check
        valid_mask = np.ones(len(df), dtype=bool)
        if self.enforce_strictly_positive:
            valid_mask &= (df[low_col] > 0.0) & (df[open_col] > 0.0) & (df[close_col] > 0.0)

        valid_mask &= df[self.volume_col] >= 0.0

        num_invalid = (~valid_mask).sum()
        if num_invalid > 0:
            logger.warning("Auditor: Dropped %d rows failing positive price/volume conditions.", num_invalid)
            df = df[valid_mask]

        return df