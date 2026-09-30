"""Universal Lag Operator (Point-in-Time Shift Lock)."""

from __future__ import annotations

import logging
from typing import Sequence
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class GlobalShiftLock(BaseEstimator, TransformerMixin):
    """Enforces the universal lag operator L: X_t -> X_{t-1}.

    Ensures contemporaneous settlement prices cannot contaminate features predicting
    barrier events starting at t and spanning horizons (t, t + h].
    """

    def __init__(
        self,
        lag: int = 1,
        columns_to_shift: Sequence[str] | None = None,
        exclude_columns: Sequence[str] | None = None,
    ) -> None:
        """Initialize Shift Lock.

        Args:
            lag: Discrete time shift backwards (default is 1 session).
            columns_to_shift: Specific column subset to shift. If None, shifts all numeric columns.
            exclude_columns: Columns explicitly exempted from shift (e.g., target labels).
        """
        self.lag = lag
        self.columns_to_shift = columns_to_shift
        self.exclude_columns = set(exclude_columns or [])

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GlobalShiftLock:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply lag operator across predictor space.

        Args:
            X: Input candidate predictor DataFrame.

        Returns:
            pd.DataFrame: Shifted predictor DataFrame.
        """
        df = X.copy()
        target_cols = [
            c for c in (self.columns_to_shift if self.columns_to_shift is not None else df.columns)
            if c not in self.exclude_columns
        ]

        logger.debug("Applying Global Shift-%d Lock across %d features.", self.lag, len(target_cols))
        df[target_cols] = df[target_cols].shift(self.lag)
        return df