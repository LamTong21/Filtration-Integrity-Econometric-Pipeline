"""Purged Time-Series Cross-Validation with Lookback Buffer Module."""

from __future__ import annotations

import logging
from typing import Generator
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit:
    """Purged cross-validation generator for serial, overlapping financial time series.

    Guarantees two structural temporal boundaries:
      1. Purge Gap: A quarantine buffer between train and test windows (purge >= label_horizon)
         preventing labels from spanning fold boundaries.
      2. Historical Lookback Buffer: A pre-test segment prepended strictly to warm up rolling
         feature operators without NaN values, evaluated strictly on the out-of-sample window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_sessions: int = 750,
        test_sessions: int | None = None,
        purge_gap: int = 5,
        buffer_sessions: int = 60,
    ) -> None:
        """Initialize purged cross-validation scheme.

        Args:
            n_splits: Number of sequential outer folds.
            max_train_sessions: Rolling/expanding maximum historical training window size.
            test_sessions: Number of evaluation sessions per fold. If None, computed evenly.
            purge_gap: Gap between end of train and start of test (strictly >= label horizon h).
            buffer_sessions: History prepended to test fold for stateless rolling operators.
        """
        self.n_splits = n_splits
        self.max_train_sessions = max_train_sessions
        self.test_sessions = test_sessions
        self.purge_gap = purge_gap
        self.buffer_sessions = buffer_sessions

    def get_n_splits(self) -> int:
        """Return number of splitting iterations."""
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generate indices for purged time-series splits.

        Yields:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
              - train_indices: Historical training bar indices.
              - test_indices: Strictly out-of-sample evaluation indices.
              - buffer_indices: Preceding lookback indices required for stateless feature warm-up.
        """
        n_samples = len(X)
        if self.test_sessions is None:
            # Reserve room for train + purge + splits
            available_test = n_samples - (self.max_train_sessions + self.purge_gap)
            if available_test <= 0:
                raise ValueError(
                    f"Dataset size ({n_samples}) insufficient for train ({self.max_train_sessions}) "
                    f"and purge gap ({self.purge_gap})."
                )
            test_size = available_test // self.n_splits
        else:
            test_size = self.test_sessions

        logger.info(
            "PurgedTimeSeriesSplit: %d splits, train_max=%d, test_size=%d, purge_gap=%d, buffer=%d",
            self.n_splits,
            self.max_train_sessions,
            test_size,
            self.purge_gap,
            self.buffer_sessions,
        )

        for fold in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - fold) * test_size
            test_start = test_end - test_size

            train_end = test_start - self.purge_gap
            train_start = max(0, train_end - self.max_train_sessions)

            if train_end <= train_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            # Historical buffer prepended strictly for feature calculation warm-up
            buffer_start = max(0, test_start - self.buffer_sessions)
            buffer_indices = np.arange(buffer_start, test_start)

            yield train_indices, test_indices, buffer_indices