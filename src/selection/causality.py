"""Dual Linear and Non-Linear Causal Lead-Lag Screening Module."""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


class DualCausalScreen(BaseEstimator):
    """Screens features for forward predictive causality using Granger and Mutual Information.

    For candidate X_j and target y over lags k in {1, ..., K_lag}:
      - Step 1: Granger Causality F-test: If p-value < 0.05 -> accept with lag k.
      - Step 2: Fallback Transfer Mutual Information: If MI(Lag(X_j, k), y) > 0.01 -> accept with lag k.
    """

    def __init__(
        self,
        max_lag: int = 5,
        granger_p_threshold: float = 0.05,
        mi_threshold: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """Initialize causal lead-lag screen.

        Args:
            max_lag: Maximum lead-lag shift horizon (K_lag, default is 5 sessions).
            granger_p_threshold: Alpha threshold for rejecting Granger non-causality.
            mi_threshold: Minimum mutual information dependency threshold.
            random_state: Random state for deterministic permutation sampling in MI.
        """
        self.max_lag = max_lag
        self.granger_p_threshold = granger_p_threshold
        self.mi_threshold = mi_threshold
        self.random_state = random_state

    def evaluate_feature(
        self, x_series: pd.Series, y_series: pd.Series
    ) -> tuple[bool, int]:
        """Test causality for a single feature against the binary target.

        Args:
            x_series: Candidate predictor series.
            y_series: Target label series.

        Returns:
            tuple[bool, int]: (is_causal, optimal_lag_k)
        """
        # Align series and discard invalid rows
        data = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
        if len(data) < (self.max_lag * 5) or data["y"].nunique() < 2:
            return False, 1

        y_clean = data["y"].to_numpy()
        x_clean = data["x"].to_numpy()

        # Step 1: Linear Granger Causality Screen across lags 1..K_lag
        for k in range(1, self.max_lag + 1):
            # grangercausalitytests expects 2D array [target, predictor]
            test_mat = np.column_stack([y_clean, x_clean])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc_res = grangercausalitytests(test_mat, maxlag=[k], verbose=False)
                    f_pval = gc_res[k][0]["ssr_ftest"][1]

                if f_pval < self.granger_p_threshold:
                    return True, k
            except Exception:
                continue

        # Step 2: Non-linear Lagged Transfer Mutual Information Screen
        for k in range(1, self.max_lag + 1):
            lagged_x = pd.Series(x_clean).shift(k).dropna()
            y_aligned = pd.Series(y_clean).iloc[k:]

            if len(lagged_x) < 20 or y_aligned.nunique() < 2:
                continue

            try:
                mi = mutual_info_classif(
                    lagged_x.to_numpy().reshape(-1, 1),
                    y_aligned.to_numpy(),
                    discrete_features=False,
                    random_state=self.random_state,
                )[0]

                if mi > self.mi_threshold:
                    return True, k
            except Exception:
                continue

        return False, 1