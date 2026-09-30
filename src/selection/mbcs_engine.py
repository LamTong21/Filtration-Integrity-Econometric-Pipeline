    """Moving-Block Consensus Selection (MBCS) Orchestrator Module."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.selection.fast_vif import FastVIFPruner
from src.selection.hrp_medoids import HRPMedoidExtractor
from src.selection.causality import DualCausalScreen

logger = logging.getLogger(__name__)


class MovingBlockConsensusSelector(BaseEstimator, TransformerMixin):
    """Full Algorithm 1: Moving-Block Consensus Selection (MBCS).

    Evaluates feature stability across overlapping temporal blocks to eliminate
    local regimes and overfitted artifacts.
    """

    def __init__(
        self,
        n_blocks: int = 8,
        block_ratio: float = 0.75,
        consensus_threshold: float = 0.60,
        max_clusters: int = 14,
        vif_threshold: float = 5.0,
        max_lag: int = 5,
        min_final_features: int = 6,
        max_final_features: int = 16,
        random_state: int = 42,
    ) -> None:
        """Initialize MBCS Orchestrator.

        Args:
            n_blocks: Number of sub-blocks B (default is 8).
            block_ratio: Length ratio rho for each sub-block relative to training partition.
            consensus_threshold: Agreement ratio theta for voting acceptance (default is 0.60).
            max_clusters: Maximum clusters C_max for HRP medoid extraction.
            vif_threshold: Multicollinearity threshold for pseudo-inverse VIF.
            max_lag: Maximum causal lead-lag search horizon (K_lag).
            min_final_features: Minimum boundary constraint on feature cardinality.
            max_final_features: Maximum boundary constraint on feature cardinality.
            random_state: Reproducibility seed.
        """
        self.n_blocks = n_blocks
        self.block_ratio = block_ratio
        self.consensus_threshold = consensus_threshold
        self.max_clusters = max_clusters
        self.vif_threshold = vif_threshold
        self.max_lag = max_lag
        self.min_final_features = min_final_features
        self.max_final_features = max_final_features
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.lag_map_: dict[str, int] = {}
        self.vote_counts_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MovingBlockConsensusSelector:
        """Execute multi-block consensus selection protocol on training fold.

        Args:
            X: Training candidate predictor matrix (N, P).
            y: Binary target label series (N,).

        Returns:
            MovingBlockConsensusSelector: Fitted selector with F_star and Lambda_star.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)

        # Align clean non-null instances
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        x_train = X.loc[valid_idx]
        y_train = y.loc[valid_idx]

        n, p = x_train.shape
        l_block = max(int(math.floor(n * self.block_ratio)), 40)

        # Generate B equidistant sub-block start indices across [0, N - L_block]
        if n - l_block <= 0:
            start_indices = [0] * self.n_blocks
        else:
            start_indices = np.linspace(0, n - l_block, num=self.n_blocks, dtype=int)

        vote_counter: dict[str, int] = defaultdict(int)
        lag_registry: dict[str, list[int]] = defaultdict(list)

        vif_pruner = FastVIFPruner(vif_threshold=self.vif_threshold, min_features=self.min_final_features)
        hrp_extractor = HRPMedoidExtractor(max_clusters=self.max_clusters)
        causal_screener = DualCausalScreen(max_lag=self.max_lag, random_state=self.random_state)

        logger.info("Executing MBCS across %d blocks with length %d bars.", len(start_indices), l_block)

        for b_idx, s_b in enumerate(start_indices, start=1):
            x_b = x_train.iloc[s_b : s_b + l_block]
            y_b = y_train.iloc[s_b : s_b + l_block]

            # Require at least two distinct target classes within the sub-block
            if y_b.nunique() < 2:
                logger.debug("Block %d skipped: single target class present.", b_idx)
                continue

            # Step A: Fast Linear Orthogonalization (Inverse Correlation VIF)
            x_b_vif = vif_pruner.fit_transform(x_b)

            # Step B: Non-Linear Hierarchical Medoid Pruning
            x_b_medoids = hrp_extractor.fit_transform(x_b_vif)

            # Step C: Dual Causal Lead-Lag Screen
            s_b_features = []
            for col in x_b_medoids.columns:
                is_causal, best_lag = causal_screener.evaluate_feature(x_b_medoids[col], y_b)
                if is_causal:
                    s_b_features.append(col)
                    lag_registry[col].append(best_lag)

            for feat in s_b_features:
                vote_counter[feat] += 1

        self.vote_counts_ = dict(vote_counter)

        # Step D: Consensus Aggregation
        v_threshold = math.ceil(self.n_blocks * self.consensus_threshold)
        f_star = [f for f, count in vote_counter.items() if count >= v_threshold]

        # Enforce cardinality constraints: min_final_features <= |F_star| <= max_final_features
        sorted_by_votes = sorted(vote_counter.keys(), key=lambda f: vote_counter[f], reverse=True)

        if len(f_star) < self.min_final_features:
            f_star = sorted_by_votes[: min(self.min_final_features, len(sorted_by_votes))]
        elif len(f_star) > self.max_final_features:
            f_star = sorted_by_votes[: self.max_final_features]

        # Fallback if no features survived tests
        if not f_star:
            f_star = list(x_train.columns[: self.min_final_features])

        # Compute optimal median lag map Lambda_star
        lambda_star = {}
        for f in f_star:
            lags = lag_registry.get(f, [1])
            lambda_star[f] = int(np.median(lags)) if lags else 1

        self.selected_features_ = f_star
        self.lag_map_ = lambda_star

        logger.info("MBCS Completed: Selected %d features: %s", len(self.selected_features_), self.selected_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature filtering and optimal lead-lag shifts.

        Args:
            X: Input DataFrame to prune and shift.

        Returns:
            pd.DataFrame: Matrix X_star with optimal temporal shifts.
        """
        if not self.selected_features_:
            return X

        df_out = pd.DataFrame(index=X.index)
        for feat in self.selected_features_:
            if feat not in X.columns:
                continue
            lag_k = self.lag_map_.get(feat, 1)
            # Shift by identified optimal lag
            df_out[f"{feat}_lag{lag_k}"] = X[feat].shift(lag_k)

        return df_out.fillna(0.0)