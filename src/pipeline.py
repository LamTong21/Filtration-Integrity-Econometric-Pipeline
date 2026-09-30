"""Modular Non-Causal Econometric Pipeline Orchestrator."""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.integrity.audit import CandleGeometryAuditor
from src.integrity.return_index import ForwardReturnSynthesizer
from src.integrity.microstructure import MicrostructureTopologyEngine
from src.labeling.volatility import CausalVolatilityEstimator
from src.labeling.triple_barrier import DynamicTripleBarrierLabeler
from src.labeling.quarantine import QuarantineOperator
from src.features.shift_lock import GlobalShiftLock
from src.features.stateless.kinematics import KinematicDerivativeEngine
from src.features.stateless.shadow_geometry import ShadowGeometryEngine
from src.features.stateless.normalized_flow import NormalizedVolumeFlowEngine
from src.features.stateless.illiquidity import AmihudIlliquidityEngine
from src.features.stateful.fracdiff import OptimalFractionalDifferentiator
from src.features.stateful.gmm_regime import GMMRegimeFilter
from src.selection.mbcs_engine import MovingBlockConsensusSelector
from src.validation.purged_cv import PurgedTimeSeriesSplit
from src.validation.metrics import classification_edge_report

logger = logging.getLogger(__name__)


class EconometricPipeline(BaseEstimator):
    """Integrated Institutional Pipeline for Filtration-Preserved Financial ML.

    Enforces strict temporal ordering, target quarantine boundaries, and multi-block
    consensus feature elimination.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin | None = None,
        barrier_horizon: int = 5,
        kappa_pt: float = 1.0,
        kappa_sl: float = 1.0,
        volatility_span: int = 20,
        mbcs_blocks: int = 8,
        mbcs_consensus_threshold: float = 0.60,
    ) -> None:
        """Initialize the end-to-end econometric pipeline.

        Args:
            base_estimator: Out-of-sample classifier (default: L2-Regularized Logistic Regression C=0.1).
            barrier_horizon: Barrier evaluation horizon in settlement bars (h).
            kappa_pt: Take-profit barrier multiplier in local volatility units.
            kappa_sl: Stop-loss barrier multiplier in local volatility units.
            volatility_span: Exponential span for local volatility calculation.
            mbcs_blocks: Number of sliding temporal sub-blocks (B).
            mbcs_consensus_threshold: Consensus vote retention ratio (theta).
        """
        self.base_estimator = base_estimator or LogisticRegression(C=0.1, penalty="l2", solver="lbfgs")
        self.barrier_horizon = barrier_horizon
        self.kappa_pt = kappa_pt
        self.kappa_sl = kappa_sl
        self.volatility_span = volatility_span
        self.mbcs_blocks = mbcs_blocks
        self.mbcs_consensus_threshold = mbcs_consensus_threshold

        # Stage 1: Integrity
        self.auditor = CandleGeometryAuditor()
        self.return_synthesizer = ForwardReturnSynthesizer()
        self.micro_engine = MicrostructureTopologyEngine()

        # Stage 2: Labeling & Quarantine
        self.vol_estimator = CausalVolatilityEstimator(span=self.volatility_span)
        self.labeler = DynamicTripleBarrierLabeler(
            horizon=self.barrier_horizon,
            kappa_pt=self.kappa_pt,
            kappa_sl=self.kappa_sl,
        )
        self.quarantine = QuarantineOperator()

        # Stage 3: Feature Engineering
        self.shift_lock = GlobalShiftLock(lag=1)
        self.kinematics = KinematicDerivativeEngine()
        self.shadow_geom = ShadowGeometryEngine()
        self.flow_engine = NormalizedVolumeFlowEngine()
        self.amihud_engine = AmihudIlliquidityEngine()
        self.fracdiff = OptimalFractionalDifferentiator()
        self.gmm_regime = GMMRegimeFilter()

        # Stage 4: MBCS
        self.mbcs_selector = MovingBlockConsensusSelector(
            n_blocks=self.mbcs_blocks,
            consensus_threshold=self.mbcs_consensus_threshold,
        )

    def prepare_dataset(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepares raw inputs into sanitized predictors, target labels, and barrier metadata.

        Args:
            raw_df: Raw OHLCV DataFrame with corporate action fields.

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
              - X_predictors: Sanitized, point-in-time candidate feature matrix.
              - y_labels: Binary target label series Y in {0, 1}.
              - metadata: Barrier metadata including exit return and timestamps.
        """
        # 1. Audit candle geometry and enforce bounds
        df_clean = self.auditor.transform(raw_df)

        # 2. Forward total return synthesis (eliminates retrospective lookahead)
        df_fwd = self.return_synthesizer.transform(df_clean)

        # 3. Microstructure dynamics
        df_micro = self.micro_engine.transform(df_fwd)

        # 4. Causal volatility
        df_vol = self.vol_estimator.transform(df_micro)

        # 5. Labeling
        meta_df = self.labeler.transform(df_vol)
        target_series = meta_df["target"]

        # 6. Stateless feature engineering
        df_feat = self.kinematics.transform(df_vol)
        df_feat = self.shadow_geom.transform(df_feat)
        df_feat = self.flow_engine.transform(df_feat)
        df_feat = self.amihud_engine.transform(df_feat)

        # 7. Apply Point-in-Time Shift-1 Lock on stateless features
        df_shifted = self.shift_lock.transform(df_feat)

        # 8. Enforce quarantine boundary
        X_sanitized = self.quarantine.transform(df_shifted)

        # Align valid indices
        valid_idx = X_sanitized.dropna().index.intersection(target_series.dropna().index)
        return X_sanitized.loc[valid_idx], target_series.loc[valid_idx], meta_df.loc[valid_idx]

    def run_purged_cross_validation(
        self,
        raw_df: pd.DataFrame,
        n_splits: int = 5,
        max_train_sessions: int = 750,
        purge_gap: int = 5,
    ) -> dict[str, Any]:
        """Runs the complete purged CV protocol with stateful feature freezing and MBCS.

        Args:
            raw_df: Raw historical input DataFrame.
            n_splits: Number of cross-validation folds.
            max_train_sessions: Historical training length per fold.
            purge_gap: Gap between train and test partitions (>= barrier horizon).

        Returns:
            dict[str, Any]: Fold summaries, predictions, selected features, and aggregated metrics.
        """
        X_all, y_all, meta_all = self.prepare_dataset(raw_df)

        cv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            max_train_sessions=max_train_sessions,
            purge_gap=purge_gap,
            buffer_sessions=60,
        )

        fold_reports = []
        oof_predictions = []
        oof_targets = []
        fold_feature_persistence = {}

        for fold_idx, (train_idx, test_idx, _) in enumerate(cv.split(X_all), start=1):
            X_tr, y_tr = X_all.iloc[train_idx].copy(), y_all.iloc[train_idx].copy()
            X_te, y_te = X_all.iloc[test_idx].copy(), y_all.iloc[test_idx].copy()

            # --- Stateful Stage: Fit on Train Fold ONLY & Freeze ---
            self.fracdiff.fit(X_tr)
            self.gmm_regime.fit(X_tr)

            X_tr_stateful = self.gmm_regime.transform(self.fracdiff.transform(X_tr))
            X_te_stateful = self.gmm_regime.transform(self.fracdiff.transform(X_te))

            # --- Stage 4: Moving-Block Consensus Selection ---
            self.mbcs_selector.fit(X_tr_stateful, y_tr)
            X_tr_star = self.mbcs_selector.transform(X_tr_stateful)
            X_te_star = self.mbcs_selector.transform(X_te_stateful)

            # Record surviving features
            fold_feature_persistence[f"Fold_{fold_idx}"] = self.mbcs_selector.selected_features_

            # Fit base classifier
            clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs")
            clf.fit(X_tr_star, y_tr)

            # Out-of-sample inference
            p_te = clf.predict_proba(X_te_star)[:, 1]

            fold_metric = classification_edge_report(y_te, p_te)
            fold_reports.append(fold_metric)

            oof_predictions.extend(p_te)
            oof_targets.extend(y_te.to_numpy())

            logger.info("Fold %d Evaluated: ROC-AUC = %.4f", fold_idx, fold_metric["ROC-AUC"])

        # Aggregate Out-of-Sample Performance
        aggregate_metrics = classification_edge_report(np.array(oof_targets), np.array(oof_predictions))

        return {
            "Aggregate Metrics": aggregate_metrics,
            "Fold Metrics": fold_reports,
            "Feature Persistence": fold_feature_persistence,
            "OOF Targets": np.array(oof_targets),
            "OOF Predictions": np.array(oof_predictions),
        }