"""Unit Tests for Moving-Block Consensus Selection (Algorithm 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.selection.mbcs_engine import MovingBlockConsensusSelector
from src.selection.fast_vif import FastVIFPruner
from src.selection.causality import DualCausalScreen


@pytest.fixture
def synthetic_mbcs_environment() -> tuple[pd.DataFrame, pd.Series]:
    """Generates an econometric environment with causal, collinear, and random noise features."""
    np.random.seed(42)
    n_samples = 250
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="B")

    # Latent predictive state
    latent_signal = np.sin(np.linspace(0, 10, n_samples))

    # Binary target driven by lagged latent signal
    prob = 1.0 / (1.0 + np.exp(-2.0 * pd.Series(latent_signal).shift(2).bfill().to_numpy()))
    y = pd.Series((prob > 0.50).astype(int), index=dates, name="target")

    data = {}

    # 1. Genuine causal lead features (correlated with latent signal)
    data["causal_feat_1"] = latent_signal + np.random.normal(0, 0.1, n_samples)
    data["causal_feat_2"] = np.roll(latent_signal, 1) + np.random.normal(0, 0.1, n_samples)
    data["causal_feat_3"] = np.roll(latent_signal, 2) + np.random.normal(0, 0.1, n_samples)

    # 2. Highly collinear clones (VIF > 5.0)
    data["collinear_clone_1"] = data["causal_feat_1"] * 1.0001 + np.random.normal(0, 1e-5, n_samples)
    data["collinear_clone_2"] = data["causal_feat_1"] * 0.9999 + np.random.normal(0, 1e-5, n_samples)

    # 3. Pure uninformative Gaussian noise features
    for i in range(15):
        data[f"random_noise_{i}"] = np.random.normal(0, 1, n_samples)

    X = pd.DataFrame(data, index=dates)
    return X, y


def test_fast_vif_drops_perfectly_collinear_columns(synthetic_mbcs_environment: tuple[pd.DataFrame, pd.Series]) -> None:
    """Verifies that pseudo-inverse VIF eliminates collinear clones."""
    X, _ = synthetic_mbcs_environment
    vif_pruner = FastVIFPruner(vif_threshold=5.0, min_features=2)

    X_filtered = vif_pruner.fit_transform(X)

    # The filter must discard redundant clones of causal_feat_1
    retained_cols = set(X_filtered.columns)
    assert not (
        "causal_feat_1" in retained_cols
        and "collinear_clone_1" in retained_cols
        and "collinear_clone_2" in retained_cols
    ), "Fast VIF failed to prune near-identical collinear variables."


def test_dual_causal_screen_rejects_pure_noise() -> None:
    """Verifies that independent random noise fails Granger and Mutual Information tests."""
    np.random.seed(42)
    n = 200
    y = pd.Series(np.random.choice([0, 1], size=n))
    noise_x = pd.Series(np.random.normal(0, 1, size=n))

    screener = DualCausalScreen(max_lag=3, granger_p_threshold=0.05, mi_threshold=0.01)
    is_causal, _ = screener.evaluate_feature(noise_x, y)

    assert not is_causal, "DualCausalScreen falsely classified white noise as causal signal."


def test_mbcs_cardinality_constraint_bounds(synthetic_mbcs_environment: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests that Algorithm 1 strictly satisfies the bounding constraint 6 <= |F*| <= 16."""
    X, y = synthetic_mbcs_environment

    mbcs = MovingBlockConsensusSelector(
        n_blocks=8,
        block_ratio=0.75,
        consensus_threshold=0.60,
        min_final_features=6,
        max_final_features=16,
    )

    mbcs.fit(X, y)
    selected_features = mbcs.selected_features_
    num_selected = len(selected_features)

    # Check cardinality invariant: 6 <= |F*| <= 16
    assert 6 <= num_selected <= 16, (
        f"MBCS Constraint breach: Expected 6 <= |F*| <= 16, but got {num_selected} features: {selected_features}"
    )

    # Output transformation matrix must contain valid shifted column names
    X_star = mbcs.transform(X)
    assert X_star.shape[1] == num_selected
    for feat in selected_features:
        assert any(feat in col for col in X_star.columns)