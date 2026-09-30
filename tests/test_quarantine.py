"""Unit Tests for Quarantine Projection Operator Q."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labeling.quarantine import QuarantineOperator
from src.labeling.triple_barrier import DynamicTripleBarrierLabeler
from src.labeling.volatility import CausalVolatilityEstimator


@pytest.fixture
def labeled_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic price data with full triple barrier and volatility artifacts."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(42)
    close_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, size=30)))

    df = pd.DataFrame(
        {
            "close_adj": close_prices,
            "open_adj": close_prices * 0.998,
            "high_adj": close_prices * 1.01,
            "low_adj": close_prices * 0.99,
            "volume": np.random.randint(1000, 5000, size=30),
            "Dividends": [0.0] * 30,
            "Stock Splits": [1.0] * 30,
        },
        index=dates,
    )

    # Estimate volatility and generate barrier labels
    vol_est = CausalVolatilityEstimator(span=10)
    df_vol = vol_est.transform(df)

    labeler = DynamicTripleBarrierLabeler(horizon=5, kappa_pt=1.0, kappa_sl=1.0)
    df_labels = labeler.transform(df_vol)

    # Combine into full candidate tensor D_t
    df_combined = pd.concat([df_vol, df_labels], axis=1)
    return df_combined, df_labels


def test_quarantine_projection_operator_removes_all_target_metadata(labeled_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies that Q(D_t) = D_t \\ M_t^{quarantine} completely isolates target artifacts."""
    df_combined, _ = labeled_dataset
    quarantine = QuarantineOperator()

    sanitized_X = quarantine.transform(df_combined)

    # Target metadata defined in Definition 3 of the paper
    purged_elements = {
        "target",
        "barrier_return",
        "touch_timestamp",
        "barrier_type",
        "local_volatility",
        "Dividends",
        "Stock Splits",
    }

    # Verify none of the quarantined items exist in the sanitized predictor space
    intersecting_columns = purged_elements.intersection(sanitized_X.columns)
    assert len(intersecting_columns) == 0, (
        f"Quarantine boundary breached! Quarantined columns found in predictors: {intersecting_columns}"
    )

    # Legitimate non-target predictors must be preserved
    assert "close_adj" in sanitized_X.columns
    assert "volume" in sanitized_X.columns
    assert len(sanitized_X) == len(df_combined)


def test_quarantine_custom_leak_purging(labeled_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Tests the addition of arbitrary user-defined leak artifacts to the boundary."""
    df_combined, _ = labeled_dataset
    df_combined["leaked_future_signal"] = 123.45

    quarantine = QuarantineOperator(additional_cols_to_purge=["leaked_future_signal"])
    sanitized_X = quarantine.transform(df_combined)

    assert "leaked_future_signal" not in sanitized_X.columns


def test_split_features_and_target_alignment(labeled_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Ensures helper split separates sanitized X and label y with identical indexing."""
    df_combined, _ = labeled_dataset
    quarantine = QuarantineOperator()

    X, y = quarantine.split_features_and_target(df_combined, target_col="target")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "target" not in X.columns
    assert y.name == "target"
    pd.testing.assert_index_equal(X.index, y.index)