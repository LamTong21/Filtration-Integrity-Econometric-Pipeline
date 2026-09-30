"""Unit Tests for Filtration Integrity and Temporal Leakage Prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.integrity.return_index import ForwardReturnSynthesizer
from src.features.shift_lock import GlobalShiftLock
from src.features.stateless.kinematics import KinematicDerivativeEngine


@pytest.fixture
def base_market_data() -> pd.DataFrame:
    """Generates synthetic baseline OHLCV series."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    data = {
        "open": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 105.0, 108.0],
        "high": [103.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 109.0, 107.0, 110.0],
        "low": [99.0, 101.0, 100.0, 102.0, 104.0, 103.0, 105.0, 106.0, 104.0, 107.0],
        "close": [102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 105.0, 108.0, 109.0],
        "volume": [1000, 1200, 1100, 1500, 1300, 1400, 1600, 1250, 1350, 1700],
        "Dividends": [0.0] * 10,
        "Stock Splits": [1.0] * 10,
    }
    return pd.DataFrame(data, index=dates)


def test_forward_return_invariance_to_future_corporate_actions(base_market_data: pd.DataFrame) -> None:
    """Verifies that an exogenous corporate event at T does not alter forward index values at t < T.

    Under Theorem 1, standard backward adjustments change past adjusted prices when
    a dividend occurs in the future. The forward return index must remain invariant.
    """
    synthesizer = ForwardReturnSynthesizer(base_index_value=1000.0)

    # 1. Transform baseline without events
    df_base = synthesizer.transform(base_market_data)
    past_p_adj_base = df_base["close_adj"].iloc[:7].copy()

    # 2. Inject an ex-dividend payout of $10.0 at day index 8
    df_event = base_market_data.copy()
    df_event.loc[df_event.index[8], "Dividends"] = 10.0

    df_transformed_event = synthesizer.transform(df_event)
    past_p_adj_event = df_transformed_event["close_adj"].iloc[:7]

    # Values prior to event date must be identically preserved: sigma(P_t^{adj}) subset F_t
    np.testing.assert_allclose(
        past_p_adj_base.to_numpy(),
        past_p_adj_event.to_numpy(),
        rtol=1e-10,
        err_msg="Filtration breach: Future corporate action changed historical forward return index.",
    )

    # The index must strictly reflect the dividend on and after the ex-date
    assert df_transformed_event["close_adj"].iloc[8] > df_base["close_adj"].iloc[8]


def test_global_shift_lock_enforces_f_t_minus_1_invariance(base_market_data: pd.DataFrame) -> None:
    """Ensures predictors at session t contain exclusively data finalized by t-1."""
    # Synthesize forward returns
    synthesizer = ForwardReturnSynthesizer()
    df_adj = synthesizer.transform(base_market_data)

    # Generate stateless kinematic velocity v_t
    kinematics = KinematicDerivativeEngine(price_col="close_adj")
    df_kin = kinematics.transform(df_adj)

    # Apply Universal Shift Lock
    shift_lock = GlobalShiftLock(lag=1)
    df_shifted = shift_lock.transform(df_kin)

    # At row index i (time t), the shifted momentum must equal row i-1 (time t-1) raw momentum
    for i in range(1, len(df_shifted)):
        raw_val_at_t_minus_1 = df_kin["log_return_momentum"].iloc[i - 1]
        locked_val_at_t = df_shifted["log_return_momentum"].iloc[i]

        assert np.isclose(raw_val_at_t_minus_1, locked_val_at_t, rtol=1e-8), (
            f"Shift-lock violation at index {i}: Feature contains contemporaneous information."
        )

    # The initial bar (index 0) must be NaN or zeroed out
    assert pd.isna(df_shifted["log_return_momentum"].iloc[0]) or df_shifted["log_return_momentum"].iloc[0] == 0.0


def test_perturbation_of_future_close_does_not_affect_past_features(base_market_data: pd.DataFrame) -> None:
    """Direct non-anticipative causality check: changing future prices cannot alter past features."""
    synthesizer = ForwardReturnSynthesizer()
    kinematics = KinematicDerivativeEngine(price_col="close_adj")
    shift_lock = GlobalShiftLock(lag=1)

    # Run base pipeline
    df_feat_base = shift_lock.transform(kinematics.transform(synthesizer.transform(base_market_data)))

    # Perturb close price at day 9 (last session)
    df_perturbed = base_market_data.copy()
    df_perturbed.loc[df_perturbed.index[9], "close"] = 999.0

    df_feat_perturbed = shift_lock.transform(kinematics.transform(synthesizer.transform(df_perturbed)))

    # Features across sessions 0..8 must remain completely identical
    pd.testing.assert_frame_equal(
        df_feat_base.iloc[:9],
        df_feat_perturbed.iloc[:9],
        check_dtype=False,
    )