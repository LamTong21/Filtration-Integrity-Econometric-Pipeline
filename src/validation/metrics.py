"""Financial Machine Learning Diagnostics and Continuous Position Sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_affine_bet_sizes(
    probabilities: np.ndarray | pd.Series,
    threshold: float = 0.50,
    min_size: float = 0.20,
    max_size: float = 1.00,
) -> np.ndarray:
    """Computes continuous asymmetric bet sizes from calibrated probabilities.

    Equation (Section 7.2):
        Bet Size_t = max(0, clip((P_hat(Y_t=1) - tau) / (1.0 - tau), min_size, max_size))

    Args:
        probabilities: Predicted positive class probabilities P(Y = 1).
        threshold: Neutral classification cutoff tau (default is 0.50).
        min_size: Lower bound threshold floor for entered positions.
        max_size: Maximum capital fraction allocation.

    Returns:
        np.ndarray: Continuous position sizes bounded in [0.0, max_size].
    """
    p = np.asarray(probabilities, dtype=np.float64)
    scaled = (p - threshold) / (1.0 - threshold)
    clipped = np.clip(scaled, min_size, max_size)
    bet_sizes = np.where(p > threshold, clipped, 0.0)
    return bet_sizes


def classification_edge_report(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Generates authentic financial machine learning metrics.

    Calculates Out-of-Sample Accuracy, ROC-AUC, Brier score loss, and directional
    long-side precision vs. base rate.

    Args:
        y_true: Ground truth binary labels Y in {0, 1}.
        y_pred_proba: Calibrated forecast probabilities P_hat(Y = 1).
        threshold: Classification cutoff.

    Returns:
        dict[str, float]: Dictionary of validated classification metrics.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    p_pred = np.asarray(y_pred_proba, dtype=np.float64)
    y_pred = (p_pred >= threshold).astype(np.int32)

    base_rate = float(np.mean(y_t))
    accuracy = float(accuracy_score(y_t, y_pred))
    auc = float(roc_auc_score(y_t, p_pred)) if len(np.unique(y_t)) > 1 else 0.50
    brier = float(brier_score_loss(y_t, p_pred))

    # Long signal evaluations
    long_precision = float(precision_score(y_t, y_pred, zero_division=0))
    long_recall = float(recall_score(y_t, y_pred, zero_division=0))
    signal_rate = float(np.mean(y_pred))

    return {
        "Base Rate (Class 1)": base_rate,
        "Signal Rate (Long %)": signal_rate,
        "Accuracy": accuracy,
        "ROC-AUC": auc,
        "Brier Loss": brier,
        "Long Precision": long_precision,
        "Long Recall": long_recall,
        "Edge Over Base": long_precision - base_rate,
    }


def evaluate_strategy_performance(
    y_true: pd.Series,
    y_proba: pd.Series,
    barrier_returns: pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Combines signal sizing with realized barrier returns for performance accounting."""
    bet_sizes = compute_affine_bet_sizes(y_proba, threshold=threshold)
    strategy_returns = bet_sizes * barrier_returns.to_numpy()

    df_res = pd.DataFrame(
        {
            "Actual": y_true,
            "Pred_Proba": y_proba,
            "Bet_Size": bet_sizes,
            "Barrier_Return": barrier_returns,
            "Strategy_Return": strategy_returns,
        },
        index=y_true.index,
    )
    return df_res