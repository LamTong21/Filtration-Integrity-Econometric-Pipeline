"""Validation, Cross-Validation Schemes, and Econometric Diagnostics.

Implements Purged Time-Series Cross-Validation with historical rolling lookback buffers
and financial machine learning evaluation metrics (Brier Score, Long Precision, Bet Sizing).
"""

from src.validation.purged_cv import PurgedTimeSeriesSplit
from src.validation.metrics import (
    classification_edge_report,
    compute_affine_bet_sizes,
    evaluate_strategy_performance,
)

__all__ = [
    "PurgedTimeSeriesSplit",
    "classification_edge_report",
    "compute_affine_bet_sizes",
    "evaluate_strategy_performance",
]