"""Quarantine Projection Operator Q Module.

Establishes a measure-theoretic boundary that purges all target-generation artifacts,
exogenous corporate events, and dynamic parameters from the candidate predictor space.
"""

from __future__ import annotations

import logging
from typing import Sequence
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class QuarantineOperator(BaseEstimator, TransformerMixin):
    """Enforces the Quarantine Projection Operator Q: D_t -> X_t.

    Formulates an explicit topological boundary:
        Q(D_t) = D_t \\ M_t^{quarantine}

    where:
        M_t^{quarantine} = {Y_t, R_t^{barrier}, tau_t^{touch}, sigma_t, S_t, D_t}

    Prevents downstream feature engineering, scaling, and feature selection engines
    from accessing future event-touch times or target parameters.
    """

    DEFAULT_QUARANTINE_COLS = (
        "target",
        "barrier_return",
        "touch_timestamp",
        "barrier_type",
        "local_volatility",
        "Dividends",
        "Stock Splits",
    )

    def __init__(
        self,
        quarantine_cols: Sequence[str] | None = None,
        additional_cols_to_purge: Sequence[str] | None = None,
    ) -> None:
        """Initialize the quarantine boundary.

        Args:
            quarantine_cols: Exact list of metadata and corporate action columns to quarantine.
            additional_cols_to_purge: Any supplementary user-specified leak vectors.
        """
        if quarantine_cols is None:
            self.quarantine_cols = set(self.DEFAULT_QUARANTINE_COLS)
        else:
            self.quarantine_cols = set(quarantine_cols)

        if additional_cols_to_purge:
            self.quarantine_cols.update(additional_cols_to_purge)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> QuarantineOperator:
        """Fit stub to adhere to Scikit-Learn transformer API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Quarantine target metadata from the feature candidate matrix.

        Args:
            X: Input DataFrame containing raw, derived, and label metadata columns.

        Returns:
            pd.DataFrame: Sanitized predictor matrix X_t free of target contamination.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        cols_present_to_purge = [col for col in self.quarantine_cols if col in X.columns]
        if cols_present_to_purge:
            logger.info("QuarantineOperator: Isolating and removing %d columns: %s", len(cols_present_to_purge), cols_present_to_purge)
            sanitized_df = X.drop(columns=cols_present_to_purge)
        else:
            sanitized_df = X.copy()

        return sanitized_df

    def split_features_and_target(
        self, X: pd.DataFrame, target_col: str = "target"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Convenience method to quarantine predictors while extracting the primary target y.

        Args:
            X: Input DataFrame containing both predictors and label columns.
            target_col: Name of the binary target label column.

        Returns:
            tuple[pd.DataFrame, pd.Series]: (Sanitized X_t, Target vector y_t)
        """
        if target_col not in X.columns:
            raise KeyError(f"Target column '{target_col}' not present in provided DataFrame.")

        y = X[target_col].copy()
        X_sanitized = self.transform(X)
        return X_sanitized, y