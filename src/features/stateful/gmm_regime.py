"""Gaussian Mixture Model (GMM) Latent Regime Classification."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class GMMRegimeFilter(BaseEstimator, TransformerMixin):
    """Identifies latent market regimes using an unsupervised 2-component GMM."""

    def __init__(
        self,
        n_components: int = 2,
        price_col: str = "close_adj",
        high_col: str = "high_adj",
        low_col: str = "low_adj",
        random_state: int = 42,
    ) -> None:
        """Initialize GMM regime filter.

        Args:
            n_components: Number of mixture clusters (default is 2: Bull vs Bear/Stagnant).
            price_col: Adjusted price column for returns.
            high_col: Adjusted high column for Parkinson volatility.
            low_col: Adjusted low column for Parkinson volatility.
            random_state: Reproducibility seed.
        """
        self.n_components = n_components
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.random_state = random_state

        self.gmm_: GaussianMixture | None = None
        self.bull_regime_index_: int = 0

    def _extract_regime_tensor(self, df: pd.DataFrame) -> np.ndarray:
        """Compute observation vector z_t = [r_t, sigma_t^{Parkinson}]."""
        c = df[self.price_col].to_numpy(dtype=np.float64)
        h = df[self.high_col].to_numpy(dtype=np.float64)
        l = df[self.low_col].to_numpy(dtype=np.float64)

        # Log return r_t
        log_ret = np.zeros(len(df), dtype=np.float64)
        log_ret[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-8))

        # Parkinson extreme value volatility proxy: ln(H/L)^2 / (4 * ln(2))
        hl_ratio = np.maximum(h / np.maximum(l, 1e-8), 1.0)
        parkinson_vol = np.sqrt((np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0)))

        z = np.column_stack([log_ret, parkinson_vol])
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GMMRegimeFilter:
        """Estimate mixture parameters Theta on training fold strictly.

        Args:
            X: Training DataFrame containing OHLC bars.
            y: Ignored target parameter.

        Returns:
            GMMRegimeFilter: Fitted model with determined bullish regime index.
        """
        z_train = self._extract_regime_tensor(X)

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=200,
            random_state=self.random_state,
        )
        self.gmm_.fit(z_train)

        # Bullish regime index is the cluster with the highest mean return: k_bull = argmax(mu_{k, 1})
        means = self.gmm_.means_  # Shape (n_components, 2)
        self.bull_regime_index_ = int(np.argmax(means[:, 0]))

        logger.info(
            "GMM Regime: Identified Bull component %d with mean return %.4f",
            self.bull_regime_index_,
            means[self.bull_regime_index_, 0],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Infer bullish regime posterior probabilities on out-of-sample data.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame augmented with P(Regime = Bull | z_t).
        """
        if self.gmm_ is None:
            raise RuntimeError("GMMRegimeFilter must be fitted before transforming.")

        df = X.copy()
        z_test = self._extract_regime_tensor(df)

        # Evaluate posterior probabilities P(Regime = k | z_t)
        posteriors = self.gmm_.predict_proba(z_test)
        df["regime_bull_prob"] = posteriors[:, self.bull_regime_index_]

        return df