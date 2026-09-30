"""Non-Linear Hierarchical Medoid Extraction Module."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class HRPMedoidExtractor(BaseEstimator, TransformerMixin):
    """Hierarchical Risk Parity (HRP) inspired Medoid Pruner.

    Steps:
      1. Compute Spearman correlation matrix rho_spearman.
      2. Derive metric distance: D = sqrt(0.5 * (1.0 - rho_spearman)).
      3. Perform Ward hierarchical clustering up to C_max clusters.
      4. For each cluster c, select the medoid:
            medoid = argmin_{i in cluster} sum_{j in cluster} D[i, j]
    """

    def __init__(self, max_clusters: int = 14) -> None:
        """Initialize medoid cluster extractor.

        Args:
            max_clusters: Upper bound on the number of clusters (C_max).
        """
        self.max_clusters = max_clusters
        self.selected_medoids_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HRPMedoidExtractor:
        """Cluster feature space and isolate medoid features.

        Args:
            X: Orthogonalized feature candidate DataFrame.
            y: Ignored target parameter.

        Returns:
            HRPMedoidExtractor: Fitted instance containing selected medoid column names.
        """
        cols = list(X.columns)
        n_features = len(cols)

        if n_features <= self.max_clusters:
            self.selected_medoids_ = cols
            return self

        # 1. Non-linear rank correlation (Spearman)
        spearman_corr = X[cols].corr(method="spearman").to_numpy()
        spearman_corr = np.nan_to_num(spearman_corr, nan=0.0)
        np.fill_diagonal(spearman_corr, 1.0)

        # 2. Distance metric D_metric = sqrt(0.5 * (1 - rho))
        dist_matrix = np.sqrt(np.clip(0.5 * (1.0 - spearman_corr), 0.0, 1.0))
        np.fill_diagonal(dist_matrix, 0.0)

        # 3. Ward's linkage clustering
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_tree = linkage(condensed_dist, method="ward")

        # Flat cut into at most C_max clusters
        k_clusters = min(self.max_clusters, n_features)
        cluster_labels = fcluster(linkage_tree, t=k_clusters, criterion="maxclust")

        # 4. Medoid extraction per cluster
        medoids = []
        for cluster_id in np.unique(cluster_labels):
            indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
            if len(indices_in_cluster) == 1:
                medoids.append(cols[indices_in_cluster[0]])
                continue

            sub_dist = dist_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]
            sum_distances = sub_dist.sum(axis=1)
            best_idx = indices_in_cluster[np.argmin(sum_distances)]
            medoids.append(cols[best_idx])

        self.selected_medoids_ = medoids
        logger.debug("HRPMedoids: Condensed %d features into %d medoids.", n_features, len(medoids))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by extracted medoids.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Sub-selected medoid features.
        """
        if not self.selected_medoids_:
            return X
        return X[self.selected_medoids_]