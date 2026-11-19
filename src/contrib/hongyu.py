"""Cluster-robust seasonal Wald test for Track 12 (#19)."""

from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist


def seasonal_wald(
    x: np.ndarray, group: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Test whether ``x`` has the same mean in every group.

    Standard errors are cluster-robust: observations in the same cluster may be
    correlated in any way, observations in different clusters are independent.
    Observations where ``x`` is NaN are dropped first.

    With group means ``m_g`` and counts ``n_g``, each observation contributes
    ``u_i = (x_i - m_g) / n_g``. Summing ``u`` by cluster and group gives the
    ``C x G`` matrix ``S`` and the covariance of the means
    ``V = C / (C - 1) * S.T @ S``. With ``R = [I | -1]`` and ``d = R @ m``,
    ``W = d @ inv(R @ V @ R.T) @ d``. The p-value uses the small-sample F form
    ``F = W * (C - q) / (q * (C - 1))`` with ``q = G - 1`` and ``C - q``
    denominator degrees of freedom.

    Returns the group means (ordered as ``np.unique(group)``), ``W`` and the
    p-value.
    """
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group)
    cluster = np.asarray(cluster)
    if x.ndim != 1 or group.ndim != 1 or cluster.ndim != 1:
        raise ValueError("x, group and cluster must be one-dimensional")
    if not (len(x) == len(group) == len(cluster)):
        raise ValueError("x, group and cluster must have the same length")

    keep = ~np.isnan(x)
    x, group, cluster = x[keep], group[keep], cluster[keep]
    if not np.isfinite(x).all():
        raise ValueError("x must not contain infinite values")

    groups, g_idx = np.unique(group, return_inverse=True)
    _, c_idx = np.unique(cluster, return_inverse=True)
    n_groups = len(groups)
    n_clusters = int(c_idx.max()) + 1 if len(c_idx) else 0
    q = n_groups - 1
    if n_groups < 2:
        raise ValueError("at least two groups with data are needed")
    if n_clusters <= q:
        raise ValueError("the number of clusters must exceed the number of groups minus one")

    counts = np.bincount(g_idx, minlength=n_groups).astype(np.float64)
    means = np.bincount(g_idx, weights=x, minlength=n_groups) / counts

    u = (x - means[g_idx]) / counts[g_idx]
    scores = np.zeros((n_clusters, n_groups), dtype=np.float64)
    np.add.at(scores, (c_idx, g_idx), u)
    cov = n_clusters / (n_clusters - 1) * (scores.T @ scores)

    restriction = np.hstack([np.eye(q), -np.ones((q, 1))])
    d = restriction @ means
    middle = restriction @ cov @ restriction.T
    wald = float(d @ np.linalg.solve(middle, d))

    f_stat = wald * (n_clusters - q) / (q * (n_clusters - 1))
    p_value = float(f_dist.sf(f_stat, q, n_clusters - q))
    return means, wald, p_value
