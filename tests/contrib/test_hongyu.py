from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import f as f_dist

from src.contrib.hongyu import seasonal_wald

N_DAYS = 623
HOURS = 24


def _panel(seed: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """623 days of hourly N(0, 1) noise scaled by a daily volatility exp(0.5 * N(0, 1))."""
    rng = np.random.default_rng(seed)
    day_vol = np.exp(0.5 * rng.standard_normal(N_DAYS))
    x = (rng.standard_normal((N_DAYS, HOURS)) * day_vol[:, None]).ravel()
    hour = np.tile(np.arange(HOURS), N_DAYS)
    day = np.repeat(np.arange(N_DAYS), HOURS)
    week = day // 7
    if shift:
        x = x + shift * (hour == 13)
    return x, hour, day, week


def _ols_cluster_wald(x: np.ndarray, group: np.ndarray, cluster: np.ndarray) -> float:
    """Reference: OLS on group dummies, CR1-style sandwich with C / (C - 1)."""
    groups, g_idx = np.unique(group, return_inverse=True)
    clusters, c_idx = np.unique(cluster, return_inverse=True)
    design = np.zeros((len(x), len(groups)))
    design[np.arange(len(x)), g_idx] = 1.0
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ beta
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((len(groups), len(groups)))
    for c in range(len(clusters)):
        score = design[c_idx == c].T @ resid[c_idx == c]
        meat += np.outer(score, score)
    n_clusters = len(clusters)
    vcov = n_clusters / (n_clusters - 1) * bread @ meat @ bread
    restriction = np.hstack([np.eye(len(groups) - 1), -np.ones((len(groups) - 1, 1))])
    d = restriction @ beta
    return float(d @ np.linalg.inv(restriction @ vcov @ restriction.T) @ d)


def test_group_means_equal_plain_means() -> None:
    x, hour, day, _ = _panel(1)
    means, _, _ = seasonal_wald(x, hour, day)
    expected = np.array([x[hour == h].mean() for h in range(HOURS)])
    np.testing.assert_allclose(means, expected, rtol=0, atol=1e-12)


def test_wald_matches_ols_cluster_robust() -> None:
    x, hour, day, _ = _panel(2)
    _, wald, _ = seasonal_wald(x, hour, day)
    assert wald == pytest.approx(_ols_cluster_wald(x, hour, day), rel=1e-8)


def test_p_value_matches_f_formula() -> None:
    x, hour, _, week = _panel(3)
    _, wald, p_value = seasonal_wald(x, hour, week)
    q, n_clusters = HOURS - 1, len(np.unique(week))
    f_stat = wald * (n_clusters - q) / (q * (n_clusters - 1))
    assert p_value == pytest.approx(f_dist.sf(f_stat, q, n_clusters - q), rel=1e-12)


@pytest.mark.parametrize("cluster_kind", ["day", "week"])
def test_size_under_the_null(cluster_kind: str) -> None:
    rejections = 0
    for seed in range(500):
        x, hour, day, week = _panel(10_000 + seed)
        _, _, p_value = seasonal_wald(x, hour, day if cluster_kind == "day" else week)
        rejections += p_value < 0.05
    assert 0.02 <= rejections / 500 <= 0.08


def test_power_against_a_one_hour_shift() -> None:
    rejections = 0
    for seed in range(50):
        x, hour, day, _ = _panel(20_000 + seed, shift=0.4)
        _, _, p_value = seasonal_wald(x, hour, day)
        rejections += p_value < 0.05
    assert rejections / 50 >= 0.90


def test_nan_input_matches_dropping_rows() -> None:
    x, hour, day, _ = _panel(4)
    rng = np.random.default_rng(5)
    missing = rng.random(len(x)) < 0.05
    with_nan = x.copy()
    with_nan[missing] = np.nan
    means_a, wald_a, p_a = seasonal_wald(with_nan, hour, day)
    means_b, wald_b, p_b = seasonal_wald(x[~missing], hour[~missing], day[~missing])
    np.testing.assert_array_equal(means_a, means_b)
    assert wald_a == wald_b
    assert p_a == p_b
