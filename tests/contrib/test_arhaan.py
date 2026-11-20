from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from src.contrib.arhaan import variance_ratio


def ar1(phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.standard_normal(n)
    return lfilter([1.0], [1.0, -phi], eps)


def test_random_walk_slowly_changing_vol():
    rng = np.random.default_rng(11)
    n, runs, q = 4_000, 200, 24
    t = np.arange(n)
    vrs, rejections = [], 0
    for i in range(runs):
        phase = rng.uniform(0, 2 * np.pi)
        sigma = 0.01 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 1_000 + phase))
        r = sigma * rng.standard_normal(n)
        vr, z = variance_ratio(r, q)
        vrs.append(vr)
        rejections += abs(z) > 1.96
    assert abs(np.mean(vrs) - 1.0) < 0.03
    assert 0.02 <= rejections / runs <= 0.10


def test_ar1_matches_theoretical_vr():
    rng = np.random.default_rng(22)
    phi, q = 0.1, 24
    r = ar1(phi, 200_000, rng)
    vr, _ = variance_ratio(r, q)
    expected = 1 + 2 * sum((1 - j / q) * phi**j for j in range(1, q))
    assert abs(vr - expected) < 0.08


def test_q2_equals_one_plus_first_autocorrelation():
    rng = np.random.default_rng(33)
    r = ar1(0.2, 500, rng)
    vr, _ = variance_ratio(r, 2)
    d = r - r.mean()
    rho1 = np.sum(d[1:] * d[:-1]) / np.sum(d**2)
    assert abs(vr - (1 + rho1)) < 0.01


def test_nan_in_input_does_not_break_it():
    rng = np.random.default_rng(44)
    r = 0.01 * rng.standard_normal(1_000)
    r_nan = r.copy()
    r_nan[[0, 10, 500, 999]] = np.nan
    vr, z = variance_ratio(r_nan, 24)
    assert np.isfinite(vr) and np.isfinite(z)
    vr_clean, z_clean = variance_ratio(r_nan[~np.isnan(r_nan)], 24)
    assert vr == vr_clean and z == z_clean
