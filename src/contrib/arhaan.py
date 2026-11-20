import numpy as np


def variance_ratio(returns: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay (1988) variance ratio with bias-corrected estimators and
    the heteroskedasticity-robust z* statistic.

    returns: 1-period log returns. q: aggregation horizon (q >= 2).
    Returns (VR(q), z*(q)).
    """
    if q < 2:
        raise ValueError("q must be >= 2")

    r = returns[~np.isnan(returns)]  # NOTE: joins returns across gaps; check the issue's NaN rule
    nq = len(r)
    if nq <= q:
        raise ValueError("need more than q returns")

    mu = r.mean()

    # 1-period variance, unbiased: divide by nq - 1
    var_1 = np.sum((r - mu) ** 2) / (nq - 1)

    # overlapping q-period sums X_k - X_{k-q}, for every window (nq - q + 1 of them)
    csum = np.concatenate(([0.0], np.cumsum(r)))
    q_sums = csum[q:] - csum[:-q]

    # q-period variance (per period), bias-corrected: divide by m
    m = q * (nq - q + 1) * (1 - q / nq)
    var_q = np.sum((q_sums - q * mu) ** 2) / m

    VR = var_q / var_1

    # heteroskedasticity-robust asymptotic variance theta(q)
    dev2 = (r - mu) ** 2
    denom = np.sum(dev2) ** 2
    theta = 0.0
    for j in range(1, q):  # j = 1 .. q-1
        delta_j = nq * np.sum(dev2[j:] * dev2[:-j]) / denom
        theta += (2 * (q - j) / q) ** 2 * delta_j

    z_star = np.sqrt(nq) * (VR - 1) / np.sqrt(theta)
    return float(VR), float(z_star)
