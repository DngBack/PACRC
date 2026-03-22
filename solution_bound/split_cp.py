"""
Finite-sample split conformal quantiles (exchangeable nonconformity scores).

Compatible with the scoring used in CP-PRE: typically |PRE| per grid point or
flattened over the calibration tensor. See Tibshirani et al. (2019) and the
inductive CP construction in Vovk et al.
"""

from __future__ import annotations

import numpy as np


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Calibration threshold q such that, for exchangeable scores,
    P(S_{n+1} <= q) >= 1 - alpha at the ideal population level, with the usual
    finite-sample adjustment (ceil((n+1)(1-alpha))/n quantile).

    Parameters
    ----------
    scores : array
        Nonconformity scores on the calibration set (any shape; flattened).
    alpha : float in (0, 1)
        Miscoverage level.

    Returns
    -------
    float
        Calibrated quantile (same units as scores).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        raise ValueError("Calibration set is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    return float(s[k - 1])
