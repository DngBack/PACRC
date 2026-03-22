"""
Evaluation metrics for CP-PRE (residual space) and PACRC (solution space).

Residual marginal coverage matches the usual CP-PRE diagnostic: fraction of grid
points whose residual lies in [-q, q] with q from split conformal on |r_cal|.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from solution_bound.split_cp import split_conformal_quantile

Array = Union[np.ndarray, torch.Tensor]


def to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(u_hat: Array, u_star: Array) -> float:
    a = to_numpy(u_hat).ravel()
    b = to_numpy(u_star).ravel()
    return float(np.mean((a - b) ** 2))


def residual_marginal_coverage(
    r_cal: Array,
    r_test: Array,
    alpha: float,
    symmetric: bool = True,
) -> Tuple[float, float]:
    """
    Marginal coverage: P(|r_test| <= q) with q = split conformal quantile of |r_cal|.
    Returns (coverage, q).
    """
    s_cal = np.abs(to_numpy(r_cal).ravel())
    s_test = to_numpy(r_test).ravel()
    q = split_conformal_quantile(s_cal, alpha)
    if symmetric:
        cov = float((np.abs(s_test) <= q).mean())
    else:
        cov = float(((s_test >= -q) & (s_test <= q)).mean())
    return cov, q


def solution_pointwise_coverage(
    u_hat: Array,
    u_star: Array,
    bound: Array,
) -> Tuple[float, np.ndarray]:
    """
    Fraction of grid points where |û - u*| <= bound (elementwise same shape).
    """
    e = np.abs(to_numpy(u_hat) - to_numpy(u_star))
    b = to_numpy(bound)
    if e.shape != b.shape:
        raise ValueError(f"|e| shape {e.shape} != bound shape {b.shape}")
    ind = (e <= b).astype(np.float64)
    return float(ind.mean()), ind


def mean_bound_width(bound: Array) -> float:
    return float(np.mean(to_numpy(bound)))


def correlation_bound_error(
    u_hat: Array,
    u_star: Array,
    bound: Array,
) -> float:
    """Pearson r between flatten(|e|) and flatten(bound)."""
    e = np.abs(to_numpy(u_hat) - to_numpy(u_star)).ravel()
    b = to_numpy(bound).ravel()
    if e.size < 2 or np.std(e) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(e, b)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def summarize_run(
    u_hat_cal: Array,
    u_star_cal: Array,
    r_cal: Array,
    u_hat_test: Array,
    u_star_test: Array,
    r_test: Array,
    bound_test: Array,
    alpha: float,
    q: float,
    abs_err_test: Optional[Array] = None,
    abs_err_cal: Optional[Array] = None,
) -> Dict[str, Any]:
    """
    If abs_err_test is provided (same shape as bound_test), solution metrics use it
    instead of |u_hat - u_star| (Navier–Stokes: L2 norm over velocity/pressure channels).
    """
    cov_res, _ = residual_marginal_coverage(r_cal, r_test, alpha)
    if abs_err_test is None:
        cov_sol, _ = solution_pointwise_coverage(u_hat_test, u_star_test, bound_test)
        corr = correlation_bound_error(u_hat_test, u_star_test, bound_test)
    else:
        cov_sol, _ = solution_pointwise_coverage_from_abs_err(abs_err_test, bound_test)
        corr = correlation_bound_error_from_abs_err(abs_err_test, bound_test)
    out: Dict[str, Any] = {
        "alpha": alpha,
        "q_alpha": q,
        "mse_test": mse(u_hat_test, u_star_test),
        "mse_cal": mse(u_hat_cal, u_star_cal),
        "residual_marginal_coverage_test": cov_res,
        "solution_pointwise_coverage_test": cov_sol,
        "mean_bound_width_test": mean_bound_width(bound_test),
        "corr_abs_err_bound_test": corr,
    }
    return out


def solution_pointwise_coverage_from_abs_err(abs_err: Array, bound: Array) -> Tuple[float, np.ndarray]:
    e = to_numpy(abs_err)
    b = to_numpy(bound)
    if e.shape != b.shape:
        raise ValueError(f"|e| shape {e.shape} != bound shape {b.shape}")
    ind = (e <= b).astype(np.float64)
    return float(ind.mean()), ind


def correlation_bound_error_from_abs_err(abs_err: Array, bound: Array) -> float:
    e = to_numpy(abs_err).ravel()
    b = to_numpy(bound).ravel()
    if e.size < 2 or np.std(e) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(e, b)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")
