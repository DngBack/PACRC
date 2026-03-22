"""
Map CP-PRE residual calibration (quantile of |r|) to a solution-space bound field
using a stability multiplier C(x).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .split_cp import split_conformal_quantile
from stability.C_estimator import StabilityEstimator


class SolutionBoundMapper:
    """
    Converts CP-PRE-style residual scores into bound = C * q_alpha on the inner grid.

    Parameters
    ----------
    stability : StabilityEstimator
    alpha : float
        Target miscoverage (passed to split conformal quantile).
    """

    def __init__(
        self,
        stability: StabilityEstimator,
        alpha: float = 0.1,
    ):
        self.stability = stability
        self.alpha = alpha
        self.q_alpha: Optional[float] = None

    def calibrate_from_residual_scores(self, scores: np.ndarray) -> float:
        self.q_alpha = split_conformal_quantile(scores, self.alpha)
        return self.q_alpha

    def calibrate(
        self,
        cal_r_mag: torch.Tensor,
        cal_u_hat: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> float:
        scores = cal_r_mag.detach().cpu().numpy()
        self.calibrate_from_residual_scores(np.abs(scores))
        # Warm C if learned — user fits g_phi offline; nothing to do here.
        return self.q_alpha  # type: ignore

    def stability_field(
        self,
        u_hat: torch.Tensor,
        r_mag: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.stability(u_hat, r_mag, residual_fn=residual_fn, features=features)

    def bound_field(
        self,
        u_hat: torch.Tensor,
        r_mag: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.q_alpha is None:
            raise RuntimeError("Call calibrate(...) first.")
        C = self.stability_field(u_hat, r_mag, residual_fn=residual_fn, features=features)
        return C * self.q_alpha

    @staticmethod
    def error_magnitude(u_hat: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
        """Pointwise |û - u*| on the same grid as u_hat (evaluation only)."""
        return (u_hat - u_star).abs()

    def coverage_indicator(
        self,
        u_hat: torch.Tensor,
        u_star: torch.Tensor,
        r_mag: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ord_reduce: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (indicator, ratio) where indicator is 1 if ||e|| <= ||bound|| under ord_reduce.

        ord_reduce: 'max' compares max |e| vs max bound; 'mean' uses means over the inner grid.
        """
        bound = self.bound_field(u_hat, r_mag, residual_fn=residual_fn)
        e = self.error_magnitude(u_hat, u_star)
        # Align shapes: evaluation scripts often slice u to inner grid
        if e.shape != bound.shape:
            raise ValueError(f"Shape mismatch e {e.shape} vs bound {bound.shape}.")
        if ord_reduce == "max":
            lhs = e.amax(dim=tuple(range(1, e.ndim)))
            rhs = bound.amax(dim=tuple(range(1, bound.ndim)))
        elif ord_reduce == "mean":
            lhs = e.mean(dim=tuple(range(1, e.ndim)))
            rhs = bound.mean(dim=tuple(range(1, bound.ndim)))
        else:
            raise ValueError("ord_reduce must be 'max' or 'mean'.")
        return (lhs <= rhs).float(), lhs / (rhs + 1e-12)
