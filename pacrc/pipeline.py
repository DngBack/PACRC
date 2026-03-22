"""
End-to-end PACRC marginal pipeline: calibrate q_α on |PRE|, map to solution bounds C·q.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch

from solution_bound.mapper import SolutionBoundMapper
from stability.C_estimator import StabilityEstimator, TinyMLP, TinyMLP3d

from pacrc.metrics import summarize_run, to_numpy

Array = Union[np.ndarray, torch.Tensor]


class PACRCMarginalPipeline:
    """
    Label-free calibration uses only nonconformity scores |r(x)| on the calibration fold.
    Evaluation of solution coverage uses u* (not available at deploy time).

    Parameters
    ----------
    alpha : float
        Miscoverage level for split conformal (target marginal coverage 1 - alpha).
    stability_mode : 'constant' | 'jacobian_fd' | 'learned'
    C_global : float
        Used when stability_mode == 'constant'.
    g_phi : nn.Module | None
        Used when stability_mode == 'learned' (train on a separate train split).
    fd_eps : float
        Finite-difference step for jacobian_fd (1D-style grids only).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        stability_mode: str = "constant",
        C_global: float = 1.0,
        g_phi: Optional[torch.nn.Module] = None,
        fd_eps: float = 1e-3,
    ):
        est = StabilityEstimator(
            mode=stability_mode,  # type: ignore[arg-type]
            C_global=C_global,
            g_phi=g_phi,
            fd_eps=fd_eps,
        )
        self.mapper = SolutionBoundMapper(est, alpha=alpha)
        self.alpha = alpha

    def calibrate(
        self,
        cal_r_mag: Array,
        cal_u_hat: Optional[Array] = None,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> float:
        """
        Set q_α from calibration residuals (|r| flattened, same as CP-PRE).
        cal_u_hat is optional (reserved for future score variants).
        """
        del cal_u_hat, residual_fn
        arr = np.abs(to_numpy(cal_r_mag).ravel())
        return self.mapper.calibrate_from_residual_scores(arr)

    def calibrate_from_scores(self, scores: np.ndarray) -> float:
        return self.mapper.calibrate_from_residual_scores(np.abs(np.asarray(scores).ravel()))

    def bound_field(
        self,
        u_hat: torch.Tensor,
        r_mag: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mapper.bound_field(u_hat, r_mag, residual_fn=residual_fn, features=features)

    def evaluate(
        self,
        u_hat_cal: torch.Tensor,
        u_star_cal: torch.Tensor,
        r_cal: torch.Tensor,
        u_hat_test: torch.Tensor,
        u_star_test: torch.Tensor,
        r_test: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        features_cal: Optional[torch.Tensor] = None,
        features_test: Optional[torch.Tensor] = None,
        abs_err_test: Optional[torch.Tensor] = None,
        abs_err_cal: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full split: calibrate q on |r_cal|, build bounds on test using r_test and C(·).

        For Navier–Stokes, pass abs_err_{test,cal} = ||(û−u*)||_2 per grid point (same
        shape as |r| on the inner grid). features_* are optional inputs for learned C(x).
        """
        del features_cal
        self.calibrate(r_cal)
        r_eval = r_test.abs() if r_test.dtype != torch.bool else r_test
        b_test = self.bound_field(
            u_hat_test,
            r_eval,
            residual_fn=residual_fn,
            features=features_test,
        )
        q = float(self.mapper.q_alpha)  # type: ignore[arg-type]
        return summarize_run(
            to_numpy(u_hat_cal),
            to_numpy(u_star_cal),
            to_numpy(r_cal),
            to_numpy(u_hat_test),
            to_numpy(u_star_test),
            to_numpy(r_test),
            to_numpy(b_test),
            self.alpha,
            q,
            abs_err_test=abs_err_test,
            abs_err_cal=abs_err_cal,
        )


def make_learned_head(spatial_dims: int, hidden: int = 32) -> torch.nn.Module:
    """Factory: spatial_dims 2 -> TinyMLP (1D+time as H,W), 3 -> TinyMLP3d."""
    if spatial_dims == 2:
        return TinyMLP(in_features=2, hidden=hidden)
    if spatial_dims == 3:
        return TinyMLP3d(in_features=2, hidden=hidden)
    raise ValueError("spatial_dims must be 2 or 3")
