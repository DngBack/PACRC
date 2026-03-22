"""
Conformal risk control style selection using a surrogate risk = C(x) * |r(x)|.

This is a lightweight empirical analogue of CRC (Angelopoulos et al.): pick a
threshold tau on the calibration surrogate so that the empirical mean risk on
accepted calibration points is at most epsilon. Guarantees require the formal
CRC construction; this module is for engineering integration and ablations.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch


class CRCSelector:
    def __init__(self, surrogate_risk_fn: Callable[..., torch.Tensor]):
        """
        surrogate_risk_fn(x, u_hat, ...) -> non-negative scalar or tensor risk.
        """
        self.surrogate_risk_fn = surrogate_risk_fn
        self.tau: Optional[float] = None

    def calibrate_tau(
        self,
        risks: np.ndarray,
        epsilon: float,
    ) -> float:
        """
        Choose the largest tau such that mean(risks[risks <= tau]) <= epsilon
        (if any); otherwise tau = min(risks) (conservative).
        """
        r = np.sort(np.asarray(risks, dtype=np.float64).ravel())
        n = r.size
        if n == 0:
            raise ValueError("Empty risk array.")
        best_tau = float(r[0])
        for i in range(n):
            t = float(r[i])
            subset = r[r <= t]
            if subset.mean() <= epsilon:
                best_tau = t
        self.tau = best_tau
        return best_tau

    def predict(
        self,
        risk: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        if self.tau is None:
            raise RuntimeError("Call calibrate_tau first.")
        val = float(risk.mean() if risk.numel() > 1 else risk.item())
        if val <= self.tau:
            return "accept", risk
        return "reject", risk
