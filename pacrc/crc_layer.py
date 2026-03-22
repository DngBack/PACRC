"""
Selective prediction: threshold surrogate risk built from PACRC bounds.

Formal CRC guarantees (Angelopoulos et al.) need their construction; this module
implements the engineering pattern: calibrate τ on mean surrogate risk over the
calibration trajectories, then accept/reject test trajectories.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from crc.selector import CRCSelector


def default_surrogate_from_bound(bound: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Scalar risk from a bound field (proxy for solution error scale)."""
    if reduction == "mean":
        return bound.mean()
    if reduction == "max":
        return bound.amax()
    raise ValueError("reduction must be 'mean' or 'max'")


class PACRCCRC:
    def __init__(
        self,
        bound_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        risk_reduction: str = "mean",
    ):
        """
        bound_fn(u_hat, r_mag) -> bound tensor (same grid as error evaluation).

        The selector stores τ; predict compares surrogate risk to τ.
        """
        self.bound_fn = bound_fn
        self.risk_reduction = risk_reduction
        self._selector = CRCSelector(surrogate_risk_fn=self._dummy)
        self.tau: Optional[float] = None

    @staticmethod
    def _dummy(**kwargs):
        return torch.tensor(0.0)

    def calibrate_tau(
        self,
        u_hats: List[torch.Tensor],
        r_mags: List[torch.Tensor],
        epsilon: float,
    ) -> float:
        risks = []
        for uh, rm in zip(u_hats, r_mags):
            b = self.bound_fn(uh, rm)
            risks.append(float(default_surrogate_from_bound(b, self.risk_reduction)))
        tau = self._selector.calibrate_tau(np.array(risks, dtype=np.float64), epsilon)
        self.tau = tau
        return tau

    def accept(self, u_hat: torch.Tensor, r_mag: torch.Tensor) -> bool:
        if self.tau is None:
            raise RuntimeError("Call calibrate_tau first.")
        b = self.bound_fn(u_hat, r_mag)
        val = float(default_surrogate_from_bound(b, self.risk_reduction))
        return val <= self.tau
