"""
Burgers PRE residual r(û) matching Marginal/Burgers_Residuals_CP.py conventions.

Expects u of shape (B, Nt, Nx) (batch, time, space) as after permute [:,0] in the
Marginal script.
"""

from __future__ import annotations

import torch

from Utils.ConvOps_1d import ConvOperator


def make_burgers_residual(dx: torch.Tensor, dt: torch.Tensor, nu: torch.Tensor):
    D_t = ConvOperator(domain="t", order=1)
    D_x = ConvOperator(domain="x", order=1)
    D_xx = ConvOperator(domain="x", order=2)

    def residual(u: torch.Tensor, boundary: bool = False):
        """
        u: (B, Nt, Nx)
        Returns interior stencil shape as in CP-PRE when boundary=False.
        """
        res = dx * D_t(u) + dt * u * D_x(u) - nu * D_xx(u) * (2 * dt / dx)
        if boundary:
            return res
        return res[..., 1:-1, 1:-1]

    return residual
