"""
Thin wrappers around Other_UQ PRE_Wave / PRE_NS for PACRC pipelines.

Tensor layout matches Other_UQ/Evaluation/PRE_estimations.py and data_loaders:
  Wave: u tensor (B, 1, Nx, Ny, Nt)
  NS:   vars (B, 3, Nx, Ny, Nt) — channels u, v, p
"""

from __future__ import annotations

import torch

from Other_UQ.Evaluation.PRE_estimations import PRE_NS, PRE_Wave


def wave_pre_operators(dt: float, dx: float, c: float = 1.0) -> PRE_Wave:
    return PRE_Wave(torch.tensor(dt), torch.tensor(dx), c=c)


def ns_pre_operators(dt: float, dx: float, dy: float) -> PRE_NS:
    return PRE_NS(torch.tensor(dt), torch.tensor(dx), torch.tensor(dy))


def wave_residual_mag(u_batch: torch.Tensor, pre: PRE_Wave) -> torch.Tensor:
    """|r| on interior grid; u_batch (B,1,Nx,Ny,Nt)."""
    r = pre.residual(u_batch, boundary=False)
    return r.abs()


def ns_residual_mag(vars_batch: torch.Tensor, pre: PRE_NS) -> torch.Tensor:
    """|r| on interior; vars_batch (B,3,Nx,Ny,Nt)."""
    r = pre.residual(vars_batch, boundary=False)
    return r.abs()
