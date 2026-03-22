"""Lightweight periodic advection data: u_t + u_x = 0 (v=1), FTCS-style roll."""

from __future__ import annotations

import torch


def lowpass_ic(n_traj: int, nx: int, n_modes: int, rng: torch.Generator, device) -> torch.Tensor:
    raw = torch.randn(n_traj, nx, generator=rng, device=device)
    fk = torch.fft.rfft(raw, dim=-1)
    fk[:, n_modes:] = 0
    ic = torch.fft.irfft(fk, n=nx, dim=-1)
    ic = ic - ic.mean(dim=-1, keepdim=True)
    return ic


def rollout_advection(ic: torch.Tensor, nt: int) -> torch.Tensor:
    """Upwind shift per step: u^{n+1}_i = u^n_{i-1} (periodic). Shape (B, nx, nt)."""
    b, nx = ic.shape
    u = torch.empty(b, nx, nt, device=ic.device, dtype=ic.dtype)
    u[:, :, 0] = ic
    for t in range(1, nt):
        u[:, :, t] = torch.roll(u[:, :, t - 1], shifts=1, dims=-1)
    return u


def coord_grid(nx: int, batch: int, device, dtype):
    xc = torch.linspace(0, 1, nx, device=device, dtype=dtype).view(1, nx, 1).expand(batch, -1, -1)
    return xc
