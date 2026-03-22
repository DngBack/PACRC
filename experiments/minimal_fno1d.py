"""
Tiny 1D Fourier Neural Operator for periodic domains (autoregressive one-step).

Input:  (B, nx, T_in)   — last T_in time slices along x
Output: (B, nx, 1)     — next time slice

No dependency on Neural_PDE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * in_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, nx)
        b, _, nx = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            b, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=nx, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral(x) + self.w(x)


class FNO1dMini(nn.Module):
    """One-step predictor: concat time channels + normalized x coordinate."""

    def __init__(self, T_in: int, width: int = 32, modes: int = 8, n_layers: int = 3):
        super().__init__()
        self.T_in = T_in
        self.fc0 = nn.Linear(T_in + 1, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, x_coord: torch.Tensor) -> torch.Tensor:
        """
        x: (B, nx, T_in) — solution history on spatial grid
        x_coord: (B, nx, 1) — values in [0,1]
        """
        h = torch.cat([x, x_coord], dim=-1)
        h = self.fc0(h)
        h = h.permute(0, 2, 1)
        for blk in self.blocks:
            h = F.gelu(blk(h))
        h = h.permute(0, 2, 1)
        h = F.gelu(self.fc1(h))
        return self.fc2(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
