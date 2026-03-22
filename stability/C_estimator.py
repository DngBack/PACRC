"""
Stability / sensitivity multipliers C(x) mapping residual scale to solution bound scale.

These estimators are meant to be composed with CP-PRE marginal quantiles q_alpha:
    bound(x) = C(x) * q_alpha
under explicit stability assumptions (see paper). Learned C must be fit on data
that is disjoint from the conformal calibration fold to avoid leakage.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn as nn

Mode = Literal["constant", "jacobian_fd", "learned"]


class StabilityEstimator:
    """
    Maps PDE operator sensitivity to a (possibly spatial) field C aligned with r(x).
    """

    def __init__(
        self,
        mode: Mode = "constant",
        C_global: float = 1.0,
        g_phi: Optional[nn.Module] = None,
        fd_eps: float = 1e-3,
    ):
        self.mode = mode
        self.C_global = float(C_global)
        self.g_phi = g_phi
        self.fd_eps = fd_eps

    def __call__(
        self,
        u_hat: torch.Tensor,
        r_mag: torch.Tensor,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "constant":
            return self.constant_field(r_mag)
        if self.mode == "jacobian_fd":
            if residual_fn is None:
                raise ValueError("jacobian_fd requires residual_fn(u).")
            rm = self._to_spatial_tensor(r_mag)
            if rm.dim() != 3:
                raise NotImplementedError(
                    "jacobian_fd is only implemented for 1D+time grids with shape (B, Nt, Nx)."
                )
            return self.jacobian_fd_local(u_hat, residual_fn, rm.shape)
        if self.mode == "learned":
            if self.g_phi is None:
                raise ValueError("learned mode requires g_phi module.")
            if features is None:
                features = self.default_features(u_hat, r_mag)
            out = self.g_phi(features)
            if out.dim() == features.dim() - 1:
                return out
            return out.squeeze(1)
        raise ValueError(f"Unknown mode {self.mode}")

    def constant_field(self, r_mag: torch.Tensor) -> torch.Tensor:
        """Scalar C on every grid point; supports arbitrary û / r layouts."""
        return torch.full_like(r_mag, self.C_global, dtype=r_mag.dtype, device=r_mag.device)

    def default_features(self, u_hat: torch.Tensor, r_mag: torch.Tensor) -> torch.Tensor:
        """Stack |u| and |r| as channels: 4D -> [B,2,H,W], 5D -> [B,2,Nx,Ny,Nt]."""
        if u_hat.shape != r_mag.shape:
            raise ValueError("u_hat and r_mag must match for default_features.")
        a = u_hat.abs()
        b = r_mag.abs()
        if a.dim() == 3:
            return torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
        if a.dim() == 4:
            return torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
        if a.dim() == 5:
            return torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)
        raise ValueError("default_features supports 3D/4D/5D tensors (after channel squeeze if any).")

    def jacobian_fd_local(
        self,
        u_hat: torch.Tensor,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Per-grid sensitivity using finite differences on a 3x3 patch around each
        inner point where r is defined.

        For one residual component r_ij, we approximate the row Jacobian
        J = dr/du_patch (1x9) and set C_ij = ||J^+||_2 = 1 / ||J||_2 for this
        scalar-output local map.

        `residual_fn` must accept u of shape (B, Nt, Nx) as in CP-PRE ConvOperator
        pipelines (see Marginal/Burgers_Residuals_CP.py).

        Note: This is expensive (O(#inner * 9) residual evaluations) but avoids
        autograd through the full grid; use on moderate grids or subsample for
        large experiments.
        """
        u3 = self._to_spatial_tensor(u_hat).detach()
        B, Nt, Nx = u3.shape
        device, dtype = u3.device, u3.dtype
        eps = torch.tensor(self.fd_eps, device=device, dtype=dtype)

        # Assume r lives on inner grid [1:Nt-1, 1:Nx-1] like Marginal/Burgers_Residuals_CP
        out = torch.zeros(B, target_shape[-2], target_shape[-1], device=device, dtype=dtype)
        for b in range(B):
            base = residual_fn(u3[b : b + 1])[0]
            ti_max, xi_max = base.shape[0], base.shape[1]
            for ti in range(ti_max):
                for xi in range(xi_max):
                    t0 = ti + 1
                    x0 = xi + 1
                    grads = []
                    for dt in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            u2 = u3.clone()
                            u2[b, t0 + dt, x0 + dx] = u2[b, t0 + dt, x0 + dx] + eps
                            r2 = residual_fn(u2[b : b + 1])[0, ti, xi]
                            g = (r2 - base[ti, xi]) / eps
                            grads.append(g)
                    J = torch.stack(grads).reshape(1, -1)
                    nrm = torch.linalg.norm(J)
                    out[b, ti, xi] = 1.0 / (nrm + torch.finfo(dtype).eps)
        return out

    @staticmethod
    def _to_spatial_tensor(x: torch.Tensor) -> torch.Tensor:
        """Drop a singleton channel dimension if present; pass through (B,*,*,*) otherwise."""
        if x.dim() == 3:
            return x
        if x.dim() == 4:
            if x.shape[1] == 1:
                return x[:, 0]
            return x
        if x.dim() == 5 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError("Expected u_hat / r_mag of shape [B,…] with optional channel dim 1.")


class TinyMLP(nn.Module):
    """1×1 Conv2d head: features [B,F,Nt,Nx] -> C field [B,Nt,Nx]."""

    def __init__(self, in_features: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class TinyMLP3d(nn.Module):
    """1×1×1 Conv3d head: features [B,F,Nx,Ny,Nt] -> C field [B,Nx,Ny,Nt]."""

    def __init__(self, in_features: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_features, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
