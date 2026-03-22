"""
Synthetic fields + PRE residuals for demos when Neural_PDE data is unavailable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from Other_UQ.Evaluation.PRE_estimations import PRE_NS, PRE_Wave


def _smooth_field(shape: tuple, rng: np.random.Generator, amp: float) -> np.ndarray:
    B, Nx, Ny, Nt = shape
    out = np.zeros(shape, dtype=np.float32)
    xs = np.linspace(0, 1, Nx)
    ys = np.linspace(0, 1, Ny)
    ts = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(xs, ys, ts, indexing="ij")
    for b in range(B):
        coeff = (
            rng.standard_normal((6, 6, 4))
            * (rng.random((6, 6, 4)) + 0.2) ** (-2)
            * float(amp)
        )
        acc = np.zeros((Nx, Ny, Nt), dtype=np.float32)
        for i in range(6):
            for j in range(6):
                for k in range(4):
                    acc += coeff[i, j, k] * np.sin((i + 1) * np.pi * X) * np.sin((j + 1) * np.pi * Y) * np.cos(
                        (k + 1) * np.pi * T
                    )
        out[b] = acc
    return out


def wave_split(
    B: int,
    nx: int,
    ny: int,
    nt: int,
    sigma: float,
    scale: float,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    dx = 2.0 / max(nx - 1, 1)
    dt = 1.0 / max(nt - 1, 1)
    pre = PRE_Wave(torch.tensor(dt), torch.tensor(dx), c=1.0)
    u_star = _smooth_field((B, nx, ny, nt), rng, scale)
    noise = rng.standard_normal((B, nx, ny, nt)).astype(np.float32) * sigma
    u_hat = u_star + noise
    u_t = torch.from_numpy(u_hat).unsqueeze(1)
    with torch.no_grad():
        r = pre.residual(u_t, boundary=False)
    u_star_t = torch.from_numpy(u_star)
    e = u_hat - u_star
    e_in = torch.from_numpy(e[:, 1:-1, 1:-1, 1:-1])
    u_in = torch.from_numpy(u_hat[:, 1:-1, 1:-1, 1:-1])
    ustar_in = torch.from_numpy(u_star[:, 1:-1, 1:-1, 1:-1])
    n_cal = B // 2
    return {
        "pre": pre,
        "r_full": r,
        "r_mag": r.abs(),
        "u_hat_inner": u_in,
        "u_star_inner": ustar_in,
        "r_cal": r[:n_cal],
        "r_test": r[n_cal:],
        "u_hat_cal": u_in[:n_cal],
        "u_star_cal": ustar_in[:n_cal],
        "u_hat_test": u_in[n_cal:],
        "u_star_test": ustar_in[n_cal:],
    }


def ns_split(
    B: int,
    nx: int,
    ny: int,
    nt: int,
    sigma: float,
    scale: float,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    dx = 1.0 / max(nx - 1, 1)
    dy = dx
    dt = 0.5 / max(nt - 1, 1)
    pre = PRE_NS(torch.tensor(dt), torch.tensor(dx), torch.tensor(dy))
    u_star = np.stack(
        [
            _smooth_field((B, nx, ny, nt), rng, scale),
            _smooth_field((B, nx, ny, nt), rng, scale),
            _smooth_field((B, nx, ny, nt), rng, scale),
        ],
        axis=1,
    )
    noise = rng.standard_normal(u_star.shape).astype(np.float32) * sigma
    u_hat = u_star + noise
    v_t = torch.from_numpy(u_hat)
    with torch.no_grad():
        r = pre.residual(v_t, boundary=False)
    eu = u_hat[:, 0] - u_star[:, 0]
    ev = u_hat[:, 1] - u_star[:, 1]
    ep = u_hat[:, 2] - u_star[:, 2]
    err_l2 = np.sqrt(eu**2 + ev**2 + ep**2)
    err_in = err_l2[:, 1:-1, 1:-1, 1:-1]
    u_in = torch.from_numpy(u_hat[:, :, 1:-1, 1:-1, 1:-1])
    ustar_in = torch.from_numpy(u_star[:, :, 1:-1, 1:-1, 1:-1])
    n_cal = B // 2
    err_t = torch.from_numpy(err_in)
    return {
        "pre": pre,
        "r_mag": r.abs(),
        "r_cal": r[:n_cal],
        "r_test": r[n_cal:],
        "u_hat_cal": u_in[:n_cal],
        "u_star_cal": ustar_in[:n_cal],
        "u_hat_test": u_in[n_cal:],
        "u_star_test": ustar_in[n_cal:],
        "err_mag_test": err_t[n_cal:],
        "err_mag_cal": err_t[:n_cal],
    }
