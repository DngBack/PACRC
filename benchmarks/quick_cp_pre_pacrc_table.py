#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick CP-PRE vs PACRC metrics on synthetic fields (same PRE operators as the paper stack).

Why synthetic: this repo clone often lacks Neural_PDE + downloaded Data/Weights, so the
full Table 3/4 (Wave / Navier–Stokes from the CP-PRE paper) cannot be reproduced here
without those assets. This script still reports:

  * MSE between surrogate û and reference u*
  * CP-PRE: marginal coverage of |PRE(û)| using split conformal (target ~95% at alpha=0.05)
  * PACRC-style: coverage of ||û−u*|| ≤ C·q with scalar C (constant or oracle from cal — see below)
  * Mean bound width (PACRC) and evaluation time

Paper tables (Calibrated Physics-Informed UQ) use real FNO checkpoints and npz data; paste
those published numbers next to your run for a conceptual comparison. Once
Neural_PDE/Data and Marginal/Weights are available, point this script’s loaders to them
(or run Marginal/Wave_Residuals_CP.py) and swap the synthetic block for decoded predictions.

Usage (from repository root):

  PYTHONPATH=. python3 benchmarks/quick_cp_pre_pacrc_table.py
  PYTHONPATH=. python3 benchmarks/quick_cp_pre_pacrc_table.py --replicates 5 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ConvOps_* import fft_conv_pytorch as a top-level package living under Utils/
_UTILS = os.path.join(ROOT, "Utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

from Other_UQ.Evaluation.PRE_estimations import PRE_NS, PRE_Wave
from solution_bound.split_cp import split_conformal_quantile


def _smooth_field(shape: tuple, rng: np.random.Generator, amp: float = 1.0) -> np.ndarray:
    """Low-frequency random field for (B, Nx, Ny, Nt)."""
    B, Nx, Ny, Nt = shape
    out = np.zeros(shape, dtype=np.float32)
    for b in range(B):
        coeff = (
            rng.standard_normal((6, 6, 4))
            * (rng.random((6, 6, 4)) + 0.2) ** (-2)
            * float(amp)
        )
        xs = np.linspace(0, 1, Nx)
        ys = np.linspace(0, 1, Ny)
        ts = np.linspace(0, 1, Nt)
        X, Y, T = np.meshgrid(xs, ys, ts, indexing="ij")
        acc = np.zeros((Nx, Ny, Nt), dtype=np.float32)
        for i in range(6):
            for j in range(6):
                for k in range(4):
                    acc += coeff[i, j, k] * np.sin((i + 1) * np.pi * X) * np.sin((j + 1) * np.pi * Y) * np.cos(
                        (k + 1) * np.pi * T
                    )
        out[b] = acc.astype(np.float32)
    return out


def marginal_residual_coverage(
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Fraction of test scores with |r| <= q (symmetric marginal interval)."""
    q = split_conformal_quantile(np.abs(scores_cal), alpha)
    cov = float((np.abs(scores_test) <= q).mean())
    return cov, q


def marginal_solution_coverage(
    err_mag_test: np.ndarray,
    q: float,
    C: float,
) -> tuple[float, float]:
    """Fraction of test points with err <= C * q; mean bound width = C * q."""
    bound = C * q
    cov = float((err_mag_test <= bound).mean())
    return cov, bound


def run_wave_synthetic(
    B: int,
    Nx: int,
    Ny: int,
    Nt: int,
    sigma: float,
    rng: np.random.Generator,
    alpha: float,
    scale: float,
) -> dict:
    dx = 2.0 / max(Nx - 1, 1)
    dt = 1.0 / max(Nt - 1, 1)
    pre = PRE_Wave(torch.tensor(dt), torch.tensor(dx), c=1.0)

    u_star = _smooth_field((B, Nx, Ny, Nt), rng, amp=scale)
    noise = rng.standard_normal((B, Nx, Ny, Nt)).astype(np.float32) * sigma
    u_hat = u_star + noise

    u_t = torch.from_numpy(u_hat).unsqueeze(1)
    with torch.no_grad():
        r = pre.residual(u_t, boundary=False).numpy()
    e = u_hat - u_star
    e_in = e[:, 1:-1, 1:-1, 1:-1]

    n_cal = B // 2
    err_flat = np.abs(e_in).reshape(B, -1)
    r_flat = r.reshape(B, -1)
    r_cal_f, r_test_f = r_flat[:n_cal].ravel(), r_flat[n_cal:].ravel()
    err_cal_f, err_test_f = err_flat[:n_cal].ravel(), err_flat[n_cal:].ravel()

    mse = float(np.mean((u_hat - u_star) ** 2))
    cov_pre, q = marginal_residual_coverage(r_cal_f, r_test_f, alpha)
    med_abs_e = float(np.median(np.abs(err_test_f)))
    q_over_mede = q / (med_abs_e + 1e-12)

    ratios = err_cal_f / (np.abs(r_cal_f) + 1e-8)
    C_oracle = float(np.quantile(ratios, 1.0 - alpha))
    cov_pacrc_oracle, width_o = marginal_solution_coverage(err_test_f, q, C_oracle)

    C_one = 1.0
    cov_pacrc_c1, width_1 = marginal_solution_coverage(err_test_f, q, C_one)

    return {
        "case": "Wave (synthetic)",
        "mse": mse,
        "cp_pre_residual_cov": cov_pre,
        "cp_pre_q": q,
        "pacrc_cov_C1": cov_pacrc_c1,
        "pacrc_cov_oracleC": cov_pacrc_oracle,
        "pacrc_mean_bound_width_C1": width_1,
        "pacrc_mean_bound_width_oracleC": width_o,
        "C_oracle": C_oracle,
        "median_abs_err_test": med_abs_e,
        "q_over_median_abs_err": q_over_mede,
    }


def run_ns_synthetic(
    B: int,
    Nx: int,
    Ny: int,
    Nt: int,
    sigma: float,
    rng: np.random.Generator,
    alpha: float,
    scale: float,
) -> dict:
    dx = 1.0 / max(Nx - 1, 1)
    dy = dx
    dt = 0.5 / max(Nt - 1, 1)
    pre = PRE_NS(torch.tensor(dt), torch.tensor(dx), torch.tensor(dy))

    u_star = np.stack(
        [
            _smooth_field((B, Nx, Ny, Nt), rng, amp=scale),
            _smooth_field((B, Nx, Ny, Nt), rng, amp=scale),
            _smooth_field((B, Nx, Ny, Nt), rng, amp=scale),
        ],
        axis=1,
    )
    noise = rng.standard_normal(u_star.shape).astype(np.float32) * sigma
    u_hat = u_star + noise

    v_t = torch.from_numpy(u_hat)
    t0 = time.perf_counter()
    with torch.no_grad():
        r = pre.residual(v_t, boundary=False).numpy()
    t_eval = time.perf_counter() - t0

    eu = u_hat[:, 0] - u_star[:, 0]
    ev = u_hat[:, 1] - u_star[:, 1]
    ep = u_hat[:, 2] - u_star[:, 2]
    err_l2 = np.sqrt(eu**2 + ev**2 + ep**2)
    err_in = err_l2[:, 1:-1, 1:-1, 1:-1]

    n_cal = B // 2
    r_cal_f = r[:n_cal].ravel()
    r_test_f = r[n_cal:].ravel()
    err_flat = err_in.reshape(B, -1)
    err_cal_f = err_flat[:n_cal].ravel()
    err_test_f = err_flat[n_cal:].ravel()

    mse = float(np.mean((u_hat - u_star) ** 2))
    cov_pre, q = marginal_residual_coverage(r_cal_f, r_test_f, alpha)
    med_abs_e = float(np.median(err_test_f))
    q_over_mede = q / (med_abs_e + 1e-12)

    ratios = err_cal_f / (np.abs(r_cal_f) + 1e-8)
    C_oracle = float(np.quantile(ratios, 1.0 - alpha))
    cov_pacrc_oracle, width_o = marginal_solution_coverage(err_test_f, q, C_oracle)
    cov_pacrc_c1, width_1 = marginal_solution_coverage(err_test_f, q, 1.0)

    return {
        "case": "Navier–Stokes (synthetic, 3 ch.)",
        "mse": mse,
        "cp_pre_residual_cov": cov_pre,
        "cp_pre_q": q,
        "pacrc_cov_C1": cov_pacrc_c1,
        "pacrc_cov_oracleC": cov_pacrc_oracle,
        "pacrc_mean_bound_width_C1": width_1,
        "pacrc_mean_bound_width_oracleC": width_o,
        "C_oracle": C_oracle,
        "eval_s_pre_only": t_eval,
        "median_abs_err_test": med_abs_e,
        "q_over_median_abs_err": q_over_mede,
    }


def format_row(d: dict) -> str:
    return (
        f"{d['case']:<38} | MSE={d['mse']:.4e} | PRE cov@95%={100*d['cp_pre_residual_cov']:.2f}% | "
        f"PACRC cov (C=1)={100*d['pacrc_cov_C1']:.2f}% | PACRC cov (C_oracle)={100*d['pacrc_cov_oracleC']:.2f}% | "
        f"width C=1={d['pacrc_mean_bound_width_C1']:.4e} | width C_oracle={d['pacrc_mean_bound_width_oracleC']:.4e} | "
        f"q/median|e|={d['q_over_median_abs_err']:.2f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=3, help="Random seeds 0..replicates-1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05, help="Miscoverage; coverage target = 1-alpha")
    p.add_argument("--B", type=int, default=256, help="Total samples (split half cal / half test)")
    p.add_argument(
        "--sigma",
        type=float,
        default=0.02,
        help="Noise std on synthetic fields (field amplitude is O(1); tune with --scale)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.15,
        help="Overall scale of smooth reference field u* (smaller => smaller PRE, tighter q)",
    )
    args = p.parse_args()

    alpha = args.alpha
    paper_cp_pre = (
        "\nReference (paper, CP-PRE @ 95% — paste beside your run):\n"
        "  Wave:   In-dist cov ≈ 95.52% | OOD cov ≈ 95.39% | Eval ≈ 32 s\n"
        "  NS:     In-dist cov ≈ 95.44% | OOD cov ≈ 95.57% | Eval ≈ 134 s\n"
    )

    print("PACRC / CP-PRE quick benchmark (synthetic fields, same PRE operators as Other_UQ)\n")
    print(paper_cp_pre)

    for case_fn, nx, ny, nt in (
        (run_wave_synthetic, 24, 24, 12),
        (run_ns_synthetic, 20, 20, 10),
    ):
        rows = []
        t_wall = 0.0
        for k in range(args.replicates):
            rng = np.random.default_rng(args.seed + k)
            t0 = time.perf_counter()
            rows.append(case_fn(args.B, nx, ny, nt, args.sigma, rng, alpha, args.scale))
            t_wall += time.perf_counter() - t0

        def mean_std(key: str):
            v = np.array([r[key] for r in rows], dtype=np.float64)
            return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0

        m_mse, s_mse = mean_std("mse")
        m_pc, s_pc = mean_std("cp_pre_residual_cov")
        m_p1, s_p1 = mean_std("pacrc_cov_C1")
        m_po, s_po = mean_std("pacrc_cov_oracleC")

        name = rows[0]["case"]
        m_ratio, s_ratio = mean_std("q_over_median_abs_err")
        print(
            f"\n=== {name} (grid {nx}x{ny}x{nt}, B={args.B}, sigma={args.sigma}, scale={args.scale}) ==="
        )
        print(f"Replicates: {args.replicates} | wall time (total): {t_wall:.3f}s")
        print(
            f"MSE: {m_mse:.4e} ± {s_mse:.4e}\n"
            f"CP-PRE marginal |PRE| coverage @ {100*(1-alpha):.0f}% target: {100*m_pc:.2f}% ± {100*s_pc:.2f}%\n"
            f"PACRC solution coverage with C=1: {100*m_p1:.2f}% ± {100*s_p1:.2f}%\n"
            f"PACRC solution coverage with C_oracle (label on cal, invalid deploy): "
            f"{100*m_po:.2f}% ± {100*s_po:.2f}%\n"
            f"Diagnostic: q / median|e| on test ≈ {m_ratio:.2f} ± {s_ratio:.2f} "
            f"(≫1 ⇒ residual band very loose vs solution error; PACRC needs finite C(x) theory)"
        )
        print("\nPer-replicate:")
        for r in rows:
            print(" ", format_row(r))

    print(
        "\nNotes:\n"
        "  • C_oracle = (1-alpha)-quantile of |e|/|r| on calibration — only for diagnostics / plotting;\n"
        "    it uses u* and is not label-free.\n"
        "  • Paper Table 3–4 numbers are not reproduced here (need Neural_PDE + npz + FNO weights).\n"
        "    Clone https://github.com/gitvicky/Neural_PDE alongside this repo and download assets from the CP-PRE README.\n"
        "  • Then run Marginal/Wave_Residuals_CP.py / Joint/NS_Residuals_CP.py and attach SolutionBoundMapper after q_alpha.\n"
    )


if __name__ == "__main__":
    main()
