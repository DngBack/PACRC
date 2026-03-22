#!/usr/bin/env python3
"""
Phase 2 (runnable now): Wave/NS synthetic benchmark with PACRC.

This gives paper-style metrics table without external Neural_PDE assets:
- residual marginal coverage
- solution-space coverage
- mean bound width
- q_alpha

Use as a placeholder table until real Wave/NS assets are available.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
UTILS = os.path.join(ROOT, "Utils")
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

from pacrc.pipeline import PACRCMarginalPipeline
from pacrc.synthetic_datasets import ns_split, wave_split


def _parse_csv_floats(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty --C-grid")
    return vals


def _mean_std(metrics: List[Dict[str, float]], key: str):
    arr = np.asarray([m[key] for m in metrics], dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def run_wave(alpha, c_val, B, nx, ny, nt, sigma, scale, seed):
    rng = np.random.default_rng(seed)
    d = wave_split(B=B, nx=nx, ny=ny, nt=nt, sigma=sigma, scale=scale, rng=rng)
    pipe = PACRCMarginalPipeline(alpha=alpha, stability_mode="constant", C_global=c_val)
    return pipe.evaluate(
        d["u_hat_cal"], d["u_star_cal"], d["r_cal"],
        d["u_hat_test"], d["u_star_test"], d["r_test"],
    )


def run_ns(alpha, c_val, B, nx, ny, nt, sigma, scale, seed):
    rng = np.random.default_rng(seed)
    d = ns_split(B=B, nx=nx, ny=ny, nt=nt, sigma=sigma, scale=scale, rng=rng)
    pipe = PACRCMarginalPipeline(alpha=alpha, stability_mode="constant", C_global=c_val)
    return pipe.evaluate(
        d["u_hat_cal"], d["u_star_cal"], d["r_cal"],
        d["u_hat_test"], d["u_star_test"], d["r_test"],
        abs_err_test=d["err_mag_test"], abs_err_cal=d["err_mag_cal"],
    )


def main():
    p = argparse.ArgumentParser(description="Phase 2: synthetic Wave/NS benchmark")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--C-grid", default="0.02,0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B-wave", type=int, default=96)
    p.add_argument("--B-ns", type=int, default=64)
    p.add_argument("--wave-grid", default="20,20,10", help="nx,ny,nt")
    p.add_argument("--ns-grid", default="16,16,8", help="nx,ny,nt")
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--scale", type=float, default=0.12)
    p.add_argument("--csv-out", default="experiments/results_phase2_wave_ns_synthetic.csv")
    args = p.parse_args()

    c_values = _parse_csv_floats(args.C_grid)
    wx, wy, wt = [int(x) for x in args.wave_grid.split(",")]
    nx, ny, nt = [int(x) for x in args.ns_grid.split(",")]

    rows = []
    print("case\tC\tres_cov(mean±std)\tsol_cov(mean±std)\twidth(mean±std)\tq(mean±std)")
    for c_val in c_values:
        wave_runs = [
            run_wave(args.alpha, c_val, args.B_wave, wx, wy, wt, args.sigma, args.scale, args.seed + k)
            for k in range(args.replicates)
        ]
        ns_runs = [
            run_ns(args.alpha, c_val, args.B_ns, nx, ny, nt, args.sigma, args.scale, args.seed + 1000 + k)
            for k in range(args.replicates)
        ]

        for case, runs in (("wave", wave_runs), ("ns", ns_runs)):
            rc_m, rc_s = _mean_std(runs, "residual_marginal_coverage_test")
            sc_m, sc_s = _mean_std(runs, "solution_pointwise_coverage_test")
            w_m, w_s = _mean_std(runs, "mean_bound_width_test")
            q_m, q_s = _mean_std(runs, "q_alpha")

            rows.append(
                dict(
                    case=case,
                    C_global=c_val,
                    alpha=args.alpha,
                    residual_coverage_mean=rc_m,
                    residual_coverage_std=rc_s,
                    solution_coverage_mean=sc_m,
                    solution_coverage_std=sc_s,
                    mean_bound_width_mean=w_m,
                    mean_bound_width_std=w_s,
                    q_alpha_mean=q_m,
                    q_alpha_std=q_s,
                    replicates=args.replicates,
                )
            )
            print(
                f"{case}\t{c_val:.6g}\t{rc_m:.4f}±{rc_s:.4f}\t"
                f"{sc_m:.4f}±{sc_s:.4f}\t{w_m:.3e}±{w_s:.3e}\t{q_m:.3e}±{q_s:.3e}"
            )

    csv_path = os.path.join(ROOT, args.csv_out) if not os.path.isabs(args.csv_out) else args.csv_out
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
