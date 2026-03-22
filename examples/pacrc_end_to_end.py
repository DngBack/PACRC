#!/usr/bin/env python3
"""
End-to-end PACRC demo: synthetic Wave + Navier–Stokes fields, PRE scores, split conformal,
solution-space bounds C·q, metrics (MSE, residual coverage, solution coverage, correlation).

Run from repository root:

  PYTHONPATH=. python3 examples/pacrc_end_to_end.py

Requires: torch, numpy; adds Utils/ to path for fft_conv_pytorch (vendored).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Utils"))

import numpy as np
import torch

from pacrc.pipeline import PACRCMarginalPipeline
from pacrc.synthetic_datasets import ns_split, wave_split


def main():
    rng = np.random.default_rng(0)
    alpha = 0.05

    print("=== Wave (synthetic) ===")
    w = wave_split(B=96, nx=20, ny=20, nt=10, sigma=0.02, scale=0.12, rng=rng)
    pipe_w = PACRCMarginalPipeline(alpha=alpha, stability_mode="constant", C_global=0.02)
    m_w = pipe_w.evaluate(
        w["u_hat_cal"],
        w["u_star_cal"],
        w["r_cal"],
        w["u_hat_test"],
        w["u_star_test"],
        w["r_test"],
    )
    for k, v in m_w.items():
        print(f"  {k}: {v}")

    print("\n=== Navier–Stokes (synthetic, 3 channels) ===")
    n = ns_split(B=64, nx=16, ny=16, nt=8, sigma=0.02, scale=0.12, rng=rng)
    pipe_n = PACRCMarginalPipeline(alpha=alpha, stability_mode="constant", C_global=0.15)
    m_n = pipe_n.evaluate(
        n["u_hat_cal"],
        n["u_star_cal"],
        n["r_cal"],
        n["u_hat_test"],
        n["u_star_test"],
        n["r_test"],
        abs_err_test=n["err_mag_test"],
        abs_err_cal=n["err_mag_cal"],
    )
    for k, v in m_n.items():
        print(f"  {k}: {v}")

    print(
        "\nNext: with real FNO outputs, replace synthetic dicts by decoded predictions and"
        "\nPRE residuals from Marginal/Wave_Residuals_CP.py or Joint/NS_Residuals_CP.py."
    )


if __name__ == "__main__":
    main()
