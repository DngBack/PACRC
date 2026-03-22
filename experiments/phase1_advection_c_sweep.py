#!/usr/bin/env python3
"""
Phase 1: train minimal 1D FNO once, then sweep PACRC C values.

Output:
- Console summary table
- CSV file with metrics for each C in the sweep
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
UTILS = os.path.join(ROOT, "Utils")
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

from experiments.advection_data import coord_grid, lowpass_ic, rollout_advection
from experiments.minimal_fno1d import FNO1dMini
from pacrc.pipeline import PACRCMarginalPipeline
from Utils.ConvOps_1d import ConvOperator


def _parse_csv_floats(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty --C-grid")
    return vals


def make_advection_residual(nx: int, device, dtype=torch.float32):
    dx = torch.tensor(1.0 / nx, device=device, dtype=dtype)
    dt = dx.clone()
    v = torch.tensor(1.0, device=device, dtype=dtype)
    D_t = ConvOperator(domain="t", order=1, device=str(device))
    D_x = ConvOperator(domain="x", order=1, device=str(device))
    D = ConvOperator()
    D.kernel = (D_t.kernel + (v * dt / dx) * D_x.kernel).to(device=device, dtype=dtype)

    def residual(u_b_t_x: torch.Tensor) -> torch.Tensor:
        # u: (B, Nt, Nx), residual returned on inner grid
        return D(u_b_t_x)[..., 1:-1, 1:-1]

    return residual


@torch.no_grad()
def autoreg_rollout(model: FNO1dMini, ic: torch.Tensor, nt: int, T_in: int) -> torch.Tensor:
    b, nx = ic.shape
    u = torch.empty(b, nx, nt, device=ic.device, dtype=ic.dtype)
    u[:, :, :T_in] = rollout_advection(ic, T_in)[:, :, :T_in]
    xc = coord_grid(nx, b, ic.device, ic.dtype)
    for t in range(T_in, nt):
        pred = model(u[:, :, t - T_in : t], xc).squeeze(-1)
        u[:, :, t] = pred
    return u


def train_one_step(model, u_train, T_in, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    b, nx, nt = u_train.shape
    device, dtype = u_train.device, u_train.dtype
    model.train()
    for ep in range(epochs):
        total = 0.0
        steps = 0
        for t in range(T_in, nt):
            inp = u_train[:, :, t - T_in : t]
            tgt = u_train[:, :, t]
            xc = coord_grid(nx, b, device, dtype)
            pred = model(inp, xc).squeeze(-1)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"epoch {ep+1:>4}/{epochs} train_step_mse={total/steps:.6e}")


def main():
    p = argparse.ArgumentParser(description="Phase 1: minimal advection + C sweep")
    p.add_argument("--preset", choices=("fast", "full"), default="fast")
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--nt", type=int, default=None)
    p.add_argument("--T-in", dest="T_in", type=int, default=None)
    p.add_argument("--n-train", dest="n_train", type=int, default=None)
    p.add_argument("--n-cal", dest="n_cal", type=int, default=None)
    p.add_argument("--n-test", dest="n_test", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--modes", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--ic-modes", dest="ic_modes", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--C-grid", default="0.5,1,2,5,10,20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--csv-out",
        default="experiments/results_phase1_advection_c_sweep.csv",
        help="Output CSV path",
    )
    args = p.parse_args()

    preset_fast = dict(
        nx=64,
        nt=48,
        T_in=8,
        n_train=48,
        n_cal=24,
        n_test=24,
        epochs=100,
        lr=1e-3,
        modes=8,
        width=32,
        ic_modes=6,
    )
    preset_full = dict(
        nx=128,
        nt=96,
        T_in=20,
        n_train=400,
        n_cal=200,
        n_test=200,
        epochs=400,
        lr=5e-4,
        modes=16,
        width=64,
        ic_modes=12,
    )
    cfg = dict(preset_full if args.preset == "full" else preset_fast)
    for k in ("nx", "nt", "T_in", "n_train", "n_cal", "n_test", "epochs", "lr", "modes", "width", "ic_modes"):
        val = getattr(args, k)
        if val is not None:
            cfg[k] = val

    c_values = _parse_csv_floats(args.C_grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(args.seed)

    print(
        f"preset={args.preset} nx={cfg['nx']} nt={cfg['nt']} T_in={cfg['T_in']} "
        f"train/cal/test={cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} epochs={cfg['epochs']}"
    )

    n_total = cfg["n_train"] + cfg["n_cal"] + cfg["n_test"]
    ic = lowpass_ic(n_total, cfg["nx"], n_modes=cfg["ic_modes"], rng=rng, device=device)
    u_all = rollout_advection(ic, cfg["nt"])
    u_train = u_all[: cfg["n_train"]]
    u_cal = u_all[cfg["n_train"] : cfg["n_train"] + cfg["n_cal"]]
    u_test = u_all[cfg["n_train"] + cfg["n_cal"] :]

    model = FNO1dMini(T_in=cfg["T_in"], width=cfg["width"], modes=cfg["modes"]).to(device)
    print(f"FNO1dMini params: {model.n_params():,} | device={device}")
    train_one_step(model, u_train, cfg["T_in"], cfg["epochs"], cfg["lr"])

    model.eval()
    residual_fn = make_advection_residual(cfg["nx"], device)

    with torch.no_grad():
        u_hat_cal = autoreg_rollout(model, u_cal[:, :, 0], cfg["nt"], cfg["T_in"]).permute(0, 2, 1)
        u_star_cal = u_cal.permute(0, 2, 1)
        r_cal = residual_fn(u_hat_cal).abs()

        u_hat_test = autoreg_rollout(model, u_test[:, :, 0], cfg["nt"], cfg["T_in"]).permute(0, 2, 1)
        u_star_test = u_test.permute(0, 2, 1)
        r_test = residual_fn(u_hat_test).abs()

    uhi_cal = u_hat_cal[:, 1:-1, 1:-1]
    uhi_star_cal = u_star_cal[:, 1:-1, 1:-1]
    uhi_test = u_hat_test[:, 1:-1, 1:-1]
    uhi_star_test = u_star_test[:, 1:-1, 1:-1]
    mse_full = float(torch.mean((u_hat_test - u_star_test) ** 2).item())

    print(f"\nfull_grid_test_mse={mse_full:.6e}")
    print("C\tq_alpha\tres_cov\tsol_cov\tmean_width\tcorr(|e|,bound)")

    rows = []
    for c in c_values:
        pipe = PACRCMarginalPipeline(alpha=args.alpha, stability_mode="constant", C_global=c)
        m = pipe.evaluate(
            uhi_cal,
            uhi_star_cal,
            r_cal,
            uhi_test,
            uhi_star_test,
            r_test,
        )
        row = dict(
            C_global=c,
            alpha=args.alpha,
            q_alpha=float(m["q_alpha"]),
            residual_coverage=float(m["residual_marginal_coverage_test"]),
            solution_coverage=float(m["solution_pointwise_coverage_test"]),
            mean_bound_width=float(m["mean_bound_width_test"]),
            corr_abs_err_bound=float(m["corr_abs_err_bound_test"]),
            mse_test_full=mse_full,
            nx=cfg["nx"],
            nt=cfg["nt"],
            T_in=cfg["T_in"],
            n_train=cfg["n_train"],
            n_cal=cfg["n_cal"],
            n_test=cfg["n_test"],
            epochs=cfg["epochs"],
            modes=cfg["modes"],
            width=cfg["width"],
            seed=args.seed,
        )
        rows.append(row)
        print(
            f"{c:.6g}\t{row['q_alpha']:.6e}\t{row['residual_coverage']:.6f}\t"
            f"{row['solution_coverage']:.6f}\t{row['mean_bound_width']:.6e}\t{row['corr_abs_err_bound']}"
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
