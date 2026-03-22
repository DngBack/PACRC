#!/usr/bin/env python3
"""
Lightest path to: train a baseline FNO (1D) on synthetic advection → CP-PRE (|PRE|) → PACRC bounds.

No Neural_PDE, no pretrained weights, small CPU-friendly defaults.

Usage (from repo root):

  PYTHONPATH=. python3 experiments/train_minimal_fno_advection_pacrc.py
  PYTHONPATH=. python3 experiments/train_minimal_fno_advection_pacrc.py --epochs 80 --n-train 40

Also add Utils/ for ConvOperator:

  PYTHONPATH=/path/to/PACRC:/path/to/PACRC/Utils python3 experiments/train_minimal_fno_advection_pacrc.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_UTILS = os.path.join(ROOT, "Utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

import torch
import torch.nn.functional as F
from experiments.advection_data import coord_grid, lowpass_ic, rollout_advection
from experiments.minimal_fno1d import FNO1dMini
from pacrc.pipeline import PACRCMarginalPipeline
from Utils.ConvOps_1d import ConvOperator


def make_advection_residual(nx: int, device, dtype=torch.float32):
    dev = str(device)
    dx = torch.tensor(1.0 / nx, device=device, dtype=dtype)
    dt = dx.clone()
    v = torch.tensor(1.0, device=device, dtype=dtype)
    D_t = ConvOperator(domain="t", order=1, device=dev)
    D_x = ConvOperator(domain="x", order=1, device=dev)
    D = ConvOperator()
    D.kernel = (D_t.kernel + (v * dt / dx) * D_x.kernel).to(device=device, dtype=dtype)

    def residual(u_b_t_x: torch.Tensor) -> torch.Tensor:
        """u: (B, Nt, Nx) — interior slice."""
        r = D(u_b_t_x)
        return r[..., 1:-1, 1:-1]

    return residual


@torch.no_grad()
def autoregressive_rollout(model: FNO1dMini, ic: torch.Tensor, nt: int, T_in: int) -> torch.Tensor:
    """ic: (B, nx); returns u (B, nx, nt) using true history for first T_in then model."""
    b, nx = ic.shape
    device, dtype = ic.device, ic.dtype
    u = torch.empty(b, nx, nt, device=device, dtype=dtype)
    u[:, :, :T_in] = rollout_advection(ic, T_in)[:, :, :T_in]
    xc = coord_grid(nx, b, device, dtype)
    for t in range(T_in, nt):
        inp = u[:, :, t - T_in : t]
        pred = model(inp, xc).squeeze(-1)
        u[:, :, t] = pred
    return u


def train(
    model: FNO1dMini,
    u: torch.Tensor,
    T_in: int,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
):
    b, nx, nt = u.shape
    model.train()
    for ep in range(epochs):
        total = 0.0
        count = 0
        for t0 in range(T_in, nt):
            inp = u[:, :, t0 - T_in : t0]
            tgt = u[:, :, t0]
            xc = coord_grid(nx, b, device, u.dtype)
            pred = model(inp, xc).squeeze(-1)
            loss = F.mse_loss(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs}  train_mse_per_step {total/count:.6e}")


def main():
    p = argparse.ArgumentParser(
        description="Train minimal 1D FNO on synthetic advection + CP-PRE + PACRC. "
        "Use --preset full for a heavier run (more data, larger net, longer training)."
    )
    p.add_argument(
        "--preset",
        choices=("fast", "full"),
        default="fast",
        help="fast: small grid & net (default). full: nx=128, nt=96, T_in=20, 400 epochs, "
        "400/200/200 trajectories, width=64, modes=16. Any explicit flag overrides preset.",
    )
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--nt", type=int, default=None, help="time steps per trajectory")
    p.add_argument("--T-in", dest="T_in", type=int, default=None)
    p.add_argument("--n-train", dest="n_train", type=int, default=None)
    p.add_argument("--n-cal", dest="n_cal", type=int, default=None)
    p.add_argument("--n-test", dest="n_test", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--modes", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument(
        "--ic-modes",
        dest="ic_modes",
        type=int,
        default=None,
        help="low-pass modes in random IC (richer PDE diversity)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--C-global", dest="C_global", type=float, default=0.5, help="PACRC constant C")
    args = p.parse_args()

    PRESET_FAST = dict(
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
    PRESET_FULL = dict(
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
    base = PRESET_FULL if args.preset == "full" else PRESET_FAST
    cfg = dict(base)
    overrides = dict(
        nx=args.nx,
        nt=args.nt,
        T_in=args.T_in,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        epochs=args.epochs,
        lr=args.lr,
        modes=args.modes,
        width=args.width,
        ic_modes=args.ic_modes,
    )
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    T_in = cfg["T_in"]

    n_tot = cfg["n_train"] + cfg["n_cal"] + cfg["n_test"]
    ic = lowpass_ic(n_tot, cfg["nx"], n_modes=cfg["ic_modes"], rng=rng, device=device)
    u_all = rollout_advection(ic, cfg["nt"])

    u_train = u_all[: cfg["n_train"]]
    u_cal = u_all[cfg["n_train"] : cfg["n_train"] + cfg["n_cal"]]
    u_test = u_all[cfg["n_train"] + cfg["n_cal"] :]

    model = FNO1dMini(T_in=T_in, width=cfg["width"], modes=cfg["modes"]).to(device)
    print(
        f"preset={args.preset} | nx={cfg['nx']} nt={cfg['nt']} T_in={T_in} | "
        f"train/cal/test={cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} | epochs={cfg['epochs']}"
    )
    print(f"FNO1dMini params: {model.n_params():,} | device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    print("Training on synthetic advection (teacher forcing, one-step MSE)...")
    train(model, u_train, T_in, opt, cfg["epochs"], device)

    model.eval()
    residual_fn = make_advection_residual(cfg["nx"], device)

    with torch.no_grad():
        u_hat_cal = autoregressive_rollout(model, u_cal[:, :, 0], cfg["nt"], T_in).permute(0, 2, 1)
        u_star_cal = u_cal.permute(0, 2, 1)
        r_cal = torch.abs(residual_fn(u_hat_cal))

        u_hat_test = autoregressive_rollout(model, u_test[:, :, 0], cfg["nt"], T_in).permute(0, 2, 1)
        u_star_test = u_test.permute(0, 2, 1)
        r_test = torch.abs(residual_fn(u_hat_test))

    r_mag_cal = r_cal
    r_mag_test = r_test

    uhi_cal = u_hat_cal[:, 1:-1, 1:-1]
    uhi_star = u_star_cal[:, 1:-1, 1:-1]
    uhi_test = u_hat_test[:, 1:-1, 1:-1]
    uhi_star_test = u_star_test[:, 1:-1, 1:-1]

    pipe = PACRCMarginalPipeline(alpha=args.alpha, stability_mode="constant", C_global=args.C_global)
    metrics = pipe.evaluate(
        uhi_cal,
        uhi_star,
        r_mag_cal,
        uhi_test,
        uhi_star_test,
        r_mag_test,
    )

    mse_baseline = float(torch.mean((u_hat_test - u_star_test) ** 2).item())

    print("\n========== Results ==========")
    print(f"Test MSE (full grid autoregressive): {mse_baseline:.6e}")
    print(f"CP-PRE marginal |PRE| coverage (target {100*(1-args.alpha):.0f}%): "
          f"{100*metrics['residual_marginal_coverage_test']:.2f}%")
    print(
        f"PACRC solution pointwise coverage (constant C={args.C_global}): "
        f"{100*metrics['solution_pointwise_coverage_test']:.2f}%"
    )
    print(f"Mean bound width: {metrics['mean_bound_width_test']:.6e}")
    print(f"q_alpha: {metrics['q_alpha']:.6e}")
    print(f"corr(|error|, bound): {metrics['corr_abs_err_bound_test']}")
    print("\nTip: tune --C-global or switch to jacobian_fd on this 1D grid in PACRCMarginalPipeline.")


if __name__ == "__main__":
    main()
