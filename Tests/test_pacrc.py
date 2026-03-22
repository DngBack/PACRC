#!/usr/bin/env python3
"""Smoke tests for PACRC modules (no Neural_PDE dependency)."""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crc.selector import CRCSelector
from pacrc.pipeline import PACRCMarginalPipeline
from solution_bound.mapper import SolutionBoundMapper
from solution_bound.split_cp import split_conformal_quantile
from stability.C_estimator import StabilityEstimator, TinyMLP


def test_split_conformal_quantile_monotone():
    rng = np.random.default_rng(0)
    s = rng.random(200)
    q09 = split_conformal_quantile(s, 0.1)
    q05 = split_conformal_quantile(s, 0.05)
    assert q05 >= q09


def test_mapper_constant_bound():
    est = StabilityEstimator(mode="constant", C_global=2.0)
    mapper = SolutionBoundMapper(est, alpha=0.1)
    r = torch.abs(torch.randn(4, 8, 8))
    mapper.calibrate_from_residual_scores(r.numpy())
    u = torch.randn(4, 8, 8)
    b = mapper.bound_field(u, r)
    assert b.shape == r.shape
    assert torch.allclose(b, torch.full_like(r, 2.0 * mapper.q_alpha))


def test_jacobian_fd_identity_slice():
    def res(u):
        return u[:, 1:-1, 1:-1]

    u = torch.randn(1, 10, 12, requires_grad=False)
    r = res(u)
    est = StabilityEstimator(mode="jacobian_fd", fd_eps=1e-3)
    C = est.jacobian_fd_local(u, res, r.shape)
    assert C.shape == r.shape
    assert (C > 0).all()


def test_learned_head_positive():
    est = StabilityEstimator(mode="learned", g_phi=TinyMLP(in_features=2, hidden=8))
    u = torch.randn(2, 16, 16)
    r = torch.randn(2, 16, 16)
    C = est(u, r)
    assert C.shape == r.shape
    assert (C > 0).all()


def test_crc_selector_tau():
    risks = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sel = CRCSelector(surrogate_risk_fn=lambda **kwargs: torch.tensor(0.0))
    tau = sel.calibrate_tau(risks, epsilon=0.35)
    assert tau <= 0.5


def test_pacrc_pipeline_evaluate_1d_inner():
    def res(u):
        return u[:, 1:-1, 1:-1]

    B = 32
    u_star = torch.randn(B, 8, 10)
    u_hat = u_star + 0.01 * torch.randn_like(u_star)
    r = res(u_hat)
    n = B // 2
    pipe = PACRCMarginalPipeline(alpha=0.1, stability_mode="constant", C_global=0.5)
    m = pipe.evaluate(
        u_hat[:n, 1:-1, 1:-1],
        u_star[:n, 1:-1, 1:-1],
        r[:n],
        u_hat[n:, 1:-1, 1:-1],
        u_star[n:, 1:-1, 1:-1],
        r[n:],
    )
    assert "solution_pointwise_coverage_test" in m
    assert "residual_marginal_coverage_test" in m
