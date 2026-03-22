"""
PACRC: Physics-Aware Conformal Risk Control layer on top of CP-PRE.

Public entry points:
  - PACRCMarginalPipeline: calibrate q_α on |PRE|, solution bounds C(x)·q
  - PACRCCRC: optional selective prediction via surrogate risk threshold
  - metrics: MSE, residual / solution coverage, correlation diagnostics
  - integrations: burgers_1d, pre_wave_ns (PRE operators)
"""

from pacrc.crc_layer import PACRCCRC, default_surrogate_from_bound
from pacrc.metrics import (
    correlation_bound_error,
    mean_bound_width,
    mse,
    residual_marginal_coverage,
    solution_pointwise_coverage,
    summarize_run,
)
from pacrc.pipeline import PACRCMarginalPipeline, make_learned_head

__all__ = [
    "PACRCMarginalPipeline",
    "make_learned_head",
    "PACRCCRC",
    "default_surrogate_from_bound",
    "mse",
    "residual_marginal_coverage",
    "solution_pointwise_coverage",
    "mean_bound_width",
    "correlation_bound_error",
    "summarize_run",
]
