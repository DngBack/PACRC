#!/usr/bin/env python3
"""
Phase 0 checks before running PACRC experiments.

Checks:
1) Core Python modules and PACRC imports.
2) Local repository layout.
3) Optional upstream assets for real CP-PRE runs (Neural_PDE data + weights).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _ok(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: PACRC environment + asset checks")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path",
    )
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="Fail when optional real-data assets are missing",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    root = args.root.resolve()
    utils = root / "Utils"
    parent_neural = root.parent / "Neural_PDE"

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(utils) not in sys.path:
        sys.path.insert(0, str(utils))

    required_modules = [
        "torch",
        "numpy",
        "pacrc.pipeline",
        "pacrc.synthetic_datasets",
        "experiments.minimal_fno1d",
        "experiments.advection_data",
    ]
    required_paths = [
        root / "pacrc",
        root / "experiments",
        root / "Utils",
        root / "solution_bound",
        root / "stability",
        root / "Marginal",
    ]
    optional_paths = [
        parent_neural,
        parent_neural / "Data" / "Burgers_1d.npz",
        parent_neural / "Data" / "Spectral_Wave_data_LHS.npz",
        parent_neural / "Data" / "NS_Spectral_combined.npz",
        root / "Marginal" / "Weights" / "FNO_Burgers_worn-insulation.pth",
        root / "Marginal" / "Weights" / "FNO_Wave_cyclic-muntin.pth",
        root / "Weights" / "FNO_Navier-Stokes_violent-remote.pth",
    ]

    mod_result: dict[str, bool] = {}
    for mod in required_modules:
        try:
            importlib.import_module(mod)
            mod_result[mod] = True
        except Exception:
            mod_result[mod] = False

    req_result = {str(p): p.exists() for p in required_paths}
    opt_result = {str(p): p.exists() for p in optional_paths}

    required_ok = all(mod_result.values()) and all(req_result.values())
    optional_ok = all(opt_result.values())
    overall_ok = required_ok and (optional_ok if args.strict_optional else True)

    report = {
        "repo_root": str(root),
        "required_ok": required_ok,
        "optional_ok": optional_ok,
        "overall_ok": overall_ok,
        "required_modules": mod_result,
        "required_paths": req_result,
        "optional_paths": opt_result,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[{_ok(required_ok)}] Required checks")
        for k, v in mod_result.items():
            print(f"  [{_ok(v)}] import {k}")
        for k, v in req_result.items():
            print(f"  [{_ok(v)}] path   {k}")
        print(f"\n[{_ok(optional_ok)}] Optional real-data assets (strict={args.strict_optional})")
        for k, v in opt_result.items():
            print(f"  [{_ok(v)}] path   {k}")
        print(f"\nOverall: {_ok(overall_ok)}")
        if not optional_ok:
            print("Note: optional assets are only needed for real CP-PRE benchmark scripts.")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
