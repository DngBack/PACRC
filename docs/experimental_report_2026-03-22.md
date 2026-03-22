# PACRC Experimental Report (2026-03-22)

## Metadata

- **Date**: 2026-03-22
- **Project**: PACRC on top of CP-PRE
- **Environment**: Linux, Python 3.11, CUDA available
- **Executed by**: `admin1` (owner) + AI coding assistant (GPT-5.3-codex)
- **Goal**: Produce runnable Phase 0/1/2 pipeline and collect first experimental results for writing draft sections.

## Scope and What Was Run

This run focused on:

1. **Phase 0**: environment/asset readiness checks.
2. **Phase 1**: minimal 1D FNO training on synthetic advection, then PACRC sweep over constant `C`.
3. **Phase 2**: synthetic Wave/NS benchmark with PACRC for tabular comparison.

### Commands executed

```bash
PYTHONPATH=.:Utils python3 experiments/phase0_asset_check.py
PYTHONPATH=.:Utils python3 experiments/phase1_advection_c_sweep.py --preset full --C-grid 0.5,1,2,5,10,20
PYTHONPATH=.:Utils python3 experiments/phase2_wave_ns_synthetic.py --replicates 3
```

### Scripts used

- `experiments/phase0_asset_check.py`
- `experiments/phase1_advection_c_sweep.py`
- `experiments/phase2_wave_ns_synthetic.py`
- Core PACRC pipeline:
  - `pacrc/pipeline.py`
  - `pacrc/metrics.py`
  - `pacrc/synthetic_datasets.py`

## Phase 0: Readiness Check

### Outcome

- **Required checks**: passed.
- **Optional real-data assets**: missing (expected in this run):
  - `../Neural_PDE/`
  - `../Neural_PDE/Data/*.npz` (Burgers/Wave/NS)
  - pretrained weights under `Marginal/Weights/` and `Weights/`

### Interpretation

- The project is ready for **synthetic runs** and PACRC prototype validation.
- Real CP-PRE benchmark reproduction is **not yet possible** without external data/weights.

## Phase 1 Results (Synthetic Advection, Full Preset)

### Training configuration

- `preset=full`
- `nx=128`, `nt=96`, `T_in=20`
- `n_train=400`, `n_cal=200`, `n_test=200`
- `epochs=400`, `modes=16`, `width=64`
- `alpha=0.05`
- `C-grid = [0.5, 1, 2, 5, 10, 20]`

### Core metrics

- **Full-grid test MSE**: `7.081646e-04`
- **Residual coverage (CP-PRE space)**: stable at `0.962138` across all `C`
- **q_alpha**: `3.066450e-03` (constant across all `C`, as expected)

### PACRC trend over C

| C | Solution coverage | Mean bound width |
|---:|---:|---:|
| 0.5 | 0.3091 | 1.533e-03 |
| 1 | 0.3869 | 3.066e-03 |
| 2 | 0.5010 | 6.133e-03 |
| 5 | 0.7072 | 1.533e-02 |
| 10 | 0.8644 | 3.066e-02 |
| 20 | 0.9590 | 6.133e-02 |

### Interpretation

- Behavior is **internally consistent**:
  - increasing `C` increases bound width and solution coverage.
  - residual-space conformal coverage remains unchanged by `C`.
- `corr(|error|, bound)` is `nan` because bound is spatially constant (`C * q_alpha`) in this sweep, so correlation is not informative.

## Phase 2 Results (Synthetic Wave/NS, 3 Replicates)

### Wave (synthetic)

- Residual coverage: `0.9425 ± 0.0045` (near target 0.95)
- Solution coverage: `1.0000 ± 0.0000` for all tested `C`
- Mean width increases linearly with `C`

### NS (synthetic)

- Residual coverage: `0.9456 ± 0.0126` (near target 0.95)
- Solution coverage improves monotonically with `C`:
  - `C=0.02`: `0.0065 ± 0.0010`
  - `C=0.05`: `0.0912 ± 0.0130`
  - `C=0.10`: `0.4624 ± 0.0458`
  - `C=0.20`: `0.9643 ± 0.0153`
  - `C>=0.5`: `1.0000 ± 0.0000`

### Interpretation

- Phase 2 confirms expected monotonic trade-off in NS.
- Wave synthetic setup is currently too easy (solution coverage saturates at 1.0), so it is less discriminative for method comparison.

## Artifacts Produced

- `experiments/results_phase1_advection_c_sweep.csv`
- `experiments/results_phase2_wave_ns_synthetic.csv`

These CSV files are suitable as raw data sources for plotting and writing preliminary results sections.

## Assessment Against ICML-Oriented Evidence Quality

### What is already strong

- End-to-end PACRC pipeline runs reproducibly.
- Split-conformal behavior in residual space is correct/near-target.
- PACRC trade-off (`C` vs solution coverage/width) is clearly visible.

### What is not yet enough for final ICML claims

- Results are still synthetic-first.
- `C` is constant; no adaptive `C(x)` ablation yet.
- No real CP-PRE benchmark data/weights integration (Wave/NS real setting).

## Next Steps (Execution Plan)

### Step 1 (Immediate, runnable now)

1. Extend Phase 1 with additional sweeps:
   - seeds: 3-5 seeds
   - `C-grid`: denser around transition region (e.g., `1,2,3,4,5,7,10,15,20`)
2. Generate figures:
   - solution coverage vs mean width
   - solution coverage vs `C`

### Step 2 (Methodic upgrade)

1. Add `jacobian_fd` option into Phase 1 script for 1D:
   - compare `constant` vs `jacobian_fd`
   - measure overhead (runtime) + quality (coverage/width)
2. Add optional learned `C` prototype on synthetic split (strict train/cal separation).

### Step 3 (Benchmark-grade evidence)

1. Acquire real assets (`Neural_PDE`, `Data`, `Weights`) and run analogous table on real Wave/NS.
2. Report:
   - residual coverage
   - solution coverage
   - bound width
   - runtime
   - ablation by `C` strategy

## Short Conclusion

The current run is a **successful prototype milestone**: PACRC mechanics are validated and produce coherent trends.  
For publication-grade evidence (especially ICML-level), the next critical move is adaptive `C(x)` ablation and real Wave/NS benchmark integration.
