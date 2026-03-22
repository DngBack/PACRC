"""
Copy–paste integration pattern for Marginal/Wave_Residuals_CP.py (after q_alpha / pred_residual).

Assumes:
  - pred_pred, pred_out decoded; residual() defined as in the Marginal script;
  - Inner-grid tensors align with residual(..., boundary=False).

Example (conceptual; adjust permute/slice to match your script):

```python
import sys
sys.path.insert(0, "<PACRC_REPO_ROOT>")
from pacrc.pipeline import PACRCMarginalPipeline
import torch
import numpy as np

# Existing CP-PRE (physics-driven scores on calibration):
# ncf_scores = np.abs(res.numpy())   # cal split
alpha = 0.1
pipe = PACRCMarginalPipeline(alpha=alpha, stability_mode="constant", C_global=1.0)
pipe.calibrate_from_scores(ncf_scores)

preds = pred_pred.permute(0, 1, 4, 2, 3)[:, 0]   # (B, Nt, Nx, Ny) — match your file
outv  = pred_out.permute(0, 1, 4, 2, 3)[:, 0]
r_pred = residual(preds)
r_mag = torch.tensor(np.abs(r_pred.numpy()))  # or r_pred.abs()
# Crop preds/outv to same inner grid as r_mag
sl = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
bound = pipe.bound_field(preds[sl], r_mag)

# Evaluation only (needs ground truth):
e = (preds[sl] - outv[sl]).abs()
solution_cov = (e <= bound).float().mean().item()
```
"""
