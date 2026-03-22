# Calibrated Physics-Informed Uncertainty Quantification

A framework for providing calibrated, physics-informed uncertainty estimates for neural PDE solvers using conformal prediction. This approach leverages physics residual errors as a nonconformity score within a conformal prediction framework to enable data-free, model-agnostic, and statistically guaranteed uncertainty estimates. 

## Key Features

- Physics Residual Error (PRE) as a nonconformity score for Conformal Prediction
- Data-free uncertainty quantification
- Model-agnostic implementation
- Marginal and Joint coverage guarantees
- Efficient gradient estimation using convolutional kernels

## Repository Structure

```
├── Active_Learning/       # Active learning experiments
├── Expts_initial/         # Initial experiments
├── Joint/                 # Joint conformal prediction implementation
├── Marginal/              # Marginal conformal prediction implementation
├── Neural_PDE/            # Neural PDE solver implementations
├── Physics_Informed/      # Physics-informed components
├── Tests/                 # Test suite
├── Utils/                 # Utility functions
├── Other_UQ/              # Bayesian Deep Learning experiments
├── pacrc/                 # PACRC: solution-space bounds on top of CP-PRE
├── stability/             # C(x) estimators (constant / Jacobian FD / learned)
├── solution_bound/        # Split conformal quantile + SolutionBoundMapper
├── crc/                   # Selective prediction helper (CRC-style thresholding)
├── benchmarks/            # Quick synthetic tables vs paper-style metrics
└── examples/              # Runnable PACRC end-to-end demo
```

## PACRC extension (Physics-Aware Conformal Risk Control)

This fork adds a **label-free calibration** path that reuses CP-PRE nonconformity scores \(|PRE(\hat u)|\), then maps them to **solution-space** intervals \(\|\hat u - u^\*\| \le C(x)\,q_\alpha\) using `StabilityEstimator` + `SolutionBoundMapper` (`solution_bound/split_cp.py` implements the finite-sample split conformal quantile).

**Minimal baseline FNO + PACRC (synthetic 1D advection, no Neural_PDE / no pretrained):**

```bash
# Quick run (preset fast — default)
PYTHONPATH=.:Utils python3 experiments/train_minimal_fno_advection_pacrc.py

# Heavier “full” training (larger grid, more trajectories, 400 epochs, wider FNO)
PYTHONPATH=.:Utils python3 experiments/train_minimal_fno_advection_pacrc.py --preset full --C-global 1.0
```

Override any preset field explicitly, e.g. `--preset full --epochs 600 --n-train 800`.

**Run a full demo (synthetic Wave + NS, no Neural_PDE):**

```bash
PYTHONPATH=. python3 examples/pacrc_end_to_end.py
```

**Quick comparison table (PRE vs PACRC-style diagnostics):**

```bash
PYTHONPATH=. python3 benchmarks/quick_cp_pre_pacrc_table.py
```

**Python API:**

```python
from pacrc.pipeline import PACRCMarginalPipeline
from pacrc.integrations.burgers_1d import make_burgers_residual
# calibrate on |r_cal|, then bound_field(u_hat, |r|) on test grids
```

**Hooking into Marginal scripts:** see `pacrc/hooks/marginal_wave_snippet.py` for a copy–paste pattern after `q_alpha` / `pred_residual` in `Marginal/Wave_Residuals_CP.py`.

**Run Phase 0/1/2 pipeline directly:**

```bash
# Phase 0: environment + optional asset check
PYTHONPATH=.:Utils python3 experiments/phase0_asset_check.py

# Phase 1: train once on synthetic advection, then sweep C values (CSV output)
PYTHONPATH=.:Utils python3 experiments/phase1_advection_c_sweep.py --preset full --C-grid 0.5,1,2,5,10,20

# Phase 2 (runnable now): Wave/NS synthetic benchmark table (CSV output)
PYTHONPATH=.:Utils python3 experiments/phase2_wave_ns_synthetic.py --replicates 3
```

## Usage

### Basic Example of PRE estimation using Convolutions. 

Within this code base for the paper, we release a utility function that constructs convolutional layers for gradient estimation based on your choice of order of differentiation and Taylor approximation. This allows for the PRE score function to be easily expressed in a single line of code. 
This section provides an overview of the code implementation and algorithm for estimating the PRE using Convolution operations. We'll use an arbitrary PDE example with a temporal gradient and a Laplacian to illustrate the process. 

$$ \frac{\partial u}{\partial t} - \alpha\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + \beta u = 0$$ 

```python
from ConvOps_2d import ConvOperator

# Define operators for PDE
D_t = ConvOperator(domain='t', order=1)  # time-derivative
D_xx_yy = ConvOperator(domain=('x','y'), order=2)  # Laplacian
D_identity = ConvOperator()  # Identity Operator
```

The **ConvOperator** class is used to set up a gradient operation. It takes in **variable(s)** of differentiation and **order of differentiation** as arguments to design the appropriate forward difference stencil and then sets up a convolutional layer with the stencil as the kernel. Under the hood, the class will take care of devising a 3D convolutional layer, and setup the kernel so that it acts on a spatio-temporal tensor of dimensionality: [BS, Nt, Nx, Ny] which expands to batch size, temporal discretisation and the spatial discretisation in $x$ and $y$. 

```python
# Combine operators
alpha, beta = 1.0, 0.5 #coefficients
D = ConvOperator()
D.kernel = D_t.kernel - alpha * D_xx_yy.kernel - beta * D_identity.kernel
```
The convolutional kernels are additive i.e. in order to estimate the residual in one convolutional operation, they could be added together to form a composite kernel that characterises the entire PDE residual. 
Once having set up the kernels, PRE estimation is as simple as passing the composite class instance $D$ the predictions from the neural PDE surrogate (ensuring that the output is in the same order as the kernel outlined above). 


```python
# Estimate PRE
y_pred = model(X)
PRE = D(y_pred)
```

Only operating on the outputs, this method of PRE estimation is memory efficient, computationally cheap and with the **ConvOperator** evaluating the PDE residual can be done in a single line of code. 

### Explore

Standalone Reproduceable experiments (Does not need any data or pretrained Models) : 

```bash
python -m Marginal/Advection_Residuals_CP.py # Run 1D advection experiment to obtain Marginal Bounds
python -m Joint/Advection_Residuals_CP.py    # Run 1D advection experiment to obtain Joint Bounds
```

In order to run the other experiments, you will need the **Neural_PDE** package (FNO, training utilities, `inductive_cp`, numerical solvers), **dataset files** (`.npz`), and **pretrained weights** (`.pth` + optional `_norms.npz`). The original sentence “downloaded from here” had **no working hyperlink** in the upstream README, and the companion repo linked in the code ([`Neural_PDE` on GitHub](https://github.com/gitvicky/Neural_PDE)) currently returns **404** (private, renamed, or removed).

**What to do in practice**

1. **Ask the authors** for the artifact bundle or the current clone URL: **v.gopakumar@ucl.ac.uk** (see Contact below), or open an issue on [gitvicky/CP-PRE](https://github.com/gitvicky/CP-PRE). ICML 2025 code/data links are sometimes added to the paper page or project site after camera-ready.
2. **Layout expected by the scripts** (typical): put `Neural_PDE` **next to** this repo so that from `Marginal/` the path resolves to `../Neural_PDE/Data/…` (e.g. `Burgers_1d.npz`, `Spectral_Wave_data_LHS.npz`, `NS_Spectral_combined.npz`). Weights are loaded from `Marginal/Weights/` or `../Weights/` depending on the script (e.g. `FNO_Burgers_worn-insulation.pth`, `FNO_Wave_cyclic-muntin.pth`).
3. **If you obtain only `Neural_PDE` source**: you can **generate data** by running the solvers under `Neural_PDE/Numerical_Solvers/` (as the README suggests), then **train** FNO yourself instead of using released checkpoints.

Until those assets are available, use the **standalone** experiments (`Marginal/Advection_Residuals_CP.py`, `Joint/Advection_Residuals_CP.py`) or the **PACRC synthetic demos** in `examples/` and `benchmarks/` in this fork.


## Experiments

The repository includes experiments over the following PDEs:

1. 1D Advection Equation
2. 1D Burgers' Equation   
3. 2D Wave Equation
4. 2D Navier-Stokes Equations
5. 2D Magnetohydrodynamics (MHD)

## Benchmarking
The methdology is benchmarked against several Bayesian Deep Learning Methods: 

1. MC Dropout
2. Deep Ensembles 
3. Bayesian Neural Networks
4. Stochastic Weighted Avergaing - Gaussian

## Requirements

- **Python 3.11** (recommended for this fork)
- Install dependencies: `pip install -r requirements.txt`
- Core: PyTorch, NumPy, SciPy, Matplotlib, tqdm; optional scripts also use **PyYAML**, **pyDOE** (LHS sampling).
- For GPU builds, install the matching `torch` wheel from [pytorch.org](https://pytorch.org/get-started/locally/) if the default pip `torch` is not suitable.


## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gopakumar2025calibratedphysicsinformeduncertaintyquantification,
      title={Calibrated Physics-Informed Uncertainty Quantification}, 
      author={Vignesh Gopakumar and Ander Gray and Lorenzo Zanisi and Timothy Nunn and Stanislas Pamela and Daniel Giles and Matt J. Kusner and Marc Peter Deisenroth},
      year={2025},
      eprint={2502.04406},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.04406}, 
}
```

## License

MIT License

## Contributors

- Vignesh Gopakumar
- Ander Gray
- Lorenzo Zanisi
- Stanislas Pamela
- Dan Giles
- Matt J. Kusner
- Marc Peter Deisenroth

## Contact

For questions and feedback, please contact v.gopakumar@ucl.ac.uk
