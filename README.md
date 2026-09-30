# Privacy-Protected Spatial Autoregressive Models: Reproducible Simulation

This repository reorganizes the original research code accompanying the paper *Privacy-Protected Spatial Autoregressive Models with Applications* into a reproducible project for STATS 607 Unit 1 Project.

## What this project reproduces

The code implements the corrected likelihood estimator (CLE) and corrected least squares estimator (CLS) for the privacy-protected spatial autoregressive (PSAR) model.

For the course project, the designated reproducible outputs are intentionally smaller than the full paper experiments so the workflow is practical to rerun:

- **Monte Carlo summary:** `N = 500`, `R = 100`, normal errors, Power-Law network, CLS estimator.
- **Timing comparison:**
The timing experiment compares CLE and CLS for 1) N = 500, 1000, 1500, 2000; 2) 5 repetitions for each sample size; 3) Dyad network; 4)lambda2 = 0.5; 5)lambdax = 1.0.

The reported CPU time is the average over the five repetitions.
The paper reports a larger timing experiment with sample sizes up to N = 5000 and 100 repetitions. For this course reproducibility project, we restrict the timing experiment to N <= 2000 and use 5 repetitions per sample size so that the complete workflow can be 
reproduced within a practical amount of time.

## Paper-aligned settings

The core parameter values follow the simulation section of the paper where applicable:

- `beta = (0.3, 0.3)`
- `rho = 0.2`
- Monte Carlo privacy noise variances `lambda2 = 0.5` and `lambdax = 0.5`
- timing experiment `lambda2 = 0.5` and `lambdax = 1.0`
- covariates are generated from a standard normal distribution and truncated so every coordinate lies in `[-2, 2]`
- fixed random seeds are used for reproducibility

The original CLE/CLS estimating equations and network-generation implementations are otherwise preserved to keep the scientific code changes minimal.

## Repository structure

```text
.
├── src/
│   ├── CLE.py            # Corrected likelihood estimator
│   ├── CLS.py            # Corrected least-squares estimator
│   ├── generation.py     # Simulation/network data generation
│   ├── simulation.py     # Monte Carlo experiment and summary table
│   └── timing.py         # CLE-vs-CLS timing experiment
├── tests/
│   └── test_project.py
├── results/
│   ├── tables/
│   └── figures/
├── requirements.txt
├── Makefile
└── README.md
```

## Setup from a fresh clone

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the designated results

```bash
make reproduce
```

This creates:

```text
results/tables/simulation_summary.csv
results/tables/timing.csv
results/figures/timing.png
```

The timing step can be slow for large `N` because CLE contains dense matrix inversions. The course version uses 5 timing repetitions per sample size rather than the larger simulation budget used in the paper.

## Run tests

```bash
make test
```

The tests validate two important pieces of the workflow:

1. **Data validation:** the generated SAR network weight matrix has the expected structural properties, including zero diagonal entries, nonnegative weights, and row sums equal to one for non-isolated nodes.
2. **Pipeline integrity:** a small end-to-end simulation runs successfully and produces a valid summary file containing results for `beta1`, `beta2`, and `rho`.

## Run individual components

Monte Carlo simulation only:

```bash
make simulation
```

Timing comparison only:

```bash
make timing
```

Remove generated outputs:

```bash
make clean
```

## Original analysis

The original version of the analysis should be preserved as the first Git commit (optionally tagged `original`). The final submission should report both that original commit hash and the final commit hash.
