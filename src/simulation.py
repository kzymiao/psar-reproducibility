"""Run the tractable Monte Carlo experiment for the PSAR project."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np

from .CLS import cls_estimate_x, cls_inference
from .generation import generate


def summarize_parameter(estimates, true_value, estimated_ses, covered):
    """Summarize one parameter across Monte Carlo replications."""
    estimates = np.asarray(estimates, dtype=float).reshape(-1)
    estimated_ses = np.asarray(estimated_ses, dtype=float).reshape(-1)
    return {
        "bias": float(abs(np.mean(estimates) - true_value)),
        "mc_sd": float(np.std(estimates)),
        "avg_est_se": float(np.mean(estimated_ses)),
        "coverage": float(np.mean(covered)),
        "mse": float(np.mean((estimates - true_value) ** 2)),
    }


def run_simulation(
    N=500,
    R=100,
    seed=123,
    type_w="PowerLaw",
    distri_e="N",
    distri_ep="N",
    lambda2=0.5,
    lambdax=0.5,
):
    """Run the CLS Monte Carlo experiment and return summary rows."""
    if N < 2:
        raise ValueError("N must be at least 2")
    if R < 1:
        raise ValueError("R must be at least 1")

    np.random.seed(seed)
    random.seed(seed)

    p = 2
    p1 = 1
    p2 = 1
    beta = np.array([0.3, 0.3]).reshape((p, 1))
    rho = 0.2
    true_values = np.array([0.3, 0.3, rho])
    parameter_names = ["beta1", "beta2", "rho"]

    estimates = np.zeros((R, p + 1))
    estimated_ses = np.zeros((R, p + 1))
    covered = np.zeros((R, p + 1), dtype=bool)
    densities = np.zeros(R)
    elapsed = np.zeros(R)

    for r in range(R):
        X, Y1, W, mu_ep = generate(
            N, p, beta, rho, type_w, distri_e, distri_ep, lambda2
        )

        # The paper uses normal covariate privacy noise in this experiment.
        xe = np.random.normal(0, lambdax**0.5, N * p2).reshape(N, p2)
        mu_ex = 3 * lambdax * lambdax
        X[:, p1:p] = X[:, p1:p] + xe

        start = time.perf_counter()
        hrbeta, hsig, D, dD, ddD = cls_estimate_x(
            X, Y1, W, p1, lambda2, lambdax
        )
        se = cls_inference(
            X,
            Y1,
            hrbeta,
            hsig,
            lambda2,
            mu_ep,
            W,
            D,
            dD,
            ddD,
            p1,
            lambdax,
            mu_ex,
        )
        elapsed[r] = time.perf_counter() - start

        est = np.asarray(hrbeta[: p + 1]).reshape(-1)
        se_vec = np.asarray(se[: p + 1]).reshape(-1)
        estimates[r, :] = est
        estimated_ses[r, :] = se_vec

        lower = est - 1.96 * se_vec
        upper = est + 1.96 * se_vec
        covered[r, :] = (true_values >= lower) & (true_values <= upper)

        densities[r] = W.nnz / N / (N - 1)
        print(f"\rSimulation progress: {(r + 1) / R * 100:5.1f}%", end="")

    print()

    rows = []
    for j, name in enumerate(parameter_names):
        summary = summarize_parameter(
            estimates[:, j], true_values[j], estimated_ses[:, j], covered[:, j]
        )
        rows.append(
            {
                "method": "CLS",
                "N": N,
                "replications": R,
                "network": type_w,
                "error_distribution": distri_e,
                "privacy_noise_distribution": distri_ep,
                "lambda2": lambda2,
                "lambdax": lambdax,
                "parameter": name,
                "true_value": true_values[j],
                **summary,
                "avg_estimation_time": float(np.mean(elapsed)),
                "avg_network_density": float(np.mean(densities)),
            }
        )
    return rows


def write_rows(rows, output):
    """Write summary rows to a CSV file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--network", default="PowerLaw", choices=["PowerLaw", "Dyad", "SBM"])
    parser.add_argument("--output", default="results/tables/simulation_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_simulation(
        N=args.n,
        R=args.replications,
        seed=args.seed,
        type_w=args.network,
    )
    write_rows(rows, args.output)
    print(f"Wrote simulation summary to {args.output}")


if __name__ == "__main__":
    main()
