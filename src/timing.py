"""Compare CLE and CLS computation time across sample sizes."""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .CLE import lce_estimate_x
from .CLS import cls_estimate_x
from .generation import generate


def run_timing(
    n_values,
    repeats=5,
    seed=123,
    type_w="Dyad",
    lambda2=0.5,
    lambdax=1.0,
):
    """Return average CLE and CLS running times for each sample size."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    np.random.seed(seed)
    random.seed(seed)

    p = 2
    p1 = 1
    p2 = 1
    beta = np.array([0.3, 0.3]).reshape((p, 1))
    rho = 0.2
    rows = []

    for N in n_values:
        cle_times = []
        cls_times = []

        for r in range(repeats):
            X, Y1, W, _ = generate(N, p, beta, rho, type_w, "N", "N", lambda2)
            xe = np.random.normal(0, lambdax**0.5, N * p2).reshape(N, p2)
            X[:, p1:p] = X[:, p1:p] + xe

            start = time.perf_counter()
            lce_estimate_x(X, Y1, W, p1, lambda2, lambdax)
            cle_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            cls_estimate_x(X, Y1, W, p1, lambda2, lambdax)
            cls_times.append(time.perf_counter() - start)

            print(
                f"\rTiming N={N}: {(r + 1) / repeats * 100:5.1f}%",
                end="",
            )
        print()

        rows.append(
            {
                "N": int(N),
                "network": type_w,
                "repeats": repeats,
                "CLE_mean_seconds": float(np.mean(cle_times)),
                "CLS_mean_seconds": float(np.mean(cls_times)),
            }
        )
    return rows


def write_csv(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timing(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x = [row["N"] for row in rows]
    cle = [row["CLE_mean_seconds"] for row in rows]
    cls = [row["CLS_mean_seconds"] for row in rows]

    plt.figure(figsize=(7, 6))
    plt.plot(x, cle, label="CLE", linestyle="-")
    plt.plot(x, cls, label="CLS", linestyle=":")
    plt.xlabel("N")
    plt.ylabel("CPU time (seconds)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--network", default="Dyad", choices=["PowerLaw", "Dyad", "SBM"])
    parser.add_argument("--table", default="results/tables/timing.csv")
    parser.add_argument("--figure", default="results/figures/timing.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_timing(
        args.n_values,
        repeats=args.repeats,
        seed=args.seed,
        type_w=args.network,
    )
    write_csv(rows, args.table)
    plot_timing(rows, args.figure)
    print(f"Wrote timing table to {args.table}")
    print(f"Wrote timing figure to {args.figure}")


if __name__ == "__main__":
    main()
