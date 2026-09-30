"""Validation tests for the PSAR reproducibility project."""

import csv
import random
import subprocess
import sys

import numpy as np

from src.generation import generate_Block


def test_weight_matrix_is_row_normalized():
    """Check structural properties of the generated SAR weight matrix."""
    np.random.seed(123)
    random.seed(123)

    n = 40

    # Small random graphs can contain isolated nodes. The original
    # generator computes inverse degrees and therefore emits a
    # divide-by-zero warning for zero-degree nodes. Suppressing that
    # warning here does not change the generated matrix.
    with np.errstate(divide="ignore"):
        W = generate_Block(n)

    assert W.shape == (n, n)

    # No self-loops.
    assert np.allclose(W.diagonal(), 0.0)

    # Network weights must be nonnegative.
    assert np.all(W.data >= 0.0)

    # Row normalization implies that every non-isolated row sums to 1.
    row_sums = np.asarray(W.sum(axis=1)).reshape(-1)
    nonisolated = row_sums > 0

    assert np.any(nonisolated)
    assert np.allclose(row_sums[nonisolated], 1.0)


def test_small_simulation_pipeline(tmp_path):
    """Check that a small end-to-end simulation produces valid output."""
    output_file = tmp_path / "simulation_summary.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.simulation",
            "--n",
            "40",
            "--replications",
            "1",
            "--seed",
            "123",
            "--network",
            "SBM",
            "--output",
            str(output_file),
        ],
        check=True,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    with output_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # The simulation should report all three model parameters.
    assert {row["parameter"] for row in rows} == {
        "beta1",
        "beta2",
        "rho",
    }

    for row in rows:
        assert int(row["N"]) == 40
        assert int(row["replications"]) == 1

        assert float(row["bias"]) >= 0.0
        assert float(row["avg_est_se"]) >= 0.0
        assert float(row["mse"]) >= 0.0

        coverage = float(row["coverage"])
        assert 0.0 <= coverage <= 1.0
