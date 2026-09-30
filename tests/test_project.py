"""Validation tests for the PSAR reproducibility project."""

import subprocess
import sys

import numpy as np

from src.generation import generate_covariates


def test_generate_covariates_respects_paper_bound():
    """Generated covariates should have the expected shape and be bounded by 2."""
    np.random.seed(123)

    n = 200
    p = 2

    x = generate_covariates(n, p)

    assert x.shape == (n, p)
    assert np.all(np.isfinite(x))
    assert np.all(np.abs(x) <= 2.0)


def test_simulation_pipeline_creates_summary(tmp_path):
    """A small simulation should create a nonempty summary CSV."""
    output_file = tmp_path / "simulation_summary.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.simulation",
            "--n",
            "30",
            "--replications",
            "1",
            "--seed",
            "123",
            "--network",
            "PowerLaw",
            "--output",
            str(output_file),
        ],
        check=True,
    )

    assert output_file.exists()

    lines = output_file.read_text().strip().splitlines()

    assert len(lines) > 1

    header = lines[0]

    assert "parameter" in header
    assert "bias" in header
    assert "coverage" in header
