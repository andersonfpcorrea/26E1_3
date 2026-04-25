"""Drift module tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardio_ml.evaluation.drift import (
    PSI_MODERATE_THRESHOLD,
    compute_drift,
    compute_psi,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    # PSI of reference against itself should be ~0.
    assert compute_psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(1.5, 1, 5000)  # shift mean by 1.5 sigma
    psi = compute_psi(reference, current)
    assert psi > PSI_MODERATE_THRESHOLD


def test_compute_drift_identifies_affected_columns():
    rng = np.random.default_rng(0)
    n = 2000
    reference = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, n),
            "drifted": rng.normal(0, 1, n),
        }
    )
    current = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, n),
            "drifted": rng.normal(3, 1, n),
        }
    )

    report = compute_drift(reference, current, feature_cols=["stable", "drifted"])
    verdicts = {f.feature: f.verdict for f in report.per_feature}

    assert verdicts["stable"] == "estavel"
    assert verdicts["drifted"] == "drift"
    assert report.has_drift()
