"""Shared fixtures across tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    """Generate a small DataFrame consistent with the real dataset schema."""

    rng = np.random.default_rng(42)
    n = 200

    frame = pd.DataFrame(
        {
            "age_years": rng.uniform(30, 70, n).round(1),
            "height": rng.normal(170, 8, n).round(0),
            "weight": rng.normal(75, 15, n).round(0),
            "ap_hi": rng.normal(125, 15, n).round(0),
            "ap_lo": rng.normal(82, 10, n).round(0),
            "bmi": rng.normal(26, 4, n).round(2),
            "gender": rng.choice([1, 2], n),
            "cholesterol": rng.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1]),
            "gluc": rng.choice([1, 2, 3], n, p=[0.8, 0.15, 0.05]),
            "smoke": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "alco": rng.choice([0, 1], n, p=[0.95, 0.05]),
            "active": rng.choice([0, 1], n, p=[0.2, 0.8]),
        }
    )
    # Target that reasonably depends on features (so models can learn).
    score = (
        0.04 * (frame["age_years"] - 50)
        + 0.03 * (frame["ap_hi"] - 120)
        + 0.5 * (frame["cholesterol"] - 1)
        + rng.normal(0, 0.5, n)
    )
    frame["cardio"] = (score > 0).astype(int)
    return frame


@pytest.fixture()
def feature_cols() -> list[str]:
    return [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
