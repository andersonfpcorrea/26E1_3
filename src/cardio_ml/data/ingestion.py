"""
Data ingestion.

Responsible for loading the raw cardiovascular disease dataset from the canonical
path, applying schema validation and minimal transformations that make sense
before any modeling step (for example, converting age from days to years).
Model-dependent transformations belong in the preprocessing pipeline, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cardio_ml.config import RAW_DATASET_PATH

# Expected columns in the raw CSV. Serves as a minimal schema contract:
# if the dataset changes, the failure happens at ingestion rather than
# silently in the middle of the pipeline.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
)

TARGET_COLUMN: str = "cardio"

# Columns that make up the feature set used by the models. The `id` column
# is discarded because it has no predictive value.
NUMERIC_FEATURES: tuple[str, ...] = (
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "bmi",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
)


@dataclass(frozen=True)
class RawDataset:
    """Encapsulates the dataset with minimal transformations already applied.

    Attributes:
        frame: DataFrame with the expected columns + `age_years` and `bmi`.
        source_path: absolute path of the CSV read (traceability).
    """

    frame: pd.DataFrame
    source_path: Path


def load_raw_dataset(path: Path | str | None = None) -> RawDataset:
    """Load the raw CSV and apply minimal deterministic transformations.

    Parameters:
        path: path to the CSV. When None, uses `RAW_DATASET_PATH` from config.

    Returns:
        RawDataset containing the DataFrame with `age_years` and `bmi`.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if any expected column is missing.
    """

    source = Path(path) if path is not None else RAW_DATASET_PATH
    if not source.exists():
        raise FileNotFoundError(f"Dataset bruto nao encontrado em {source}.")

    # The original dataset uses ';' as separator.
    frame = pd.read_csv(source, sep=";")

    _validate_schema(frame)

    frame = _augment_features(frame)
    return RawDataset(frame=frame, source_path=source.resolve())


def _validate_schema(frame: pd.DataFrame) -> None:
    """Ensure the CSV contains exactly the expected columns."""

    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {sorted(missing)}. "
            f"Esperado: {EXPECTED_COLUMNS}."
        )


def _augment_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable derived columns (age in years and BMI).

    These derivations are deterministic, do not depend on split statistics
    (and therefore do not cause data leakage), and appear at every point in
    the pipeline.
    """

    enriched = frame.copy()
    enriched["age_years"] = (enriched["age"] / 365.25).round(1)

    # BMI in kg/m^2 — height is in cm.
    height_m = enriched["height"] / 100.0
    enriched["bmi"] = (enriched["weight"] / (height_m**2)).round(2)

    return enriched


def feature_columns() -> list[str]:
    """Flat list of columns used as features by the models."""

    return [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
