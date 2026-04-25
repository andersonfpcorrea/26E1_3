"""Data layer tests (ingestion, quality, split)."""

from __future__ import annotations

import pandas as pd
import pytest

from cardio_ml.data.ingestion import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    NUMERIC_FEATURES,
    _augment_features,
    _validate_schema,
)
from cardio_ml.data.quality import diagnose_quality
from cardio_ml.data.split import stratified_train_test_split


def test_schema_validation_rejects_missing_columns():
    frame = pd.DataFrame({col: [1] for col in EXPECTED_COLUMNS[:-1]})  # without target
    with pytest.raises(ValueError):
        _validate_schema(frame)


def test_augment_features_adds_age_and_bmi():
    frame = pd.DataFrame(
        {
            "id": [1],
            "age": [20000],
            "gender": [1],
            "height": [170],
            "weight": [70],
            "ap_hi": [120],
            "ap_lo": [80],
            "cholesterol": [1],
            "gluc": [1],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "cardio": [0],
        }
    )
    enriched = _augment_features(frame)
    assert "age_years" in enriched.columns
    assert "bmi" in enriched.columns
    assert abs(enriched.loc[0, "age_years"] - 54.8) < 0.2


def test_quality_report_detects_inverted_pressure():
    frame = pd.DataFrame(
        {
            "age_years": [55.0],
            "height": [170],
            "weight": [70],
            "ap_hi": [80],
            "ap_lo": [120],  # inverted!
            "bmi": [24.2],
            "gender": [1],
            "cholesterol": [1],
            "gluc": [1],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "cardio": [0],
        }
    )
    report = diagnose_quality(frame)
    assert any(i.column == "ap_hi/ap_lo" for i in report.issues)


def test_stratified_split_preserves_class_ratio(synthetic_frame):
    feature_cols = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    split = stratified_train_test_split(
        frame=synthetic_frame,
        feature_cols=feature_cols,
        target_col="cardio",
        test_size=0.25,
    )
    overall = synthetic_frame["cardio"].mean()
    train_ratio = split.y_train.mean()
    test_ratio = split.y_test.mean()
    assert abs(train_ratio - overall) < 0.05
    assert abs(test_ratio - overall) < 0.05
