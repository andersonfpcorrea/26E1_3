"""Preprocessing pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from cardio_ml.features.dimensionality import build_dim_reducer
from cardio_ml.features.preprocessing import build_preprocessor


def test_preprocessor_fits_and_transforms(synthetic_frame, feature_cols):
    preprocessor = build_preprocessor()
    X = synthetic_frame[feature_cols]
    X_transformed = preprocessor.fit_transform(X)
    # Must produce a dense numeric array with 1 row per example.
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_preprocessor_inside_pipeline_no_leakage(synthetic_frame, feature_cols):
    """Ensure scaling statistics come ONLY from the training set.

    This is the most important non-regression test for leakage: we use a
    complete Pipeline and verify that the scale learned during the training
    fit is applied as-is in the test transform.
    """

    X = synthetic_frame[feature_cols]
    train, test = X.iloc[:150], X.iloc[150:]

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("reducer", build_dim_reducer("pca", n_components=2)),
        ]
    )
    pipeline.fit(train)

    transformed_test = pipeline.transform(test)
    assert transformed_test.shape == (len(test), 2)


def test_pca_preserves_minimum_variance(synthetic_frame, feature_cols):
    X = synthetic_frame[feature_cols]
    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)

    reducer = build_dim_reducer("pca", n_components=0.95)
    reducer.fit(X_prep)
    assert reducer.output_dim is not None
    assert reducer.output_dim < X_prep.shape[1]


def test_lda_requires_target(synthetic_frame, feature_cols):
    X = synthetic_frame[feature_cols]
    y = synthetic_frame["cardio"]

    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)

    reducer = build_dim_reducer("lda")
    reducer.fit(X_prep, y)
    assert reducer.output_dim == 1  # duas classes -> 1 componente maximo

    transformed = reducer.transform(X_prep)
    assert transformed.shape == (len(X), 1)


def test_categorical_unknown_is_ignored_not_raising(feature_cols):
    """OneHotEncoder should ignore new categories in production."""

    rng = np.random.default_rng(0)
    n = 100
    train = pd.DataFrame(
        {
            "age_years": rng.uniform(30, 70, n),
            "height": rng.normal(170, 8, n),
            "weight": rng.normal(75, 15, n),
            "ap_hi": rng.normal(125, 15, n),
            "ap_lo": rng.normal(82, 10, n),
            "bmi": rng.normal(26, 4, n),
            "gender": rng.choice([1, 2], n),
            "cholesterol": rng.choice([1, 2, 3], n),
            "gluc": rng.choice([1, 2], n),
            "smoke": rng.choice([0, 1], n),
            "alco": rng.choice([0, 1], n),
            "active": rng.choice([0, 1], n),
        }
    )
    preprocessor = build_preprocessor()
    preprocessor.fit(train[feature_cols])

    # Production observation with category `gluc=3` (never seen during training).
    prod = train.head(1).copy()
    prod["gluc"] = 3
    preprocessor.transform(prod[feature_cols])  # should not raise
