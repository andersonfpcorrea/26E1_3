"""
Feature preprocessing.

Builds a `ColumnTransformer` that should be wrapped inside a
`sklearn.pipeline.Pipeline`. This architecture is the standard for avoiding
data leakage: normalization and imputation statistics are learned only from
the training split during `fit`, and reapplied on test/production via
`transform`.

Numeric columns -> median imputation + StandardScaler.
Categorical columns -> mode imputation + OneHotEncoder (handle_unknown=ignore).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create the standardized ColumnTransformer for the modeling pipeline.

    Design decisions:
      - `StandardScaler` on numeric columns because linear and distance-based
        models require it, and XGBoost/RandomForest are scale-invariant (no
        harm). Using it always standardizes the pipeline across different models.
      - `OneHotEncoder(handle_unknown='ignore')` prevents never-before-seen
        categories from breaking inference in production.
      - `SimpleImputer` acts as a safety net for data points that may have
        nulls in production, even if the current dataset does not.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
