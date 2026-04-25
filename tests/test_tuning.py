"""Tuning module tests (fast — without full Grid/Random search)."""

from __future__ import annotations

import pytest

from cardio_ml.models.registry import MODEL_REGISTRY
from cardio_ml.models.tuning import tune_with_cv


@pytest.mark.parametrize("dim_technique", ["none", "pca", "lda"])
def test_tune_dummy_runs_without_search(synthetic_frame, feature_cols, dim_technique):
    """Dummy has search_strategy='none' and should finish quickly in any config."""

    X = synthetic_frame[feature_cols]
    y = synthetic_frame["cardio"]

    result = tune_with_cv(
        spec=MODEL_REGISTRY["dummy"],
        X=X,
        y=y,
        dim_technique=dim_technique,
        cv_splits=2,
    )
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.pipeline.named_steps["clf"] is not None


def test_tune_logistic_regression_small_grid(synthetic_frame, feature_cols):
    """LogReg with reduced grid to keep the test fast."""

    spec = MODEL_REGISTRY["logistic_regression"]
    # Reduce the grid to 1 combination to speed up the test.
    object.__setattr__(
        spec,
        "param_grid",
        {"clf__C": [1.0], "clf__class_weight": [None]},
    )

    X = synthetic_frame[feature_cols]
    y = synthetic_frame["cardio"]

    result = tune_with_cv(spec=spec, X=X, y=y, cv_splits=2)
    assert result.cv_mean > 0.0
    assert "clf__C" in result.best_params
