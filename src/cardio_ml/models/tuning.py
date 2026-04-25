"""
Hyperparameter tuning with cross-validation.

The `tune_with_cv` function encapsulates the logic common to all registry
models:
  - builds the complete Pipeline (preprocess -> dim reducer -> classifier);
  - chooses between GridSearchCV and RandomizedSearchCV according to the
    `ModelSpec`;
  - returns a `TunedResult` with the refitted estimator, the best parameters,
    the mean score and the standard deviation from CV.

This way, the training script remains declarative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from cardio_ml.config import N_JOBS, SEED
from cardio_ml.features.dimensionality import DimReducer, DimTechnique, build_dim_reducer
from cardio_ml.features.preprocessing import build_preprocessor
from cardio_ml.models.registry import ModelSpec


@dataclass(frozen=True)
class TunedResult:
    """Encapsulated result of a tuning run."""

    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_mean: float
    cv_std: float
    scoring: str
    dim_technique: DimTechnique
    dim_output: int | None


def _build_pipeline(
    estimator,
    dim_technique: DimTechnique,
    n_components: float | int | None,
) -> Pipeline:
    """Build the 3-step Pipeline: preprocess -> reducer -> classifier."""

    reducer: DimReducer = build_dim_reducer(
        technique=dim_technique,
        n_components=n_components,
        random_state=SEED,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("reducer", reducer),
            ("clf", estimator),
        ]
    )


def tune_with_cv(
    spec: ModelSpec,
    X,
    y,
    dim_technique: DimTechnique = "none",
    n_components: float | int | None = None,
    scoring: str = "f1",
    cv_splits: int = 5,
) -> TunedResult:
    """Train and tune a model according to the strategy defined in `spec`.

    Parameters:
        spec: candidate model description from the registry.
        X, y: training data (features and target).
        dim_technique: `none`, `pca` or `lda`.
        n_components: number of components (only used if dim_technique != none).
        scoring: metric used for selection. Default F1 due to the clinical cost
            of false negatives and the recommendation from the previous project.
        cv_splits: number of folds. 5 is a good trade-off between estimate
            variance and execution time.

    Returns:
        TunedResult with the pipeline trained on all data.
    """

    pipeline = _build_pipeline(spec.estimator, dim_technique, n_components)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)

    if spec.search_strategy == "none" or not spec.param_grid:
        # Baseline: no search. Evaluate via cross_val_score and retrain on all data.
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=N_JOBS)
        pipeline.fit(X, y)
        return TunedResult(
            pipeline=pipeline,
            best_params={},
            cv_mean=float(np.mean(scores)),
            cv_std=float(np.std(scores)),
            scoring=scoring,
            dim_technique=dim_technique,
            dim_output=_dim_output(pipeline),
        )

    if spec.search_strategy == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            return_train_score=False,
        )
    elif spec.search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spec.param_grid,
            n_iter=spec.n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            random_state=SEED,
            return_train_score=False,
        )
    else:
        raise ValueError(f"Estrategia de busca desconhecida: {spec.search_strategy}")

    search.fit(X, y)
    best = search.best_estimator_
    best_idx = int(search.best_index_)
    return TunedResult(
        pipeline=best,
        best_params=search.best_params_,
        cv_mean=float(search.cv_results_["mean_test_score"][best_idx]),
        cv_std=float(search.cv_results_["std_test_score"][best_idx]),
        scoring=scoring,
        dim_technique=dim_technique,
        dim_output=_dim_output(best),
    )


def _dim_output(pipeline: Pipeline) -> int | None:
    """Extract the output dimension from the reducer, if present."""

    reducer = pipeline.named_steps.get("reducer")
    if isinstance(reducer, DimReducer):
        return reducer.output_dim
    return None
