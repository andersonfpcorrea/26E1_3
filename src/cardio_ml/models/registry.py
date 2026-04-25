"""
Candidate model registry.

Each model is described by a `ModelSpec` that carries:
  - the estimator instantiated with reasonable defaults;
  - the hyperparameter grid to be explored;
  - the search strategy (`grid` or `random`) and the number of iterations in
    the case of RandomizedSearch.

Including the search strategy together with the model is pragmatic: lightweight
models (logistic, tree) support GridSearch; heavy ensembles (RF, XGBoost) use
RandomizedSearch with a fixed budget, allowing computational cost control
aligned with the resource policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from cardio_ml.config import N_JOBS, SEED

SearchStrategy = Literal["grid", "random", "none"]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable contract of a candidate model."""

    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    search_strategy: SearchStrategy
    n_iter: int = 0  # relevante apenas para RandomizedSearch
    # A flag to distinguish "baseline" models from actual candidates.
    # Baselines enter the experiment for comparison but are not optimized.
    is_baseline: bool = False


# Prefix "clf__" because in the final Pipeline the classifier step is named "clf".
# This way the grids are consumed directly by GridSearchCV without transformations.
_DUMMY = ModelSpec(
    name="dummy",
    estimator=DummyClassifier(strategy="stratified", random_state=SEED),
    param_grid={},
    search_strategy="none",
    is_baseline=True,
)

_LOGREG = ModelSpec(
    name="logistic_regression",
    estimator=LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
        n_jobs=N_JOBS,
    ),
    param_grid={
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    },
    search_strategy="grid",
)

_TREE = ModelSpec(
    name="decision_tree",
    estimator=DecisionTreeClassifier(random_state=SEED),
    param_grid={
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    },
    search_strategy="grid",
)

_RF = ModelSpec(
    name="random_forest",
    estimator=RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
    param_grid={
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [5, 10, 20, None],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
    search_strategy="random",
    n_iter=12,
)

_XGB = ModelSpec(
    name="xgboost",
    estimator=XGBClassifier(
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="logloss",
        tree_method="hist",
    ),
    param_grid={
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    },
    search_strategy="random",
    n_iter=15,
)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec for spec in (_DUMMY, _LOGREG, _TREE, _RF, _XGB)
}


def build_model_spec(name: str) -> ModelSpec:
    """Retrieve a ModelSpec by name (useful for CLI)."""

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Modelo {name!r} nao registrado. Disponiveis: {available}")
    return MODEL_REGISTRY[name]
