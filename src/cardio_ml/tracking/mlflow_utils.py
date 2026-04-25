"""
Utilities for experiment tracking with MLflow.

Centralizes tracking URI configuration, experiment creation and the
standardization of tags/artifacts registered in each run. Avoids duplication
and ensures all runs are comparable.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from cardio_ml.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    summarize_policy,
)

_initialized = False


def _initialize() -> None:
    """Configure the tracking URI once per process."""

    global _initialized
    if _initialized:
        return
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _initialized = True


def ensure_experiment(name: str | None = None) -> str:
    """Create the experiment if needed and return its name."""

    _initialize()
    exp_name = name or MLFLOW_EXPERIMENT_NAME
    existing = mlflow.get_experiment_by_name(exp_name)
    if existing is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    return exp_name


@contextmanager
def start_run_context(
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run with standardized reproducibility tags.

    Automatically includes:
      - platform and Python version;
      - global seed;
      - resource policy summary (n_jobs, nice, thread env vars).
    """

    _initialize()
    ensure_experiment()

    base_tags = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": str(SEED),
    }
    for key, value in summarize_policy().items():
        base_tags[f"resources.{key}"] = str(value)
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=base_tags, nested=nested) as run:
        yield run


def log_run_artifacts(
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, Path] | None = None,
    dict_artifacts: dict[str, dict] | None = None,
) -> None:
    """Log parameters, metrics and files in an already started run.

    The function is idempotent with respect to names: calling with the same
    names multiple times simply overwrites/adds, per MLflow rules.
    """

    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifacts:
        for name, path in artifacts.items():
            mlflow.log_artifact(str(path), artifact_path=name)
    if dict_artifacts:
        for name, payload in dict_artifacts.items():
            mlflow.log_text(json.dumps(payload, indent=2, default=str), f"{name}.json")
