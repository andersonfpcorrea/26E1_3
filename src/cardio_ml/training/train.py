"""
Single model training CLI.

This module is the basic entry point: given a model name (and optionally a
dimensionality reduction technique), it runs the complete pipeline, logs the
run to MLflow and prints a summary.

Usage:
    python -m cardio_ml.training.train --model random_forest --dim pca
    cardio-train --model xgboost --dim lda

The script `scripts/run_full_experiment.py` orchestrates batch execution (5
models x 3 dim configurations), but this CLI is useful for quick iterations.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature

from cardio_ml.data.ingestion import TARGET_COLUMN, feature_columns, load_raw_dataset
from cardio_ml.data.split import stratified_train_test_split
from cardio_ml.evaluation.metrics import (
    compute_business_metrics,
    compute_technical_metrics,
    flatten_metrics,
)
from cardio_ml.features.dimensionality import DimTechnique
from cardio_ml.models.registry import MODEL_REGISTRY, build_model_spec
from cardio_ml.models.tuning import TunedResult, tune_with_cv
from cardio_ml.tracking.mlflow_utils import log_run_artifacts, start_run_context


@dataclass
class TrainOutput:
    """Friendly summary of what happened during a training run."""

    model: str
    dim_technique: str
    cv_score: float
    test_metrics: dict[str, float]
    mlflow_run_id: str


def train_model(
    model_name: str,
    dim_technique: DimTechnique = "none",
    scoring: str = "f1",
    test_size: float = 0.2,
) -> TrainOutput:
    """Run the training pipeline for a model and log to MLflow.

    Returns a `TrainOutput` for programmatic use.
    """

    spec = build_model_spec(model_name)
    dataset = load_raw_dataset()
    split = stratified_train_test_split(
        frame=dataset.frame,
        feature_cols=feature_columns(),
        target_col=TARGET_COLUMN,
        test_size=test_size,
    )

    run_name = _format_run_name(model_name, dim_technique)
    tags = {
        "model_family": model_name,
        "dim_technique": dim_technique,
        "scoring": scoring,
    }

    with start_run_context(run_name=run_name, tags=tags) as run:
        t0 = time.perf_counter()
        result = tune_with_cv(
            spec=spec,
            X=split.X_train,
            y=split.y_train,
            dim_technique=dim_technique,
            scoring=scoring,
        )
        fit_seconds = time.perf_counter() - t0

        metrics_dict, test_metrics = _evaluate(result, split.X_test, split.y_test)
        metrics_dict["train.fit_seconds"] = fit_seconds
        metrics_dict["cv.mean"] = result.cv_mean
        metrics_dict["cv.std"] = result.cv_std

        params = {
            "model": model_name,
            "dim_technique": dim_technique,
            "dim_output": result.dim_output,
            "scoring": scoring,
            "train_size": split.sizes["train"],
            "test_size": split.sizes["test"],
            **{k: _stringify(v) for k, v in result.best_params.items()},
        }

        log_run_artifacts(params=params, metrics=metrics_dict)
        _log_model(result, split.X_train)

        run_id = run.info.run_id

    return TrainOutput(
        model=model_name,
        dim_technique=dim_technique,
        cv_score=result.cv_mean,
        test_metrics=test_metrics,
        mlflow_run_id=run_id,
    )


def _evaluate(
    result: TunedResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run predictions on the test set and compute technical + business metrics."""

    pipeline = result.pipeline
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    tech = compute_technical_metrics(y_test, y_pred, y_proba)
    biz = compute_business_metrics(y_test, y_pred)
    flat = flatten_metrics(tech, biz)
    return flat, flat


def _log_model(result: TunedResult, X_sample: pd.DataFrame) -> None:
    """Log the pipeline as a model artifact in MLflow.

    The signature allows the API to automatically validate the request format
    at inference time.
    """

    signature = infer_signature(
        X_sample.head(5),
        result.pipeline.predict(X_sample.head(5)),
    )
    mlflow.sklearn.log_model(
        sk_model=result.pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_sample.head(2),
    )


def _format_run_name(model: str, dim: str) -> str:
    suffix = "raw" if dim == "none" else dim
    return f"{model}__{suffix}"


def _stringify(value) -> str | float | int:
    """Convert non-primitive values to string for logging in MLflow params."""

    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina um modelo e loga no MLflow.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="logistic_regression",
        help="Nome do modelo registrado.",
    )
    parser.add_argument(
        "--dim",
        choices=["none", "pca", "lda"],
        default="none",
        help="Tecnica de reducao de dimensionalidade.",
    )
    parser.add_argument(
        "--scoring",
        default="f1",
        help="Metrica usada na validacao cruzada.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output = train_model(
        model_name=args.model,
        dim_technique=args.dim,
        scoring=args.scoring,
    )

    print("\n=== Resultado do treino ===")
    print(f"Modelo            : {output.model}")
    print(f"Reducao de dim.   : {output.dim_technique}")
    print(f"CV ({args.scoring:<8})     : {output.cv_score:.4f}")
    print(f"Test F1           : {output.test_metrics.get('tech.f1', float('nan')):.4f}")
    print(f"Test ROC-AUC      : {output.test_metrics.get('tech.roc_auc', float('nan')):.4f}")
    print(f"MLflow run_id     : {output.mlflow_run_id}")


if __name__ == "__main__":
    main()
