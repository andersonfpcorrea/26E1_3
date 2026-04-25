"""
Select the champion model from the runs registered in MLflow.

Composite (weighted) criterion:
  - 60% performance: `tech.f1` on test;
  - 25% robustness: -1 * expected cost (`biz.expected_cost_per_case`);
  - 15% efficiency: -1 * normalized training time (`train.fit_seconds`).

The weights are explicit to facilitate auditing. In case of a tie, breaks by
`roc_auc`. The winning model is registered in the MLflow Model Registry and
promoted to the `Production` stage.

Usage:
    python scripts/select_final_model.py
    python scripts/select_final_model.py --experiment cardio-dcv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardio_ml.config import (  # noqa: E402
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

# Selection function weights. Sum to 1.0 in absolute value for clarity.
WEIGHT_PERFORMANCE: float = 0.60
WEIGHT_ROBUSTNESS: float = 0.25
WEIGHT_EFFICIENCY: float = 0.15


@dataclass
class Candidate:
    run_id: str
    name: str
    f1: float
    roc_auc: float
    expected_cost: float
    fit_seconds: float
    score: float


def main() -> None:
    args = _parser().parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    exp_name = args.experiment
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise SystemExit(
            f"Experimento {exp_name!r} nao encontrado. Rode scripts/run_full_experiment.py primeiro."
        )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if runs.empty:
        raise SystemExit("Nenhum run finalizado disponivel para selecao.")

    candidates = _score_candidates(runs)
    if not candidates:
        raise SystemExit("Os runs encontrados nao possuem metricas suficientes para selecao.")

    winner = max(candidates, key=lambda c: (c.score, c.roc_auc))
    _print_ranking(candidates, winner)

    version = _register_winner(winner)
    _promote(version)

    print(
        f"\nModelo vencedor: {winner.name!r} (run_id={winner.run_id[:8]}) "
        f"registrado como {MLFLOW_REGISTERED_MODEL_NAME} v{version.version}."
    )


def _score_candidates(runs: pd.DataFrame) -> list[Candidate]:
    # Discard baselines (dummy) from selection — they serve only as a reference.
    runs = runs[~runs["tags.model_family"].fillna("").eq("dummy")]

    # Normalizations to compose the score.
    needed = ["metrics.tech.f1", "metrics.biz.expected_cost_per_case", "metrics.train.fit_seconds"]
    for col in needed:
        if col not in runs.columns:
            runs[col] = float("nan")
    runs = runs.dropna(subset=["metrics.tech.f1"])
    if runs.empty:
        return []

    f1 = runs["metrics.tech.f1"].to_numpy()
    cost = runs["metrics.biz.expected_cost_per_case"].fillna(f1.max()).to_numpy()
    fit = runs["metrics.train.fit_seconds"].fillna(0.0).to_numpy()

    # Min-max scaling, inverting signs when "lower is better".
    f1_norm = _minmax(f1)
    cost_norm = 1.0 - _minmax(cost)  # menor custo -> maior score
    fit_norm = 1.0 - _minmax(fit)  # menor tempo -> maior score

    scores = WEIGHT_PERFORMANCE * f1_norm + WEIGHT_ROBUSTNESS * cost_norm + WEIGHT_EFFICIENCY * fit_norm

    candidates: list[Candidate] = []
    for (_, row), score in zip(runs.iterrows(), scores, strict=True):
        candidates.append(
            Candidate(
                run_id=row["run_id"],
                name=row.get("tags.mlflow.runName", row["run_id"][:8]),
                f1=float(row["metrics.tech.f1"]),
                roc_auc=float(row.get("metrics.tech.roc_auc", float("nan"))),
                expected_cost=float(row["metrics.biz.expected_cost_per_case"])
                if pd.notna(row["metrics.biz.expected_cost_per_case"])
                else float("nan"),
                fit_seconds=float(row.get("metrics.train.fit_seconds", float("nan"))),
                score=float(score),
            )
        )
    return candidates


def _minmax(values) -> pd.Series:
    arr = pd.Series(values).astype(float)
    if arr.max() == arr.min():
        return pd.Series([0.5] * len(arr))
    return (arr - arr.min()) / (arr.max() - arr.min())


def _print_ranking(candidates: list[Candidate], winner: Candidate) -> None:
    print("\n=== Ranking de selecao ===")
    header = f"{'Run':<28} {'F1':<8} {'AUC':<8} {'Cost':<8} {'Fit(s)':<8} {'Score':<8}"
    print(header)
    print("-" * len(header))
    for c in sorted(candidates, key=lambda x: -x.score):
        mark = " * " if c.run_id == winner.run_id else "   "
        print(
            f"{mark}{c.name[:25]:<25} {c.f1:<8.4f} {c.roc_auc:<8.4f} "
            f"{c.expected_cost:<8.4f} {c.fit_seconds:<8.2f} {c.score:<8.4f}"
        )


def _register_winner(candidate: Candidate):
    """Create a new model version in the Registry pointing to the winning run."""

    client = mlflow.tracking.MlflowClient()
    _ensure_registered_model(client)

    model_uri = f"runs:/{candidate.run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=MLFLOW_REGISTERED_MODEL_NAME)
    return version


def _ensure_registered_model(client) -> None:
    try:
        client.get_registered_model(MLFLOW_REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MLFLOW_REGISTERED_MODEL_NAME)


def _promote(version) -> None:
    """Promote the version to Production stage (archiving previous versions)."""

    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=MLFLOW_REGISTERED_MODEL_NAME,
            version=version.version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as exc:
        # Stages API was marked deprecated in MLflow 2.9; if the API is not
        # available, we proceed — load_production_model falls back to the
        # latest version.
        print(f"[aviso] nao foi possivel promover a versao: {exc}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seleciona e promove o modelo campeao.")
    parser.add_argument(
        "--experiment",
        default=MLFLOW_EXPERIMENT_NAME,
        help="Nome do experimento do MLflow.",
    )
    return parser


if __name__ == "__main__":
    main()
