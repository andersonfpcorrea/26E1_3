"""
Run the full experiment suite for this project.

For each model in the registry and each dimensionality reduction technique
(`none`, `pca`, `lda`), launches a separate MLflow run. At the end, prints a
table with the CV score and test F1 of each execution, allowing a quick read
without opening the MLflow UI.

Usage:
    python scripts/run_full_experiment.py
    python scripts/run_full_experiment.py --models xgboost logistic_regression
    python scripts/run_full_experiment.py --dims none pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script without installing the package in editable mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardio_ml.models.registry import MODEL_REGISTRY  # noqa: E402
from cardio_ml.training.train import train_model  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roda todos os experimentos da suite.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="Modelos a treinar (default: todos).",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        default=["none", "pca", "lda"],
        choices=["none", "pca", "lda"],
        help="Tecnicas de reducao de dimensionalidade.",
    )
    parser.add_argument(
        "--scoring",
        default="f1",
        help="Metrica usada na validacao cruzada.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    results = []
    total = len(args.models) * len(args.dims)
    counter = 0

    print(f"Iniciando suite: {total} experimentos.")
    for model_name in args.models:
        for dim in args.dims:
            counter += 1
            print(f"\n[{counter}/{total}] Treinando {model_name} ({dim})...")
            output = train_model(
                model_name=model_name,
                dim_technique=dim,
                scoring=args.scoring,
            )
            results.append(output)

    _print_summary(results, args.scoring)


def _print_summary(results, scoring: str) -> None:
    print("\n=== Resumo da suite ===")
    header = f"{'Modelo':<22} {'Dim':<6} {'CV ' + scoring:<12} {'Test F1':<10} {'Test AUC':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        f1 = r.test_metrics.get("tech.f1", float("nan"))
        auc = r.test_metrics.get("tech.roc_auc", float("nan"))
        print(f"{r.model:<22} {r.dim_technique:<6} {r.cv_score:<12.4f} {f1:<10.4f} {auc:<10.4f}")


if __name__ == "__main__":
    main()
