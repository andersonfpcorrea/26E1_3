"""
Simulate a drift scenario and generate a quantitative + HTML report.

Demonstrative purpose: we deliberately perturb key features (age, blood
pressure, cholesterol) to emulate a change in the served population. In a real
system, `current` would come from production traffic, not from a simulation.

Outputs:
  - reports/drift_summary.json   (PSI/KS per feature, via `evaluation/drift.py`)
  - reports/drift_evidently.html (rich Evidently report)

Both formats are useful: the JSON feeds into automated alerts, the HTML
is designed for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardio_ml.config import REPORTS_DIR, SEED  # noqa: E402
from cardio_ml.data.ingestion import TARGET_COLUMN, feature_columns, load_raw_dataset  # noqa: E402
from cardio_ml.data.split import stratified_train_test_split  # noqa: E402
from cardio_ml.evaluation.drift import compute_drift  # noqa: E402


def _inject_drift(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Apply deterministic perturbations to the DataFrame.

    - Shifts age by +3 years (aging population).
    - Increases systolic blood pressure by +7 mmHg (underdiagnosed hypertension).
    - Inflates the proportion of patients with elevated cholesterol.
    """

    rng = np.random.default_rng(seed)
    drifted = frame.copy()

    drifted["age_years"] = drifted["age_years"] + 3.0
    drifted["ap_hi"] = drifted["ap_hi"] + 7.0 + rng.normal(0, 2, size=len(drifted))

    # Push ~20% of patients with cholesterol=1 to category 2 (above normal).
    mask = (drifted["cholesterol"] == 1) & (rng.random(len(drifted)) < 0.20)
    drifted.loc[mask, "cholesterol"] = 2

    return drifted


def _render_evidently(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> Path | None:
    """Generate an HTML report with Evidently. Silent failure if not available."""

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception as exc:
        print(f"[aviso] Evidently indisponivel ({exc}); ignorando HTML report.")
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output))
    return output


def main() -> None:
    args = _parser().parse_args()

    dataset = load_raw_dataset()
    cols = feature_columns()
    split = stratified_train_test_split(
        frame=dataset.frame,
        feature_cols=cols,
        target_col=TARGET_COLUMN,
    )

    reference = split.X_train.copy()
    reference[TARGET_COLUMN] = split.y_train.values

    # Simulated "production traffic": a sample from the test set with injected drift.
    current = _inject_drift(split.X_test.sample(frac=args.sample, random_state=SEED))
    current[TARGET_COLUMN] = split.y_test.sample(frac=args.sample, random_state=SEED).values[: len(current)]

    print(f"Reference: {len(reference):,} linhas  |  Current: {len(current):,} linhas")

    report = compute_drift(reference, current, feature_cols=cols)
    print("\n=== Drift por feature ===")
    for entry in report.per_feature:
        print(
            f"  {entry.feature:<15} PSI={entry.psi:7.4f}  KS={entry.ks_statistic:6.4f}  "
            f"p={entry.ks_pvalue:7.4f}  [{entry.verdict}]"
        )
    print(f"\nDrift detectado: {'SIM' if report.has_drift() else 'NAO'}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(report.as_dict(), indent=2))
    print(f"\nResumo salvo em: {summary_path}")

    html_path = REPORTS_DIR / "drift_evidently.html"
    if _render_evidently(reference, current, html_path):
        print(f"Relatorio Evidently: {html_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simula e analisa drift de dados.")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fracao do conjunto de teste a ser usada como 'producao'.",
    )
    return parser


if __name__ == "__main__":
    main()
