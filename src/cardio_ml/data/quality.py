"""
Data quality diagnostics.

Raises signals that influence modeling and operational decisions:
  - Impossible blood pressure values (negative, zero, extreme outliers);
  - Physiological outliers in height, weight and BMI;
  - Class and sensitive subgroup (gender) balance;
  - Duplicates;
  - Presence of null values.

The return value is an immutable structure (QualityReport) that can be serialized
to the technical report and also logged as an MLflow artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Clinically plausible ranges for filtering outliers. The bounds are intentionally
# conservative: the goal is to report, not silently remove.
_PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "ap_hi": (80.0, 220.0),
    "ap_lo": (40.0, 140.0),
    "height": (140.0, 210.0),
    "weight": (35.0, 200.0),
    "bmi": (14.0, 60.0),
    "age_years": (25.0, 75.0),
}


@dataclass(frozen=True)
class QualityIssue:
    """A structural issue found in the dataset."""

    column: str
    description: str
    affected_rows: int
    severity: str  # "baixa" | "media" | "alta"


@dataclass(frozen=True)
class QualityReport:
    """Summary of the quality diagnostics."""

    n_rows: int
    n_duplicates: int
    n_missing_total: int
    class_balance: dict[int, float]
    gender_balance: dict[int, float]
    issues: list[QualityIssue] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as markdown ready for the technical report."""

        lines = [
            "## Diagnostico de Qualidade dos Dados",
            "",
            f"- Total de registros: **{self.n_rows:,}**",
            f"- Duplicatas: **{self.n_duplicates}**",
            f"- Valores ausentes (total): **{self.n_missing_total}**",
            "",
            "### Balanceamento de classes (target `cardio`)",
            "",
        ]
        for cls, pct in sorted(self.class_balance.items()):
            lines.append(f"- Classe `{cls}`: {pct:.1%}")

        lines.extend(["", "### Balanceamento por genero", ""])
        for g, pct in sorted(self.gender_balance.items()):
            lines.append(f"- Genero `{g}`: {pct:.1%}")

        lines.extend(["", "### Problemas detectados", ""])
        if not self.issues:
            lines.append("- Nenhum problema estrutural detectado.")
        else:
            for issue in self.issues:
                lines.append(
                    f"- **[{issue.severity}]** `{issue.column}` — "
                    f"{issue.description} ({issue.affected_rows:,} linhas)"
                )

        return "\n".join(lines)


def diagnose_quality(frame: pd.DataFrame, target_col: str = "cardio") -> QualityReport:
    """Produce a quality report for the given DataFrame."""

    issues: list[QualityIssue] = []

    # Nulls per column — contribute to the total and generate an issue per column.
    missing_by_col = frame.isna().sum()
    n_missing_total = int(missing_by_col.sum())
    for col, n in missing_by_col.items():
        if n > 0:
            issues.append(
                QualityIssue(
                    column=str(col),
                    description="Presenca de valores ausentes",
                    affected_rows=int(n),
                    severity="media",
                )
            )

    # Duplicates based on all columns except `id`.
    cols_for_dup = [c for c in frame.columns if c != "id"]
    n_duplicates = int(frame.duplicated(subset=cols_for_dup).sum())
    if n_duplicates > 0:
        issues.append(
            QualityIssue(
                column="<dataset>",
                description="Linhas duplicadas em features+target",
                affected_rows=n_duplicates,
                severity="media",
            )
        )

    # Physiological outliers.
    for col, (low, high) in _PLAUSIBLE_BOUNDS.items():
        if col not in frame.columns:
            continue
        mask = (frame[col] < low) | (frame[col] > high)
        n_out = int(mask.sum())
        if n_out > 0:
            severity = "alta" if n_out > 0.01 * len(frame) else "baixa"
            issues.append(
                QualityIssue(
                    column=col,
                    description=f"Valores fora do intervalo fisiologico plausivel [{low}, {high}]",
                    affected_rows=n_out,
                    severity=severity,
                )
            )

    # Systolic vs diastolic pressure relationship. Inversion signals a data entry error.
    if {"ap_hi", "ap_lo"}.issubset(frame.columns):
        inverted = (frame["ap_lo"] > frame["ap_hi"]).sum()
        if inverted > 0:
            issues.append(
                QualityIssue(
                    column="ap_hi/ap_lo",
                    description="Pressao diastolica maior que sistolica (inversao)",
                    affected_rows=int(inverted),
                    severity="alta",
                )
            )

    class_balance = _proportion(frame[target_col]) if target_col in frame.columns else {}
    gender_balance = _proportion(frame["gender"]) if "gender" in frame.columns else {}

    return QualityReport(
        n_rows=len(frame),
        n_duplicates=n_duplicates,
        n_missing_total=n_missing_total,
        class_balance=class_balance,
        gender_balance=gender_balance,
        issues=issues,
    )


def _proportion(series: pd.Series) -> dict[int, float]:
    """Return the proportion of each category in a Series."""

    counts = series.value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in counts.items()}
