"""
Data and model drift detection.

Custom implementation of PSI (Population Stability Index) and KS test, which
are the two classic instruments for production monitoring:
  - **PSI** measures how much a variable's distribution has shifted relative to
    a reference, summarizing it in a single number. Industry reference
    thresholds: PSI < 0.1 (stable), 0.1-0.25 (moderate change, investigate),
    > 0.25 (significant change, probable drift).
  - **KS (Kolmogorov-Smirnov)** is a non-parametric statistical test that
    returns a statistic and a p-value for the hypothesis that the two samples
    come from the same distribution.

The `compute_drift` function applies both for each numeric feature and
aggregates into a `DriftReport`. For model drift, the same mechanism is applied
to the distribution of predicted probabilities.

Integration with Evidently is in `scripts/simulate_drift.py`, which uses this
module for quantitative analysis and Evidently for a rich HTML report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_STABLE_THRESHOLD: float = 0.10
PSI_MODERATE_THRESHOLD: float = 0.25


@dataclass(frozen=True)
class FeatureDrift:
    """Drift analysis result for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    verdict: str  # "estavel" | "moderado" | "drift"


@dataclass(frozen=True)
class DriftReport:
    """Aggregation of drift across all analyzed features."""

    per_feature: list[FeatureDrift] = field(default_factory=list)
    prediction_drift: FeatureDrift | None = None

    def as_dict(self) -> dict:
        return {
            "per_feature": [asdict(f) for f in self.per_feature],
            "prediction_drift": asdict(self.prediction_drift) if self.prediction_drift else None,
        }

    def has_drift(self) -> bool:
        """Return True if any feature or the prediction shows significant drift."""

        for feat in self.per_feature:
            if feat.verdict == "drift":
                return True
        return bool(self.prediction_drift and self.prediction_drift.verdict == "drift")


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute the PSI between two 1-D samples.

    Uses the reference sample's quantiles as bin edges, ensuring that the
    reference is uniformly distributed across bins. A small epsilon prevents
    division by zero.
    """

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        return float("nan")

    # Use reference quantiles as edges; discard duplicate edges for
    # low-variance features.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 3:
        # Nearly constant distribution — PSI is not informative.
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _classify(psi: float) -> str:
    if psi < PSI_STABLE_THRESHOLD:
        return "estavel"
    if psi < PSI_MODERATE_THRESHOLD:
        return "moderado"
    return "drift"


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    ref_predictions: np.ndarray | None = None,
    cur_predictions: np.ndarray | None = None,
) -> DriftReport:
    """Compare two samples (reference and current) and return a DriftReport.

    Only numeric columns are evaluated via PSI + KS. Categorical columns
    can be added in a future evolution via chi2.
    """

    per_feature: list[FeatureDrift] = []

    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        if not np.issubdtype(reference[col].dtype, np.number):
            continue

        ref_arr = reference[col].dropna().to_numpy()
        cur_arr = current[col].dropna().to_numpy()

        psi = compute_psi(ref_arr, cur_arr)
        ks = ks_2samp(ref_arr, cur_arr)

        per_feature.append(
            FeatureDrift(
                feature=col,
                psi=psi,
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                verdict=_classify(psi),
            )
        )

    prediction_drift = None
    if ref_predictions is not None and cur_predictions is not None:
        ref_p = np.asarray(ref_predictions, dtype=float)
        cur_p = np.asarray(cur_predictions, dtype=float)
        psi = compute_psi(ref_p, cur_p)
        ks = ks_2samp(ref_p, cur_p)
        prediction_drift = FeatureDrift(
            feature="prediction",
            psi=psi,
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            verdict=_classify(psi),
        )

    return DriftReport(per_feature=per_feature, prediction_drift=prediction_drift)
