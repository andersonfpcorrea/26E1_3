"""
Technical metrics and business impact metrics.

Technical metrics answer "did the model learn?": accuracy, precision, recall,
F1, ROC-AUC, average precision.

Business metrics answer "is the model worth it?". They translate confusion into
real costs/benefits for the CVD clinical screening context:
  - Expected error cost: weighting of FN and FP. A false negative (undetected
    disease) is typically 5x more costly than a false positive.
  - Capture rate: recall on the positive class (actually sick patients).
  - False alarm rate: FPR (healthy patients classified as sick).
  - Positive cases caught: estimate of additional sick patients identified vs.
    a random baseline (absolute uplift).

These weights are explicit so they can be easily audited and adjusted in the
technical report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Default weights for error costs (see documentation at the top of the module).
COST_FALSE_NEGATIVE: float = 5.0
COST_FALSE_POSITIVE: float = 1.0


@dataclass(frozen=True)
class TechnicalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BusinessMetrics:
    expected_cost_per_case: float
    capture_rate: float
    false_alarm_rate: float
    positives_caught_over_random: float
    cost_weight_fn: float
    cost_weight_fp: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_technical_metrics(y_true, y_pred, y_proba=None) -> TechnicalMetrics:
    """Compute standard technical metrics. Optional `y_proba` enables AUC/AP."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    roc_auc = float("nan")
    avg_prec = float("nan")
    if y_proba is not None:
        proba = np.asarray(y_proba)
        roc_auc = float(roc_auc_score(y_true, proba))
        avg_prec = float(average_precision_score(y_true, proba))

    return TechnicalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        average_precision=avg_prec,
    )


def compute_business_metrics(
    y_true,
    y_pred,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> BusinessMetrics:
    """Translate the confusion matrix into business metrics.

    The "positives caught over random" calculation assumes that a random model
    with the same positive rate as the target would, on average, correctly
    identify `prevalence * n_positives` positive cases.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n_total = len(y_true)
    n_positives = int(tp + fn)
    prevalence = n_positives / n_total if n_total else 0.0

    cost_total = cost_fn * fn + cost_fp * fp
    expected_cost = float(cost_total / n_total) if n_total else 0.0

    capture_rate = float(tp / n_positives) if n_positives else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0

    random_positives_caught = prevalence * n_positives
    uplift = float(tp - random_positives_caught)

    return BusinessMetrics(
        expected_cost_per_case=expected_cost,
        capture_rate=capture_rate,
        false_alarm_rate=false_alarm_rate,
        positives_caught_over_random=uplift,
        cost_weight_fn=cost_fn,
        cost_weight_fp=cost_fp,
    )


def flatten_metrics(tech: TechnicalMetrics, biz: BusinessMetrics) -> dict[str, float]:
    """Merge the two structures into a single dict — ideal format for MLflow."""

    out: dict[str, float] = {}
    for k, v in tech.as_dict().items():
        out[f"tech.{k}"] = v
    for k, v in biz.as_dict().items():
        out[f"biz.{k}"] = v
    return out
