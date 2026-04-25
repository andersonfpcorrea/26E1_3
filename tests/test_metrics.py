"""Technical and business metrics tests."""

from __future__ import annotations

import numpy as np

from cardio_ml.evaluation.metrics import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    compute_business_metrics,
    compute_technical_metrics,
)


def test_perfect_prediction_metrics():
    y = np.array([0, 1, 0, 1, 1, 0])
    tech = compute_technical_metrics(y, y, y.astype(float))
    assert tech.accuracy == 1.0
    assert tech.precision == 1.0
    assert tech.recall == 1.0
    assert tech.f1 == 1.0


def test_business_cost_weights_false_negatives_more():
    # Both with 1 error: one FN and one FP. FN should weigh more.
    y_true = np.array([1, 0, 1, 0])
    y_pred_fn = np.array([0, 0, 1, 0])  # 1 FN
    y_pred_fp = np.array([1, 1, 1, 0])  # 1 FP

    biz_fn = compute_business_metrics(y_true, y_pred_fn)
    biz_fp = compute_business_metrics(y_true, y_pred_fp)

    # Expected cost of FN > expected cost of FP in the same proportion as the weights.
    assert biz_fn.expected_cost_per_case > biz_fp.expected_cost_per_case
    ratio = biz_fn.expected_cost_per_case / biz_fp.expected_cost_per_case
    assert abs(ratio - COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE) < 1e-6


def test_business_metrics_uplift_positive_when_better_than_random():
    y_true = np.array([1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 0])  # 3 de 4 positivos capturados
    biz = compute_business_metrics(y_true, y_pred)
    assert biz.positives_caught_over_random > 0
    assert 0 <= biz.capture_rate <= 1
