"""CardioPredictor and API tests (without loading a real model)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cardio_ml.data.ingestion import feature_columns
from cardio_ml.features.preprocessing import build_preprocessor
from cardio_ml.inference.predict import CardioPredictor, ModelMetadata


def _fake_predictor(X: pd.DataFrame, y) -> CardioPredictor:
    """Build a CardioPredictor with a fast pipeline for use in tests."""

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X, y)
    return CardioPredictor(
        pipeline=pipeline,
        metadata=ModelMetadata(
            name="cardio-dcv-classifier",
            version="test",
            stage="Staging",
            run_id=None,
            features=feature_columns(),
        ),
    )


def test_predictor_rejects_frame_with_missing_columns(synthetic_frame, feature_cols):
    predictor = _fake_predictor(synthetic_frame[feature_cols], synthetic_frame["cardio"])
    bad = synthetic_frame[feature_cols].drop(columns=["ap_hi"])
    with pytest.raises(ValueError):
        predictor.predict(bad)


def test_predictor_predicts_and_returns_probability(synthetic_frame, feature_cols):
    predictor = _fake_predictor(synthetic_frame[feature_cols], synthetic_frame["cardio"])
    X_pred = synthetic_frame[feature_cols].head(10)

    classes = predictor.predict(X_pred)
    assert classes.shape == (10,)
    assert set(classes).issubset({0, 1})

    proba = predictor.predict_proba(X_pred)
    assert proba.shape == (10,)
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_api_health_endpoint_returns_ok():
    """The /health endpoint should respond even without a loaded model.

    The lifespan tries to load the production model; if it fails, the API
    starts in degraded mode, with /health responding 200 and `model_loaded=False`.
    """

    from cardio_ml.serving import api as api_module

    with TestClient(api_module.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] in (True, False)


def test_api_predict_returns_predictions(monkeypatch, synthetic_frame, feature_cols):
    """Test /predict by injecting a fake predictor in place of the real load."""

    from cardio_ml.serving import api as api_module

    predictor = _fake_predictor(synthetic_frame[feature_cols], synthetic_frame["cardio"])
    # The lifespan calls load_production_model on startup; we replace it to
    # return the fake predictor without hitting the MLflow Registry.
    monkeypatch.setattr(api_module, "load_production_model", lambda: predictor)

    with TestClient(api_module.app) as client:
        resp = client.post(
            "/predict",
            json={
                "patients": [
                    {
                        "age_years": 55.0,
                        "height": 170.0,
                        "weight": 75.0,
                        "ap_hi": 130.0,
                        "ap_lo": 85.0,
                        "gender": 1,
                        "cholesterol": 1,
                        "gluc": 1,
                        "smoke": 0,
                        "alco": 0,
                        "active": 1,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test"
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert prediction["predicted_class"] in (0, 1)
    assert prediction["risk_level"] in ("baixo", "moderado", "alto")
