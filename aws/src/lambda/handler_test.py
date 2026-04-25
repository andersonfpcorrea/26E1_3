"""
Lambda handler unit tests.

Strategy: create a minimal model (LogisticRegression inside a Pipeline) before
importing the handler, so that the cold-start loading works without depending
on the real trained model. The fixture model is disposable — the goal is to test
routing (warming vs API Gateway) and the FastAPI-Mangum integration, not the
prediction quality.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from cardio_ml.features.preprocessing import build_preprocessor


@pytest.fixture(autouse=True, scope="module")
def _setup_model():
    """Create a fixture model and point MODEL_PATH to it before importing the handler."""

    feature_cols = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    rng = np.random.default_rng(42)
    n = 100

    X = pd.DataFrame(
        {
            "age_years": rng.uniform(30, 70, n).round(1),
            "height": rng.normal(170, 8, n).round(0),
            "weight": rng.normal(75, 15, n).round(0),
            "ap_hi": rng.normal(125, 15, n).round(0),
            "ap_lo": rng.normal(82, 10, n).round(0),
            "bmi": rng.normal(26, 4, n).round(2),
            "gender": rng.choice([1, 2], n),
            "cholesterol": rng.choice([1, 2, 3], n),
            "gluc": rng.choice([1, 2, 3], n),
            "smoke": rng.choice([0, 1], n),
            "alco": rng.choice([0, 1], n),
            "active": rng.choice([0, 1], n),
        }
    )
    y = rng.choice([0, 1], n)

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X[feature_cols], y)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name
    joblib.dump(pipeline, model_path)

    os.environ["MODEL_PATH"] = model_path

    yield

    Path(model_path).unlink(missing_ok=True)


def _get_handler():
    """Import the handler after the fixture has set MODEL_PATH."""

    # Reimport to force _load_model() with the correct MODEL_PATH.
    import importlib
    import handler as h

    importlib.reload(h)
    return h.handler


class TestWarmingEvent:
    def test_returns_200_for_cloudwatch_event(self):
        h = _get_handler()
        event = {"source": "aws.events", "detail-type": "Scheduled Event"}
        result = h(event, None)
        assert result["statusCode"] == 200
        assert result["body"] == "warm"


class TestApiGatewayEvent:
    def _make_apigw_event(
        self, method: str, path: str, body: str | None = None
    ) -> dict:
        """Build a minimal HTTP API v2 event."""

        return {
            "version": "2.0",
            "routeKey": f"{method} {path}",
            "rawPath": path,
            "rawQueryString": "",
            "headers": {"content-type": "application/json"},
            "requestContext": {
                "http": {
                    "method": method,
                    "path": path,
                    "protocol": "HTTP/1.1",
                    "sourceIp": "127.0.0.1",
                    "userAgent": "pytest",
                },
                "requestId": "test-id",
                "routeKey": f"{method} {path}",
                "stage": "$default",
                "time": "01/Jan/2026:00:00:00 +0000",
                "timeEpoch": 0,
                "accountId": "123456789",
                "apiId": "test",
                "domainName": "test.execute-api.us-east-1.amazonaws.com",
                "domainPrefix": "test",
            },
            "body": body,
            "isBase64Encoded": False,
        }

    def test_health_returns_200(self):
        h = _get_handler()
        event = self._make_apigw_event("GET", "/health")
        result = h(event, None)
        assert result["statusCode"] == 200

    def test_predict_returns_200(self):
        import json

        h = _get_handler()
        body = json.dumps(
            {
                "patients": [
                    {
                        "age_years": 55,
                        "height": 170,
                        "weight": 78,
                        "ap_hi": 140,
                        "ap_lo": 90,
                        "gender": 1,
                        "cholesterol": 2,
                        "gluc": 1,
                        "smoke": 0,
                        "alco": 0,
                        "active": 1,
                    }
                ],
            }
        )
        event = self._make_apigw_event("POST", "/predict", body)
        result = h(event, None)
        assert result["statusCode"] == 200

        response = json.loads(result["body"])
        assert "predictions" in response
        assert len(response["predictions"]) == 1
        assert response["predictions"][0]["predicted_class"] in (0, 1)
        assert response["predictions"][0]["risk_level"] in ("baixo", "moderado", "alto")
