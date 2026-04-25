"""
AWS Lambda adapter for the FastAPI inference API.

Loads the exported model (joblib) once on cold start and injects it into the
FastAPI app state, bypassing the default lifespan (which tries to load from
MLflow Registry, unavailable in Lambda).

Warming requests (CloudWatch Events every 10 min) are intercepted before
reaching Mangum.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from mangum import Mangum

from cardio_ml.data.ingestion import feature_columns
from cardio_ml.inference.predict import CardioPredictor, ModelMetadata
from cardio_ml.serving.api import _state, app

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/var/task/model/pipeline.joblib"))


def _load_model() -> CardioPredictor:
    """Load the pipeline from disk and inject into the app state."""

    pipeline = joblib.load(MODEL_PATH)
    predictor = CardioPredictor(
        pipeline=pipeline,
        metadata=ModelMetadata(
            name=os.environ.get("MODEL_NAME", "cardio-dcv-classifier"),
            version=os.environ.get("MODEL_VERSION", "1"),
            stage="Production",
            run_id=os.environ.get("MODEL_RUN_ID"),
            features=feature_columns(),
        ),
    )
    _state["predictor"] = predictor
    return predictor


_load_model()

_mangum = Mangum(app, lifespan="off")


def handler(event, context):
    """Lambda runtime entry point.

    CloudWatch warming events have a 'source' field; API Gateway events don't.
    This distinction is cheap and sufficient.
    """

    if event.get("source") == "aws.events":
        return {"statusCode": 200, "body": "warm"}

    return _mangum(event, context)
