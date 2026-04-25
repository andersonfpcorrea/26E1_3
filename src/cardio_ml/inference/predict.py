"""
Inference from the versioned model in the MLflow Model Registry.

The `CardioPredictor` class acts as a thin facade over the loaded pipeline,
exposing only what the API needs (`predict`, `predict_proba`, metadata). This
keeps the API agnostic to MLflow details and makes it easy to replace the model
source in the future (e.g., S3, remote MLflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES, feature_columns


@dataclass(frozen=True)
class ModelMetadata:
    """Served model information — exposed by the /model-info route."""

    name: str
    version: str
    stage: str
    run_id: str | None
    features: list[str]


class CardioPredictor:
    """Inference facade.

    Constructors:
      - `load_production_model(...)` (recommended): loads from the MLflow Registry.
      - Direct usage: initialize with an already-trained `sklearn` pipeline.
    """

    def __init__(self, pipeline: Any, metadata: ModelMetadata):
        self._pipeline = pipeline
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the predicted classes (0 or 1)."""

        self._validate_frame(frame)
        return self._pipeline.predict(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the probability of the positive class (cardio=1)."""

        self._validate_frame(frame)
        if not hasattr(self._pipeline, "predict_proba"):
            raise AttributeError("Modelo carregado nao expoe predict_proba.")
        return self._pipeline.predict_proba(frame)[:, 1]

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        """Validate that the DataFrame has the columns expected by the pipeline."""

        expected = set(feature_columns())
        missing = expected - set(frame.columns)
        if missing:
            raise ValueError(
                f"Colunas ausentes para inferencia: {sorted(missing)}. "
                f"Esperado: numericas={NUMERIC_FEATURES}, categoricas={CATEGORICAL_FEATURES}."
            )


def load_production_model(
    model_name: str | None = None,
    stage: str = "Production",
) -> CardioPredictor:
    """Load the promoted model from the MLflow Registry.

    In environments where the `Production` stage is not yet configured, falls
    back to the latest registered version. This flexibility is useful during
    development and in CI.
    """

    import mlflow

    from cardio_ml.config import MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = model_name or MLFLOW_REGISTERED_MODEL_NAME

    client = mlflow.tracking.MlflowClient()
    version_info = _resolve_model_version(client, model_name, stage)
    model_uri = f"models:/{model_name}/{version_info.version}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    metadata = ModelMetadata(
        name=model_name,
        version=str(version_info.version),
        stage=version_info.current_stage,
        run_id=version_info.run_id,
        features=feature_columns(),
    )
    return CardioPredictor(pipeline=pipeline, metadata=metadata)


def _resolve_model_version(client, model_name: str, stage: str):
    """Find the most suitable model version in the Registry.

    Tries (1) the given `stage`; (2) the highest available version. If nothing
    is found, raises a clear error.
    """

    try:
        versions_in_stage = client.get_latest_versions(model_name, stages=[stage])
    except Exception:
        versions_in_stage = []

    if versions_in_stage:
        return versions_in_stage[0]

    # Fallback: highest registered version, regardless of stage.
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        raise RuntimeError(
            f"Modelo {model_name!r} nao esta registrado no MLflow. "
            "Execute scripts/select_final_model.py antes de iniciar o serviço."
        )
    return max(all_versions, key=lambda v: int(v.version))
