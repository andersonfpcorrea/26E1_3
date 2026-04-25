"""
Inference API (FastAPI).

Endpoints:
  GET  /health       - liveness + readiness probe.
  GET  /model-info   - served model metadata (version, stage, features).
  POST /predict      - inference on one or more patients.

The API is deliberately thin: it loads the model once at startup and delegates
to `CardioPredictor`. Input validation is handled by Pydantic, which rejects
malformed payloads before they reach the model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cardio_ml.data.ingestion import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from cardio_ml.inference.predict import CardioPredictor, load_production_model

STATIC_DIR = Path(__file__).parent / "static"

# Module-level model state — lifecycle controlled by lifespan.
_state: dict[str, CardioPredictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model at startup and release at shutdown.

    Loading errors are caught here to let the API start in degraded mode
    (returning 503 on /predict), useful in CI before a model has been
    promoted to production.
    """

    try:
        _state["predictor"] = load_production_model()
    except Exception as exc:
        # Keep the service up so /health responds; /predict returns 503.
        _state["predictor"] = None
        _state["load_error"] = str(exc)  # type: ignore[assignment]

    yield

    _state["predictor"] = None


app = FastAPI(
    title="Cardio ML — Inference API",
    description="Servico de inferencia para o modelo de triagem de doencas cardiovasculares.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PatientRecord(BaseModel):
    """A single observation to be classified. Names mirror the dataset features."""

    age_years: float = Field(..., ge=18, le=100, description="Idade em anos.")
    height: float = Field(..., ge=120, le=230, description="Altura em centimetros.")
    weight: float = Field(..., ge=30, le=250, description="Peso em kg.")
    ap_hi: float = Field(..., ge=60, le=260, description="Pressao sistolica (mmHg).")
    ap_lo: float = Field(..., ge=30, le=180, description="Pressao diastolica (mmHg).")
    bmi: float | None = Field(
        None, description="IMC em kg/m^2. Se omitido, calculado a partir de altura e peso."
    )
    gender: Literal[1, 2] = Field(..., description="1=Feminino, 2=Masculino.")
    cholesterol: Literal[1, 2, 3]
    gluc: Literal[1, 2, 3]
    smoke: Literal[0, 1]
    alco: Literal[0, 1]
    active: Literal[0, 1]

    def to_dict(self) -> dict:
        data = self.model_dump()
        if data.get("bmi") is None:
            height_m = data["height"] / 100.0
            data["bmi"] = round(data["weight"] / (height_m**2), 2)
        return data


class PredictRequest(BaseModel):
    patients: list[PatientRecord] = Field(..., min_length=1, max_length=1000)


class PredictionItem(BaseModel):
    predicted_class: int = Field(..., description="0 = sem DCV, 1 = com DCV.")
    probability: float | None = Field(..., description="Probabilidade da classe positiva.")
    risk_level: Literal["baixo", "moderado", "alto"]


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[PredictionItem]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    stage: str
    run_id: str | None
    features_numeric: list[str]
    features_categorical: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
@app.get("/ui")
def ui():
    """Web interface for interactive screening."""

    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict:
    """Liveness + readiness. Returns 200 even without a loaded model."""

    return {
        "status": "ok",
        "model_loaded": _state["predictor"] is not None,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Served model metadata. Useful for auditing."""

    predictor = _require_predictor()
    meta = predictor.metadata
    return ModelInfoResponse(
        name=meta.name,
        version=meta.version,
        stage=meta.stage,
        run_id=meta.run_id,
        features_numeric=list(NUMERIC_FEATURES),
        features_categorical=list(CATEGORICAL_FEATURES),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Run inference on one or more observations."""

    predictor = _require_predictor()

    frame = pd.DataFrame([p.to_dict() for p in payload.patients])

    try:
        classes = predictor.predict(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probabilities: list[float | None]
    try:
        proba = predictor.predict_proba(frame)
        probabilities = [float(p) for p in proba]
    except (AttributeError, Exception):
        probabilities = [None] * len(classes)

    items: list[PredictionItem] = []
    for cls, prob in zip(classes, probabilities, strict=True):
        items.append(
            PredictionItem(
                predicted_class=int(cls),
                probability=prob,
                risk_level=_risk_level(prob),
            )
        )

    return PredictResponse(
        model_version=predictor.metadata.version,
        predictions=items,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_predictor() -> CardioPredictor:
    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo nao carregado. "
                f"Motivo: {_state.get('load_error', 'desconhecido')}. "
                "Execute scripts/select_final_model.py para registrar um modelo."
            ),
        )
    return predictor


def _risk_level(probability: float | None) -> Literal["baixo", "moderado", "alto"]:
    """Translate probability into a human-readable risk level.

    Thresholds chosen based on the clinical context: false negative costs more
    than false positive, so the threshold for 'alto' is deliberately sensitive.
    """

    if probability is None:
        return "moderado"
    if probability < 0.35:
        return "baixo"
    if probability < 0.65:
        return "moderado"
    return "alto"
