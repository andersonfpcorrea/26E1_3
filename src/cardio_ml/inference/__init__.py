"""Inference layer: versioned model loading and prediction."""

from cardio_ml.inference.predict import CardioPredictor, load_production_model

__all__ = ["CardioPredictor", "load_production_model"]
