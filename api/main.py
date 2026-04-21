from __future__ import annotations

import math
import pickle
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.features import (
    FEATURE_COLUMNS,
    NOTEBOOK_SUMMARY,
    TRAINING_CONSTRAINTS,
    build_features_frame,
    build_trip_features,
)
from src.schema import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TaxiTripRequest,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "src" / "model.pkl"


def _validate_model_features(model: Any) -> None:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return

    resolved_feature_names = [str(name) for name in feature_names]
    if resolved_feature_names != FEATURE_COLUMNS:
        raise RuntimeError(
            "Loaded model feature order does not match the notebook feature pipeline."
        )


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    _validate_model_features(model)
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="NYC Taxi Trip Duration API",
    description=(
        "Predicts NYC taxi trip duration using the XGBoost model trained in "
        "src/taxidata.ipynb."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "message": "NYC Taxi Trip Duration API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    model_loaded = False
    try:
        load_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    model = load_model()
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model_type = f"{model.__class__.__name__}({model.named_steps['model'].__class__.__name__})"
    else:
        model_type = model.__class__.__name__

    return ModelInfoResponse(
        model_type=model_type,
        feature_names=FEATURE_COLUMNS,
        accepted_input_fields=list(TaxiTripRequest.model_fields),
        notebook_summary=NOTEBOOK_SUMMARY,
        training_constraints=TRAINING_CONSTRAINTS,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(trip: TaxiTripRequest) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    features = build_trip_features(trip)
    feature_frame = build_features_frame([trip])

    predicted_log_duration = float(model.predict(feature_frame)[0])
    predicted_duration_seconds = max(0.0, math.expm1(predicted_log_duration))
    predicted_duration_minutes = predicted_duration_seconds / 60.0
    estimated_dropoff_datetime = trip.pickup_datetime + timedelta(
        seconds=predicted_duration_seconds
    )

    return PredictionResponse(
        predicted_duration_seconds=round(predicted_duration_seconds, 2),
        predicted_duration_minutes=round(predicted_duration_minutes, 2),
        estimated_dropoff_datetime=estimated_dropoff_datetime,
        distance_km=round(features["distance"], 3),
        within_training_duration_range=60 <= predicted_duration_seconds <= 7200,
        engineered_features={
            feature_name: round(feature_value, 6)
            for feature_name, feature_value in features.items()
        },
    )
