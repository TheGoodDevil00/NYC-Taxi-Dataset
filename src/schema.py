from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


NYC_LATITUDE_MIN = 40.5
NYC_LATITUDE_MAX = 41.0
NYC_LONGITUDE_MIN = -74.5
NYC_LONGITUDE_MAX = -73.0


class TaxiTripRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pickup_datetime": "2016-06-15T08:30:00",
                "passenger_count": 2,
                "pickup_latitude": 40.758,
                "pickup_longitude": -73.9855,
                "dropoff_latitude": 40.7128,
                "dropoff_longitude": -74.006,
            }
        },
    )

    pickup_datetime: datetime = Field(
        ...,
        description="Pickup timestamp in local NYC time.",
    )
    passenger_count: int = Field(
        ...,
        ge=1,
        le=6,
        description="Passenger count used during model training.",
    )
    pickup_latitude: float = Field(..., ge=NYC_LATITUDE_MIN, le=NYC_LATITUDE_MAX)
    pickup_longitude: float = Field(..., ge=NYC_LONGITUDE_MIN, le=NYC_LONGITUDE_MAX)
    dropoff_latitude: float = Field(..., ge=NYC_LATITUDE_MIN, le=NYC_LATITUDE_MAX)
    dropoff_longitude: float = Field(..., ge=NYC_LONGITUDE_MIN, le=NYC_LONGITUDE_MAX)

    @model_validator(mode="after")
    def validate_route(self) -> "TaxiTripRequest":
        same_pickup_and_dropoff = (
            self.pickup_latitude == self.dropoff_latitude
            and self.pickup_longitude == self.dropoff_longitude
        )
        if same_pickup_and_dropoff:
            raise ValueError("Pickup and dropoff coordinates must be different.")
        return self


class PredictionResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    estimated_dropoff_datetime: datetime
    distance_km: float
    within_training_duration_range: bool
    engineered_features: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_names: list[str]
    accepted_input_fields: list[str]
    notebook_summary: list[str]
    training_constraints: dict[str, Any]
