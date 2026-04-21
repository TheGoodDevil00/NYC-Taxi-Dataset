from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from src.schema import TaxiTripRequest

FEATURE_COLUMNS = [
    "passenger_count",
    "distance",
    "pickup_latitude_bin",
    "pickup_longitude_bin",
    "dropoff_latitude_bin",
    "dropoff_longitude_bin",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "distance_hour",
    "distance_dow",
    "distance_month",
]

NOTEBOOK_SUMMARY = [
    "The notebook removes missing values and filters trips to 1-6 passengers.",
    "It engineers haversine distance, rounded coordinate bins, and cyclic time features.",
    "The trained estimator is an XGBoost regressor wrapped in a sklearn Pipeline.",
    "Training keeps trips between 60 and 7200 seconds and uses a chronological 80/20 split.",
]

TRAINING_CONSTRAINTS = {
    "passenger_count": "1 to 6 passengers",
    "geography": "NYC-like coordinates within latitude 40.5-41.0 and longitude -74.5 to -73.0",
    "duration_window_seconds": "60 to 7200 seconds in the training set",
    "feature_count": len(FEATURE_COLUMNS),
}


def haversine_distance_km(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
) -> float:
    radius_km = 6371.0

    pickup_latitude_rad = math.radians(pickup_latitude)
    pickup_longitude_rad = math.radians(pickup_longitude)
    dropoff_latitude_rad = math.radians(dropoff_latitude)
    dropoff_longitude_rad = math.radians(dropoff_longitude)

    delta_latitude = dropoff_latitude_rad - pickup_latitude_rad
    delta_longitude = dropoff_longitude_rad - pickup_longitude_rad

    haversine_term = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(pickup_latitude_rad)
        * math.cos(dropoff_latitude_rad)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(haversine_term))


def build_trip_features(trip: TaxiTripRequest) -> dict[str, float]:
    pickup_datetime = trip.pickup_datetime
    pickup_hour = pickup_datetime.hour
    pickup_day_of_week = pickup_datetime.weekday()
    pickup_month = pickup_datetime.month

    distance_km = haversine_distance_km(
        pickup_latitude=trip.pickup_latitude,
        pickup_longitude=trip.pickup_longitude,
        dropoff_latitude=trip.dropoff_latitude,
        dropoff_longitude=trip.dropoff_longitude,
    )

    hour_sin = math.sin(2 * math.pi * pickup_hour / 24)
    hour_cos = math.cos(2 * math.pi * pickup_hour / 24)
    dow_sin = math.sin(2 * math.pi * pickup_day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * pickup_day_of_week / 7)
    month_sin = math.sin(2 * math.pi * pickup_month / 12)
    month_cos = math.cos(2 * math.pi * pickup_month / 12)

    return {
        "passenger_count": float(trip.passenger_count),
        "distance": distance_km,
        "pickup_latitude_bin": round(trip.pickup_latitude, 2),
        "pickup_longitude_bin": round(trip.pickup_longitude, 2),
        "dropoff_latitude_bin": round(trip.dropoff_latitude, 2),
        "dropoff_longitude_bin": round(trip.dropoff_longitude, 2),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "distance_hour": distance_km * hour_sin,
        "distance_dow": distance_km * dow_sin,
        "distance_month": distance_km * month_sin,
    }


def build_features_frame(trips: Iterable[TaxiTripRequest]) -> pd.DataFrame:
    rows = []
    for trip in trips:
        features = build_trip_features(trip)
        rows.append([features[column] for column in FEATURE_COLUMNS])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
