from __future__ import annotations

import math
import os
from datetime import date, datetime, time

import pandas as pd
import requests
import streamlit as st

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

ROUTE_PRESETS = {
    "Midtown to Wall Street": {
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7060,
        "dropoff_longitude": -74.0086,
    },
    "JFK to Times Square": {
        "pickup_latitude": 40.6413,
        "pickup_longitude": -73.7781,
        "dropoff_latitude": 40.7580,
        "dropoff_longitude": -73.9855,
    },
    "Upper West Side to Brooklyn": {
        "pickup_latitude": 40.7870,
        "pickup_longitude": -73.9754,
        "dropoff_latitude": 40.6782,
        "dropoff_longitude": -73.9442,
    },
}

ROUTE_META = {
    "Midtown to Wall Street": {
        "pickup": "Times Square corridor",
        "dropoff": "Lower Manhattan",
        "tempo": "weekday commuter",
    },
    "JFK to Times Square": {
        "pickup": "Airport terminal zone",
        "dropoff": "Midtown hotel zone",
        "tempo": "airport transfer",
    },
    "Upper West Side to Brooklyn": {
        "pickup": "Upper West Side",
        "dropoff": "Central Brooklyn",
        "tempo": "cross-borough",
    },
}

TRAINING_NOTE = (
    "The UI sends raw trip details to FastAPI. The backend recreates the notebook "
    "feature pipeline before scoring with the saved model."
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --page: #f5f6f1;
                --paper: #ffffff;
                --ink: #171713;
                --muted: #696b61;
                --line: #d9d8ce;
                --line-strong: #bdbbac;
                --taxi: #f5c400;
                --taxi-strong: #d99b00;
                --green: #297d63;
                --red: #9e3f30;
                --charcoal: #23231f;
                --charcoal-2: #303028;
                --shadow: 0 20px 55px rgba(32, 31, 24, 0.12);
                --soft-shadow: 0 10px 30px rgba(32, 31, 24, 0.08);
                --radius: 8px;
            }

            .stApp {
                color: var(--ink);
                background:
                    linear-gradient(90deg, rgba(23, 23, 19, 0.035) 1px, transparent 1px) 0 0 / 56px 56px,
                    linear-gradient(180deg, #fbfbf8 0%, var(--page) 42%, #eceee6 100%);
                font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }

            [data-testid="stHeader"] {
                background: rgba(245, 246, 241, 0.82);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(23, 23, 19, 0.08);
            }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            #MainMenu,
            footer {
                visibility: hidden;
            }

            [data-testid="block-container"] {
                max-width: 1360px;
                padding: 2.25rem 2.25rem 4rem;
            }

            [data-testid="stSidebar"] > div:first-child {
                background: var(--charcoal);
                border-right: 0;
                box-shadow: 16px 0 42px rgba(25, 24, 19, 0.16);
                padding-top: 1.2rem;
            }

            [data-testid="stSidebar"] * {
                color: #f7f4e8;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label {
                color: #f7f4e8 !important;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: #c9c5b5 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="input"] > div,
            [data-testid="stSidebar"] .stTextInput input {
                background: #151512 !important;
                border: 1px solid rgba(245, 196, 0, 0.34) !important;
                color: #fff9dc !important;
                border-radius: var(--radius) !important;
            }

            .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
                color: var(--ink);
                letter-spacing: 0;
            }

            .stApp h1 {
                font-size: clamp(2.4rem, 5vw, 5.7rem);
                line-height: 0.93;
                font-weight: 850;
                margin: 0;
            }

            .stApp h2 {
                font-size: 1.4rem;
                line-height: 1.12;
                font-weight: 800;
                margin-top: 0;
            }

            .stApp h3 {
                font-size: 1rem;
                line-height: 1.25;
                font-weight: 780;
                margin-top: 0;
            }

            .stApp p,
            .stApp li,
            .stApp label,
            .stApp [data-testid="stCaptionContainer"] {
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.55;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1.1rem;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                color: var(--ink);
                font-size: 0.83rem;
                font-weight: 820;
                text-transform: uppercase;
            }

            .brand-mark {
                width: 34px;
                height: 34px;
                border-radius: 7px;
                background:
                    linear-gradient(90deg, rgba(23, 23, 19, 0.18) 1px, transparent 1px) 0 0 / 8px 8px,
                    var(--taxi);
                border: 1px solid rgba(23, 23, 19, 0.22);
                box-shadow: inset 0 -4px 0 rgba(23, 23, 19, 0.12);
            }

            .top-actions {
                display: flex;
                gap: 0.45rem;
                align-items: center;
                flex-wrap: wrap;
            }

            .nav-chip {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.45rem 0.75rem;
                background: rgba(255, 255, 255, 0.72);
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .hero {
                position: relative;
                overflow: hidden;
                min-height: 350px;
                margin-bottom: 1.35rem;
                border-radius: var(--radius);
                border: 1px solid var(--line);
                background:
                    linear-gradient(110deg, rgba(255, 255, 255, 0.96) 0%, rgba(255, 255, 255, 0.92) 46%, rgba(245, 196, 0, 0.18) 100%),
                    linear-gradient(90deg, rgba(23, 23, 19, 0.055) 1px, transparent 1px) 0 0 / 48px 48px;
                box-shadow: var(--shadow);
                padding: clamp(1.35rem, 3vw, 2.4rem);
            }

            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.02fr) minmax(310px, 0.72fr);
                gap: clamp(1.2rem, 3vw, 2.4rem);
                align-items: stretch;
            }

            .hero-copy {
                display: flex;
                min-height: 292px;
                flex-direction: column;
                justify-content: space-between;
                gap: 2rem;
            }

            .hero-copy p {
                max-width: 640px;
                margin: 1.1rem 0 0;
                color: #484940;
                font-size: 1.08rem;
            }

            .hero-metrics {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                max-width: 760px;
            }

            .hero-stat,
            .panel,
            .result-panel,
            .model-panel,
            .input-zone {
                border-radius: var(--radius);
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.92);
                box-shadow: var(--soft-shadow);
            }

            .hero-stat {
                min-height: 86px;
                padding: 0.88rem;
            }

            .hero-stat .value {
                display: block;
                color: var(--ink);
                font-size: 1.18rem;
                font-weight: 850;
                line-height: 1.15;
                overflow-wrap: anywhere;
            }

            .hero-stat .label {
                display: block;
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .route-board {
                min-height: 292px;
                border-radius: var(--radius);
                background: var(--charcoal);
                color: #fff8d6;
                padding: 1.15rem;
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(23, 23, 19, 0.24);
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
            }

            .route-board::before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(90deg, rgba(245, 196, 0, 0.09) 1px, transparent 1px) 0 0 / 40px 40px,
                    linear-gradient(0deg, rgba(245, 196, 0, 0.08) 1px, transparent 1px) 0 0 / 40px 40px;
                pointer-events: none;
            }

            .route-board > * {
                position: relative;
                z-index: 1;
            }

            .route-kicker {
                display: flex;
                justify-content: space-between;
                gap: 0.7rem;
                color: #e8dd9a;
                font-size: 0.72rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .route-line {
                display: grid;
                grid-template-columns: 22px 1fr;
                gap: 0.72rem;
                margin-top: 1.15rem;
            }

            .route-track {
                display: flex;
                align-items: center;
                flex-direction: column;
            }

            .route-dot {
                width: 14px;
                height: 14px;
                border-radius: 999px;
                background: var(--taxi);
                box-shadow: 0 0 0 5px rgba(245, 196, 0, 0.16);
            }

            .route-stem {
                width: 2px;
                min-height: 96px;
                flex: 1;
                margin: 0.45rem 0;
                background: repeating-linear-gradient(180deg, var(--taxi), var(--taxi) 8px, transparent 8px, transparent 14px);
            }

            .route-stop {
                padding-bottom: 1rem;
            }

            .route-stop:last-child {
                padding-bottom: 0;
            }

            .route-stop strong {
                display: block;
                color: #fff6c7;
                font-size: 1rem;
                line-height: 1.25;
            }

            .route-stop span {
                display: block;
                margin-top: 0.18rem;
                color: #c9c5b5;
                font-size: 0.82rem;
            }

            .status-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.7rem;
                margin-bottom: 1.2rem;
            }

            .status-card {
                min-height: 96px;
                padding: 0.95rem;
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: var(--paper);
                box-shadow: var(--soft-shadow);
            }

            .status-card .label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 780;
                text-transform: uppercase;
            }

            .status-card .value {
                display: block;
                margin-top: 0.5rem;
                color: var(--ink);
                font-size: 1.1rem;
                font-weight: 850;
                line-height: 1.18;
            }

            .input-zone,
            .result-panel,
            .model-panel,
            .panel {
                padding: 1.05rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--line) !important;
                border-radius: var(--radius) !important;
                background: rgba(255, 255, 255, 0.92) !important;
                box-shadow: var(--soft-shadow);
            }

            .section-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: 0 0 0.8rem;
            }

            .section-title h2,
            .section-title h3 {
                margin: 0;
            }

            .section-code {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .input-zone [data-baseweb="select"] > div,
            .input-zone [data-baseweb="input"] > div,
            .input-zone .stDateInput input,
            .input-zone .stTimeInput input,
            .input-zone .stNumberInput input,
            .stTextInput input {
                min-height: 44px;
                border: 1px solid var(--line-strong) !important;
                border-radius: var(--radius) !important;
                background: #fffef9 !important;
                color: var(--ink) !important;
                box-shadow: none !important;
            }

            .input-zone label,
            .stSlider label {
                color: var(--ink) !important;
                font-size: 0.78rem !important;
                font-weight: 780 !important;
                text-transform: uppercase;
            }

            .stSlider [data-baseweb="slider"] > div {
                color: var(--taxi-strong);
            }

            div[data-testid="stButton"] > button {
                min-height: 48px;
                border-radius: var(--radius);
                border: 1px solid #1e1e1a;
                background: var(--ink) !important;
                color: #fff4bb !important;
                font-size: 0.88rem;
                font-weight: 840;
                text-transform: uppercase;
                box-shadow: 0 12px 24px rgba(23, 23, 19, 0.18);
            }

            div[data-testid="stButton"] > button * {
                color: #fff4bb !important;
            }

            div[data-testid="stButton"] > button:hover {
                border-color: #000000;
                background: #050504;
                color: var(--taxi);
            }

            .result-hero {
                background: var(--charcoal);
                border-radius: var(--radius);
                color: #fff8d6;
                padding: 1.1rem;
                border: 1px solid rgba(23, 23, 19, 0.2);
            }

            .result-hero .label {
                color: #d8cf96;
                font-size: 0.72rem;
                font-weight: 760;
                text-transform: uppercase;
            }

            .result-hero .value {
                display: block;
                margin-top: 0.35rem;
                color: #fff9d9;
                font-size: clamp(2.2rem, 5vw, 4.4rem);
                font-weight: 880;
                line-height: 0.95;
            }

            .result-hero .sub {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.9rem;
                color: #d6d0ad;
                font-size: 0.86rem;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                width: fit-content;
                min-height: 28px;
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                border: 1px solid rgba(23, 23, 19, 0.14);
                background: var(--taxi);
                color: var(--ink);
                font-size: 0.73rem;
                font-weight: 820;
                text-transform: uppercase;
            }

            .badge.soft {
                background: #edf0e7;
                color: var(--muted);
            }

            .badge.green {
                background: #dbece5;
                color: var(--green);
            }

            .badge.red {
                background: #f1dfdb;
                color: var(--red);
            }

            .mini-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 0.8rem;
            }

            .mini-card {
                min-height: 84px;
                padding: 0.85rem;
                border-radius: var(--radius);
                border: 1px solid var(--line);
                background: #fbfbf7;
            }

            .mini-card span {
                color: var(--muted);
                font-size: 0.7rem;
                font-weight: 780;
                text-transform: uppercase;
            }

            .mini-card strong {
                display: block;
                margin-top: 0.35rem;
                color: var(--ink);
                font-size: 0.98rem;
                line-height: 1.15;
            }

            .empty-state {
                min-height: 310px;
                display: grid;
                place-items: center;
                border-radius: var(--radius);
                border: 1px dashed var(--line-strong);
                background:
                    linear-gradient(90deg, rgba(23, 23, 19, 0.04) 1px, transparent 1px) 0 0 / 36px 36px,
                    #fbfbf7;
                text-align: center;
                padding: 1.4rem;
            }

            .empty-state strong {
                display: block;
                color: var(--ink);
                font-size: 1.2rem;
                line-height: 1.2;
            }

            .empty-state span {
                display: block;
                margin-top: 0.55rem;
                color: var(--muted);
                font-size: 0.92rem;
                max-width: 360px;
            }

            [data-testid="stMetric"] {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: #fbfbf7;
                padding: 0.78rem;
                box-shadow: none;
            }

            [data-testid="stMetricLabel"] * {
                color: var(--muted) !important;
                font-size: 0.74rem !important;
                font-weight: 780 !important;
                text-transform: uppercase;
            }

            [data-testid="stMetricValue"] * {
                color: var(--ink) !important;
                font-size: 1.1rem !important;
                font-weight: 850 !important;
            }

            [data-testid="stAlert"] {
                border-radius: var(--radius);
                border: 1px solid var(--line);
                background: #fff9d8;
                color: var(--ink);
            }

            [data-testid="stExpander"] details {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: var(--paper);
                box-shadow: none;
            }

            [data-testid="stExpander"] summary {
                color: var(--ink);
                font-weight: 780;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                overflow: hidden;
            }

            .map-note {
                margin: 0.8rem 0 0;
                color: var(--muted);
                font-size: 0.8rem;
            }

            hr {
                border-color: var(--line);
                margin: 1.2rem 0;
            }

            @media (max-width: 980px) {
                [data-testid="block-container"] {
                    padding: 1.25rem 1rem 3rem;
                }

                .hero-grid,
                .status-strip {
                    grid-template-columns: 1fr;
                }

                .hero-metrics,
                .mini-grid {
                    grid-template-columns: 1fr;
                }

                .topbar {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .hero-copy {
                    min-height: 0;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=3)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_info(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url.rstrip('/')}/model/info", timeout=3)
    response.raise_for_status()
    return response.json()


def predict_trip(api_base_url: str, payload: dict) -> dict:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/predict",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


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


def apply_preset() -> None:
    preset = ROUTE_PRESETS[st.session_state.route_preset]
    for field_name, value in preset.items():
        st.session_state[field_name] = value


def initialize_state() -> None:
    default_preset = "Midtown to Wall Street"
    if "route_preset" not in st.session_state:
        st.session_state.route_preset = default_preset

    if "trip_date" not in st.session_state:
        st.session_state.trip_date = date.today()

    if "trip_time" not in st.session_state:
        st.session_state.trip_time = time(8, 30)

    for field_name, value in ROUTE_PRESETS[default_preset].items():
        st.session_state.setdefault(field_name, value)

    st.session_state.setdefault("passenger_count", 2)


def render_topbar() -> None:
    st.html(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark"></div>
                <span>NYC Taxi Duration Lab</span>
            </div>
            <div class="top-actions">
                <span class="nav-chip">FastAPI scoring</span>
                <span class="nav-chip">Notebook features</span>
                <span class="nav-chip">Streamlit console</span>
            </div>
        </div>
        """
    )


def route_board_html(route_name: str) -> str:
    route_meta = ROUTE_META[route_name]
    preset = ROUTE_PRESETS[route_name]
    preview_distance = haversine_distance_km(
        preset["pickup_latitude"],
        preset["pickup_longitude"],
        preset["dropoff_latitude"],
        preset["dropoff_longitude"],
    )
    return f"""
        <div class="route-board">
            <div class="route-kicker">
                <span>Selected route</span>
                <span>{preview_distance:.1f} km direct</span>
            </div>
            <div class="route-line">
                <div class="route-track">
                    <div class="route-dot"></div>
                    <div class="route-stem"></div>
                    <div class="route-dot"></div>
                </div>
                <div>
                    <div class="route-stop">
                        <strong>{route_meta["pickup"]}</strong>
                        <span>{preset["pickup_latitude"]:.4f}, {preset["pickup_longitude"]:.4f}</span>
                    </div>
                    <div class="route-stop">
                        <strong>{route_meta["dropoff"]}</strong>
                        <span>{preset["dropoff_latitude"]:.4f}, {preset["dropoff_longitude"]:.4f}</span>
                    </div>
                </div>
            </div>
            <span class="badge">{route_meta["tempo"]}</span>
        </div>
        """


def render_hero(model_info: dict | None) -> None:
    model_type = model_info["model_type"] if model_info else "Model pending"
    model_display = "XGB pipeline" if "XGBRegressor" in model_type else model_type
    feature_count = len(model_info["feature_names"]) if model_info else "--"
    passenger_range = (
        model_info["training_constraints"]["passenger_count"] if model_info else "--"
    )
    route_board = route_board_html(st.session_state.route_preset)

    st.html(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div>
                        <h1>Taxi ETA, scored like dispatch.</h1>
                        <p>Estimate trip duration from pickup time, passenger count, and two NYC coordinate points using the same engineered features from the training notebook.</p>
                    </div>
                    <div class="hero-metrics">
                        <div class="hero-stat">
                            <span class="value">{model_display}</span>
                            <span class="label">Active estimator</span>
                        </div>
                        <div class="hero-stat">
                            <span class="value">{feature_count}</span>
                            <span class="label">Model features</span>
                        </div>
                        <div class="hero-stat">
                            <span class="value">{passenger_range}</span>
                            <span class="label">Passenger range</span>
                        </div>
                    </div>
                </div>
                {route_board}
            </div>
        </div>
        """
    )


def render_status_strip(health: dict | None, model_info: dict | None) -> None:
    api_status = "Online" if health and health.get("status") == "ok" else "Check API"
    model_loaded = "Loaded" if health and health.get("model_loaded") else "Unknown"
    duration_window = (
        model_info["training_constraints"]["duration_window_seconds"]
        if model_info
        else "Unavailable"
    )
    geography = (
        model_info["training_constraints"]["geography"] if model_info else "Unavailable"
    )
    st.html(
        f"""
        <div class="status-strip">
            <div class="status-card">
                <span class="label">API status</span>
                <span class="value">{api_status}</span>
            </div>
            <div class="status-card">
                <span class="label">Model file</span>
                <span class="value">{model_loaded}</span>
            </div>
            <div class="status-card">
                <span class="label">Duration band</span>
                <span class="value">{duration_window}</span>
            </div>
            <div class="status-card">
                <span class="label">Geography</span>
                <span class="value">{geography}</span>
            </div>
        </div>
        """
    )


def render_input_panel() -> tuple[dict | None, dict | None]:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">
                <h2>Trip setup</h2>
                <span class="section-code">request payload</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.selectbox(
            "Route preset",
            options=list(ROUTE_PRESETS),
            key="route_preset",
            on_change=apply_preset,
        )

        time_col_1, time_col_2 = st.columns(2)
        time_col_1.date_input("Pickup date", key="trip_date")
        time_col_2.time_input("Pickup time", key="trip_time", step=1800)

        st.slider(
            "Passenger count",
            min_value=1,
            max_value=6,
            key="passenger_count",
        )

        pickup_col, dropoff_col = st.columns(2, gap="medium")
        with pickup_col:
            st.markdown("### Pickup")
            st.number_input(
                "Latitude",
                key="pickup_latitude",
                format="%.5f",
                min_value=40.5,
                max_value=41.0,
            )
            st.number_input(
                "Longitude",
                key="pickup_longitude",
                format="%.5f",
                min_value=-74.5,
                max_value=-73.0,
            )

        with dropoff_col:
            st.markdown("### Dropoff")
            st.number_input(
                "Latitude ",
                key="dropoff_latitude",
                format="%.5f",
                min_value=40.5,
                max_value=41.0,
            )
            st.number_input(
                "Longitude ",
                key="dropoff_longitude",
                format="%.5f",
                min_value=-74.5,
                max_value=-73.0,
            )

        prediction = st.session_state.get("last_prediction")
        payload = st.session_state.get("last_payload")

        if st.button("Estimate trip duration", type="primary", width="stretch"):
            pickup_datetime = datetime.combine(
                st.session_state.trip_date,
                st.session_state.trip_time,
            )
            payload = {
                "pickup_datetime": pickup_datetime.isoformat(),
                "passenger_count": int(st.session_state.passenger_count),
                "pickup_latitude": float(st.session_state.pickup_latitude),
                "pickup_longitude": float(st.session_state.pickup_longitude),
                "dropoff_latitude": float(st.session_state.dropoff_latitude),
                "dropoff_longitude": float(st.session_state.dropoff_longitude),
            }
            try:
                with st.spinner("Calling the model API..."):
                    prediction = predict_trip(st.session_state.api_base_url, payload)
                st.session_state.last_prediction = prediction
                st.session_state.last_payload = payload
            except requests.HTTPError as exc:
                prediction = None
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Prediction failed: {detail}")
            except requests.RequestException as exc:
                prediction = None
                st.error(f"Unable to reach the API: {exc}")

    return prediction, payload


def render_result_panel(prediction: dict | None, payload: dict | None) -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">
                <h2>Dispatch intelligence</h2>
                <span class="section-code">prediction response</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not prediction or not payload:
            st.markdown(
                """
                <div class="empty-state">
                    <div>
                        <strong>No scored trip yet</strong>
                        <span>Choose a preset or tune the coordinates, then run the estimator to see duration, ETA, distance, and engineered features.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        training_badge_class = (
            "green" if prediction["within_training_duration_range"] else "red"
        )
        training_badge = (
            "Inside training band"
            if prediction["within_training_duration_range"]
            else "Outside training band"
        )
        eta = str(prediction["estimated_dropoff_datetime"]).replace("T", " ")

        st.markdown(
            f"""
            <div class="result-hero">
                <span class="label">Predicted trip duration</span>
                <span class="value">{prediction["predicted_duration_minutes"]:.2f} min</span>
                <div class="sub">
                    <span>{prediction["predicted_duration_seconds"]:.0f} seconds</span>
                    <span class="badge {training_badge_class}">{training_badge}</span>
                </div>
            </div>
            <div class="mini-grid">
                <div class="mini-card">
                    <span>Distance</span>
                    <strong>{prediction["distance_km"]:.2f} km</strong>
                </div>
                <div class="mini-card">
                    <span>ETA</span>
                    <strong>{eta}</strong>
                </div>
                <div class="mini-card">
                    <span>Passengers</span>
                    <strong>{payload["passenger_count"]}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        map_points = pd.DataFrame(
            [
                {
                    "label": "Pickup",
                    "lat": payload["pickup_latitude"],
                    "lon": payload["pickup_longitude"],
                },
                {
                    "label": "Dropoff",
                    "lat": payload["dropoff_latitude"],
                    "lon": payload["dropoff_longitude"],
                },
            ]
        )
        st.map(
            map_points,
            latitude="lat",
            longitude="lon",
            width="stretch",
            height=320,
        )
        st.markdown(
            '<p class="map-note">Map pins use the exact payload coordinates sent to the backend.</p>',
            unsafe_allow_html=True,
        )

        with st.expander("Engineered feature breakdown", expanded=False):
            feature_frame = (
                pd.DataFrame(
                    prediction["engineered_features"].items(),
                    columns=["feature", "value"],
                )
                .sort_values("feature")
                .reset_index(drop=True)
            )
            st.dataframe(feature_frame, width="stretch", hide_index=True)


def render_model_panel(model_info: dict | None) -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">
                <h2>Notebook provenance</h2>
                <span class="section-code">model info endpoint</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="panel">
                <h3>Feature pipeline</h3>
                <p>{TRAINING_NOTE}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not model_info:
            st.info("Model metadata is unavailable until the API responds.")
            return

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Accepted fields", len(model_info["accepted_input_fields"]))
        metric_col_2.metric("Feature columns", len(model_info["feature_names"]))
        metric_col_3.metric(
            "Passenger range",
            model_info["training_constraints"]["passenger_count"],
        )

        with st.expander("Notebook assessment", expanded=False):
            for bullet in model_info["notebook_summary"]:
                st.write(f"- {bullet}")

        with st.expander("Accepted input fields", expanded=False):
            fields_frame = pd.DataFrame(
                {"field": model_info["accepted_input_fields"]}
            )
            st.dataframe(fields_frame, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="NYC Taxi Trip Duration",
        page_icon="NYC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    initialize_state()

    with st.sidebar:
        st.markdown("## API connection")
        st.session_state.api_base_url = st.text_input(
            "API base URL",
            value=API_BASE_URL_DEFAULT,
        )
        st.caption("Used by /health, /model/info, and /predict.")

        try:
            health = fetch_health(st.session_state.api_base_url)
            st.success(f'API status: {health["status"]}')
        except requests.RequestException as exc:
            health = None
            st.error(f"API unreachable: {exc}")

        st.markdown("## Operating note")
        st.caption(TRAINING_NOTE)

    try:
        model_info = fetch_model_info(st.session_state.api_base_url)
    except requests.RequestException:
        model_info = None

    render_topbar()
    render_hero(model_info)
    render_status_strip(health, model_info)

    left_col, right_col = st.columns([0.95, 1.05], gap="large")
    with left_col:
        prediction, payload = render_input_panel()
    with right_col:
        render_result_panel(prediction, payload)

    render_model_panel(model_info)


if __name__ == "__main__":
    main()
