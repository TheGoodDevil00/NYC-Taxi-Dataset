from __future__ import annotations

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

TRAINING_NOTE = (
    "This UI sends raw trip details to the FastAPI backend, which recreates the "
    "same feature engineering steps used in the training notebook."
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-top: #071619;
                --bg-mid: #0d2023;
                --bg-bottom: #13282c;
                --surface: rgba(14, 29, 33, 0.88);
                --surface-strong: rgba(10, 23, 26, 0.96);
                --border: rgba(153, 214, 201, 0.14);
                --text-primary: #edf7f3;
                --text-secondary: #bdd6cf;
                --accent-soft: rgba(132, 215, 182, 0.16);
                --accent-strong: #b9f2df;
            }
            .stApp {
                color: var(--text-primary);
                background:
                    radial-gradient(circle at top left, rgba(62, 176, 156, 0.22), transparent 34%),
                    radial-gradient(circle at top right, rgba(180, 148, 87, 0.12), transparent 28%),
                    linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 46%, var(--bg-bottom) 100%);
            }
            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }
            [data-testid="stSidebar"] > div:first-child {
                background:
                    linear-gradient(180deg, rgba(7, 22, 25, 0.98) 0%, rgba(12, 29, 33, 0.98) 100%);
                border-right: 1px solid var(--border);
            }
            [data-testid="stHeader"] {
                background: rgba(7, 22, 25, 0.28);
            }
            [data-testid="stToolbar"] {
                color: var(--text-secondary);
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
            .stApp p, .stApp li, .stApp span, .stApp small,
            .stApp [data-testid="stMarkdownContainer"] *,
            .stApp [data-testid="stCaptionContainer"] {
                color: var(--text-primary);
            }
            .hero-card, .result-card, .note-card {
                border-radius: 20px;
                padding: 1.2rem 1.3rem;
                background: linear-gradient(180deg, rgba(18, 38, 42, 0.96) 0%, rgba(11, 27, 30, 0.96) 100%);
                border: 1px solid var(--border);
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(8px);
            }
            .hero-card h1, .result-card h3 {
                margin: 0;
                color: var(--text-primary);
            }
            .hero-card p, .note-card p {
                margin-bottom: 0;
                color: var(--text-secondary);
            }
            .pill {
                display: inline-block;
                margin-top: 0.75rem;
                padding: 0.3rem 0.7rem;
                border-radius: 999px;
                background: var(--accent-soft);
                border: 1px solid rgba(185, 242, 223, 0.18);
                color: var(--accent-strong);
                font-size: 0.85rem;
                font-weight: 600;
            }
            .metric-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .metric-value {
                color: var(--text-primary);
                font-size: 2rem;
                font-weight: 700;
                line-height: 1.2;
            }
            .metric-subtle {
                color: var(--text-secondary);
                font-size: 0.95rem;
            }
            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
            }
            [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {
                color: var(--text-primary) !important;
            }
            [data-testid="stMetricDelta"] * {
                color: var(--text-secondary) !important;
            }
            [data-baseweb="input"] > div,
            [data-baseweb="select"] > div,
            .stDateInput input,
            .stTimeInput input,
            .stNumberInput input,
            .stTextInput input {
                background: rgba(8, 20, 23, 0.9) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--border) !important;
            }
            .stSlider [data-testid="stTickBarMin"],
            .stSlider [data-testid="stTickBarMax"] {
                background: rgba(189, 214, 207, 0.28);
            }
            [data-testid="stAlert"] {
                background: var(--surface-strong);
                color: var(--text-primary);
                border: 1px solid var(--border);
            }
            [data-testid="stExpander"] details {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 16px;
            }
            [data-testid="stExpander"] summary {
                color: var(--text-primary);
            }
            button[kind="primary"] {
                background: linear-gradient(135deg, #93e2c4 0%, #72c6b3 100%);
                color: #081517;
                border: none;
            }
            button[kind="primary"]:hover {
                color: #081517;
                filter: brightness(1.04);
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


def render_results(prediction: dict | None, payload: dict | None) -> None:
    st.subheader("Prediction")

    if not prediction or not payload:
        st.info("Enter a trip and click Estimate Trip Duration to see the result.")
        return

    training_badge = (
        "Inside training duration band"
        if prediction["within_training_duration_range"]
        else "Outside training duration band"
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="metric-label">Predicted trip duration</div>
            <div class="metric-value">{prediction["predicted_duration_minutes"]:.2f} min</div>
            <div class="metric-subtle">{prediction["predicted_duration_seconds"]:.0f} seconds</div>
            <div class="pill">{training_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Distance", f'{prediction["distance_km"]:.2f} km')
    metric_col_2.metric(
        "ETA",
        str(prediction["estimated_dropoff_datetime"]).replace("T", " "),
    )
    metric_col_3.metric("Passengers", str(payload["passenger_count"]))

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
    st.map(map_points, latitude="lat", longitude="lon", use_container_width=True)

    with st.expander("Engineered feature breakdown", expanded=False):
        feature_frame = (
            pd.DataFrame(
                prediction["engineered_features"].items(),
                columns=["feature", "value"],
            )
            .sort_values("feature")
            .reset_index(drop=True)
        )
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="NYC Taxi Trip Duration",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    initialize_state()

    with st.sidebar:
        st.markdown("## Connection")
        api_base_url = st.text_input("API base URL", value=API_BASE_URL_DEFAULT)
        st.caption("Point this at the FastAPI server running through uvicorn.")

        try:
            health = fetch_health(api_base_url)
            st.success(f'API status: {health["status"]}')
        except requests.RequestException as exc:
            health = None
            st.error(f"API unreachable: {exc}")

        st.markdown("## Model Notes")
        st.caption(TRAINING_NOTE)

    st.markdown(
        """
        <div class="hero-card">
            <h1>NYC Taxi Trip Duration Estimator</h1>
            <p>Forecast trip time from raw pickup details, passenger count, and route coordinates.</p>
            <div class="pill">FastAPI backend + Streamlit frontend</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model_info = fetch_model_info(api_base_url)
    except requests.RequestException:
        model_info = None

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric(
        "Model",
        model_info["model_type"] if model_info else "Unavailable",
    )
    summary_col_2.metric(
        "Notebook features",
        str(len(model_info["feature_names"])) if model_info else "--",
    )
    summary_col_3.metric(
        "Passenger range",
        model_info["training_constraints"]["passenger_count"]
        if model_info
        else "--",
    )

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.subheader("Trip Inputs")
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

        pickup_col, dropoff_col = st.columns(2)
        with pickup_col:
            st.markdown("#### Pickup")
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
            st.markdown("#### Dropoff")
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

        st.markdown(
            """
            <div class="note-card">
                <p>The notebook bins coordinates to 2 decimal places and uses cyclical encodings for hour, weekday, and month before scoring the trip.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prediction = st.session_state.get("last_prediction")
        payload = st.session_state.get("last_payload")

        if st.button("Estimate Trip Duration", type="primary", use_container_width=True):
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
                    prediction = predict_trip(api_base_url, payload)
                st.session_state.last_prediction = prediction
                st.session_state.last_payload = payload
            except requests.HTTPError as exc:
                prediction = None
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Prediction failed: {detail}")
            except requests.RequestException as exc:
                prediction = None
                st.error(f"Unable to reach the API: {exc}")

    with right_col:
        render_results(prediction, payload)

        if model_info:
            with st.expander("Notebook assessment", expanded=False):
                for bullet in model_info["notebook_summary"]:
                    st.write(f"- {bullet}")


if __name__ == "__main__":
    main()
