# NYC Taxi Trip Duration Demo

This project exposes the trained model from `src/taxidata.ipynb` through a FastAPI backend and a Streamlit frontend.

## What the notebook does

- Cleans the NYC taxi dataset and filters to 1-6 passenger trips.
- Engineers haversine distance, rounded coordinate bins, and cyclic time features.
- Trains an `XGBRegressor` inside a sklearn `Pipeline`.
- Uses a chronological 80/20 split and log-transformed target for trip duration.

## Run locally

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python app/run.py
```

Or run the services separately:

```bash
venv/bin/python -m uvicorn api.main:app --reload
venv/bin/python -m streamlit run app/main.py
```

## Deploy on Railway

This repo now includes a `railpack.json` for a single-service Railway deploy:

- Streamlit is served on Railway's public `PORT`.
- FastAPI runs inside the same container on `127.0.0.1:8000`.
- `.dockerignore` keeps the training notebook and raw dataset zip out of the deploy upload.

To deploy, create a Railway service from this repository and let Railway build it with Railpack. The service will start with:

```bash
python app/run.py
```

Once Railway assigns a public URL, open that URL and the Streamlit UI will call the colocated FastAPI backend automatically.
