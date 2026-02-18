from fastapi import FastAPI, BackgroundTasks, HTTPException
import pandas as pd
import requests
import numpy as np
import os
import io
import json
import time
from prophet import Prophet
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime
from typing import Optional

app = FastAPI(title="Climate Monitor API & Forecast Engine")

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Mehrdat/weapi-store"
API_ROOT = "https://climatemonitor.info/api/public/v1"

FACTORS = {
    "carbon_dioxide": {"endpoint": "/co2/daily", "fallback": "/co2/monthly_gl", "bad_direction": "up", "unit": "ppm"},
    "methane": {"endpoint": "/ch4/monthly", "fallback": "/ch4/annual", "bad_direction": "up", "unit": "ppb"},
    "nitrous_oxide": {"endpoint": "/n2o/monthly", "fallback": "/n2o/annual", "bad_direction": "up", "unit": "ppb"},
    "sulfur_hexafluoride": {"endpoint": "/sf6/monthly", "fallback": "/sf6/annual", "bad_direction": "up", "unit": "ppt"},
    "temperature_anomaly": {"endpoint": "/temp/daily_avg", "fallback": "/temp/monthly_anomaly_avg", "bad_direction": "up", "unit": "°C"},
    "ocean_level": {"endpoint": "/ocean/level", "fallback": None, "bad_direction": "up", "unit": "mm"},
    "glaciers_greenland": {"endpoint": "/glaciers/greenland", "fallback": None, "bad_direction": "down", "unit": "Gt"},
    "glaciers_antarctica": {"endpoint": "/glaciers/antarctica", "fallback": None, "bad_direction": "down", "unit": "Gt"}
}

# --- Helper functions from daily_engine.py ---

def fetch_data(factor_name, config):
    url = f"{API_ROOT}{config['endpoint']}"
    try:
        r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code != 200 and config["fallback"]:
            url = f"{API_ROOT}{config['fallback']}"
            r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        
        data = r.json().get("data", {})
        readings = data.get("readings", []) if isinstance(data, dict) and "readings" in data else data
        if not readings:
            return None
        
        df = pd.DataFrame(readings)
        date_col = "label" if "label" in df.columns else "date"
        if date_col not in df.columns: return None
        
        df["ds"] = pd.to_datetime(df[date_col])
        df["y"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["ds", "y"]).sort_values("ds")[["ds", "y"]]
    except:
        return None

def run_ml_forecast(df):
    if len(df) < 20: return None
    last_gap = (df.iloc[-1]["ds"] - df.iloc[-2]["ds"]).days
    freq = "D" if last_gap < 5 else "M"
    
    model = Prophet(daily_seasonality=(freq == "D"), yearly_seasonality=True, weekly_seasonality=(freq == "D"))
    try:
        model.fit(df)
        future = model.make_future_dataframe(periods=1, freq=freq)
        forecast = model.predict(future)
        return forecast.tail(1)
    except:
        return None

def calculate_index(current, forecast, bad_direction):
    if forecast is None or forecast.empty: return 0
    predicted = forecast.iloc[0]["yhat"]
    diff = current - predicted
    denominator = abs(predicted) * 0.5 if abs(predicted) > 0 else 1
    raw_score = (diff / denominator) * 10
    final_score = -raw_score if bad_direction == "up" else raw_score
    return max(min(int(final_score), 10), -10)

def generate_full_report():
    today_str = datetime.now().strftime("%Y-%m-%d")
    report = {"date": today_str, "generated_at": datetime.now().isoformat(), "factors": {}}
    total_score, count = 0, 0

    for name, config in FACTORS.items():
        df = fetch_data(name, config)
        if df is None or df.empty: continue
        
        latest = df.iloc[-1]
        forecast = run_ml_forecast(df)
        index = calculate_index(latest['y'], forecast, config["bad_direction"])
        
        report["factors"][name] = {
            "current": round(latest['y'], 2),
            "unit": config['unit'],
            "index": index,
            "status": "Better" if index > 2 else ("Worse" if index < -2 else "Expected")
        }
        total_score += index
        count += 1
    
    report["global_score"] = round(total_score / count, 2) if count > 0 else 0
    return report

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "message": "Climate Monitor Forecast Engine", "endpoints": ["/latest", "/generate", "/factor/{name}"]}

@app.get("/latest")
def get_latest_report():
    """Fetches the pre-calculated report from Hub."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        path = hf_hub_download(repo_id=DATASET_REPO, filename=f"data/{today}.json", repo_type="dataset")
        with open(path, "r") as f:
            return json.load(f)
    except:
        raise HTTPException(status_code=404, detail="Today's report not generated yet.")

@app.get("/generate")
def trigger_generation(background_tasks: BackgroundTasks):
    """Generates the report live (slow) and optionally uploads in background."""
    report = generate_full_report()
    # You could add background_tasks.add_task(upload_to_hf, report) here
    return report

@app.get("/factor/{name}")
def get_factor_status(name: str):
    if name not in FACTORS:
        raise HTTPException(status_code=404, detail="Factor not found")
    
    df = fetch_data(name, FACTORS[name])
    if df is None:
        raise HTTPException(status_code=503, detail="Could not fetch data")
        
    latest = df.iloc[-1]
    forecast = run_ml_forecast(df)
    index = calculate_index(latest['y'], forecast, FACTORS[name]["bad_direction"])
    
    return {
        "factor": name,
        "current_value": latest['y'],
        "predicted_value": forecast.iloc[0]["yhat"] if forecast is not None else None,
        "index_score": index
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
