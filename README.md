---
title: Climate Monitor API
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 🌍 Climate Monitor API & Forecast Engine

This is a **FastAPI** application designed to track, analyze, and forecast key climate indicators like CO2 levels, Methane, Global Temperature Anomalies, and Ocean Levels.

It powers the [ClimateMonitor.info](https://climatemonitor.info) platform and stores daily snapshots in the [Hugging Face Dataset](https://huggingface.co/datasets/Mehrdat/weapi-store).

## 🚀 Features

- **Live Data Fetching**: Pulls real-time data from scientific sources (NOAA, NASA, etc.).
- **ML Forecasting**: Uses **Facebook Prophet** to predict trends for the next 7 days.
- **Health Index**: Calculates a "WEAPI Index" (-10 to +10) to score specific climate factors against expected trends.
- **Hugging Face Integration**: Automatically uploads daily JSON reports to a persistent dataset.

## 📡 API Endpoints

The API is served at `https://huggingface.co/spaces/mehrdat/weapi` (replace with your actual URL).

### 1. Get Latest Snapshot
**`GET /latest`**  
Returns the most recent daily report stored in the dataset. This is the fastest way to get data.

```json
{
  "date": "2026-02-18",
  "global_climate_score": -2.5,
  "factors": {
    "carbon_dioxide": { "current": 425.1, "status": "Worse than expected" }
  }
}
```

### 2. Get Live Forecast
**`GET /generate`**  
Triggers a real-time fetch and ML prediction cycle.
*Warning: This takes 15-30 seconds to run.*

### 3. Check Specific Factor
**`GET /factor/{name}`**  
Get data for a single factor.  
**Example:** `/factor/methane`

**Available Factors:**
- `carbon_dioxide`
- `methane`
- `nitrous_oxide`
- `temperature_anomaly`
- `ocean_level`
- `glaciers_greenland`

## 🛠️ Local Development

1. **Clone the repo**
   ```bash
   git clone https://huggingface.co/spaces/mehrdat/weapi
   cd weapi
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   You need a Hugging Face Token with write access to upload data.
   ```bash
   export HF_TOKEN="hf_..."
   ```

4. **Run the server**
   ```bash
   uvicorn app:app --reload --port 7860
   ```

## 🐳 Docker Deployment

The Space uses the default Dockerfile for FastAPI. Ensure your `requirements.txt` includes `prophet`, `fastapi`, `uvicorn`, and `huggingface_hub`.

---
*Built for the Open Science Community.*
