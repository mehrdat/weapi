import pandas as pd
import requests
import numpy as np
import os
from prophet import Prophet
from huggingface_hub import HfApi,hf_hub_download
from datetime import datetime, timedelta
import io
import json
import time

HF_TOKEN=os.getenv("HF_TOKEN")
DATASET_REPO="https://huggingface.co/datasets/Mehrdat/weapi-store"

API_ROOT = "https://climatemonitor.info/api/public/v1"

FACTORS = {
    "carbon_dioxide": {
        "endpoint": "/co2/daily",
        "fallback": "/co2/monthly_gl",
        "bad_direction": "up",
        "unit": "ppm"
    },
    "methane": {
        "endpoint": "/ch4/monthly", 
        "fallback": "/ch4/annual",
        "bad_direction": "up", 
        "unit": "ppb"
    },
    "nitrous_oxide": {
        "endpoint": "/n2o/monthly", 
        "fallback": "/n2o/annual",
        "bad_direction": "up", 
        "unit": "ppb"
    },
    "sulfur_hexafluoride": {
        "endpoint": "/sf6/monthly", 
        "fallback": "/sf6/annual",
        "bad_direction": "up", 
        "unit": "ppt"
    },
    "temperature_anomaly": {
        "endpoint": "/temp/daily_avg", 
        "fallback": "/temp/monthly_anomaly_avg",
        "bad_direction": "up", 
        "unit": "°C"
    },
    "ocean_level": {
        "endpoint": "/ocean/level", 
        "fallback": None,
        "bad_direction": "up", 
        "unit": "mm"
    },
    "glaciers_greenland": {
        "endpoint": "/glaciers/greenland", 
        "fallback": None,
        "bad_direction": "down", 
        "unit": "Gt"
    },
    "glaciers_antarctica": {
        "endpoint": "/glaciers/antarctica", 
        "fallback": None,
        "bad_direction": "down", 
        "unit": "Gt"
    }
}


def fetch_data(factor_name,config):
    """Gets data from API and cleans it for Prophet."""
    url=f"{API_ROOT}{config['endpoint']}"
    print(f"   ...fetching {factor_name} from {url}")
    
    try:
        r=requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code!=200 and config["fallback"]:
            print(f"      -> Primary failed, trying fallback: {config['fallback']}")
            url = f"{API_ROOT}{config['fallback']}"
            r=requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        data=r.json().get("data", {})
        readings=data.get("readings",[]) if "readings" in data else data
        if not readings:
            print(f"      [!] No readings found for {factor_name}")
            return None
        df=pd.DataFrame(readings)
        if "label" in df.columns:
            df["ds"]=pd.to_datetime(df["label"])
        elif "date" in df.columns:
            df["ds"]=pd.to_datetime(df["date"])
        
        df["y"]=pd.to_numeric(df["value"], errors="coerce")
        df=df.dropna(subset=["ds","y"]).sort_values("ds")
        
        return df[["ds","y"]]
    except Exception as e:
        print(f"      [!] Error fetching {factor_name}: {e}")
        return None

def run_ml_forecast(df):
    """Runs Prophet to predict Today + next 7 days."""
    if len(df) < 20: 
        return None
    
    last_gap=(df.iloc[-1]["ds"] - df.iloc[-2]["ds"]).days
    freq="D" if last_gap<5 else "M"
    
    model=Prophet(
        daily_seasonality=(freq=="D"),
        yearly_seasonality=True,
        weekly_seasonality=(freq=="D"),
        ##seasonality_mode="additive"
    )
    try:
        model.fit(df)
        future=model.make_future_dataframe(periods=7, freq=freq)
        forecast=model.predict(future)
        
        return forecast.tail(7)
    except Exception as e:
        print(f"      [!] ML Error: {e}")
        return None
    

def calculate_index(current,forecast,bad_direction):
    """
    Returns Index Score: -10 (Crisis) to +10 (Great).
    Logic: Compares actual reality vs ML prediction.
    """
    if forecast is None or forecast.empty:
        return 0
    
    predicted=forecast.iloc[0]["yhat"]
    
    diff=current - predicted
    denominator=abs(predicted) * 0.5 if abs(predicted) > 0 else 1
    raw_score=(diff / denominator) * 10
    
    if bad_direction=="up":
        final_score=-raw_score
        
    elif bad_direction=="down":
        final_score=raw_score
        
    return max(min(int(final_score),10),-10)

def main():
    print("🚀 Starting daily CO2 index calculation...")
    today_str=datetime.now().strftime("%Y-%m-%d")
    
    full_report={
        "date": today_str,
        "generated_at": datetime.now().isoformat(),
        "climate_factors": {}
    }
    overall_health_sum=0
    valid_factors_count=0
    
    for name,config in FACTORS.items():
        print(f"\n Processing {name}...")
        df=fetch_data(name,config)
        if df is None or df.empty:
            print(f"   [!] Skipping {name} due to insufficient data.")
            continue
        latest_real=df.iloc[-1]
        forecast=run_ml_forecast(df)
        
        predicted_val=forecast.iloc[0]["yhat"] if forecast is not None else latest_real['y']
        lower_bound=forecast.iloc[0]["yhat_lower"] if forecast is not None else 0
        upper_bound=forecast.iloc[0]["yhat_upper"] if forecast is not None else 0
        
        weapi_index=calculate_index(latest_real['y'], forecast, config["bad_direction"])
        
        if weapi_index <-2:
            status = "Worse than expected"
        elif weapi_index > 2:
            status = "Better than expected"
        else:
            status = "As expected"
            
        full_report["climate_factors"][name] = {
            "current_value": round(latest_real['y'], 2),
            "unit": config['unit'],
            "last_updated": latest_real['ds'].strftime("%Y-%m-%d"),
            "forecast_expected": round(predicted_val, 2),
            "confidence_interval": [round(lower_bound, 2), round(upper_bound, 2)],
            "weapi_index": round(weapi_index, 2),
            "status": status,
            "trend_direction": config['bad_direction'] + "_is_bad"
        }
        
        overall_health_sum += weapi_index
        valid_factors_count += 1
        time.sleep(1)
    if valid_factors_count > 0:
        global_score = overall_health_sum / valid_factors_count
        full_report["global_climate_score"] = round(global_score, 2)
        
    else:
        full_report["global_climate_score"] = 0
    print("\n💾 SAVING REPORT TO HUGGING FACE...")
    print(json.dumps(full_report, indent=2))
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(full_report).encode('utf-8')),
            path_in_repo=f"daily_reports/{today_str}.json",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("✅ SUCCESS: Uploaded to Hugging Face Datasets")
    except Exception as e:
        print(f"❌ UPLOAD FAILED: {e}")

if __name__ == "__main__":
    main()
