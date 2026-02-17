from fastapi import FastAPI
from huggingface_hub import hf_hub_download
import json
import os
from datetime import datetime, timedelta

app=FastAPI(title="Climate Monitor API - CO2 Data")

DATASET_REPO = "/Mehrdat/weapi-store"

@app.get("/")
def home():
    return {"message":"Welcome to the Climate Monitor API - CO2 Data"}

@app.get("/latest")
def get_last_data():
    today=datetime.now().strftime("%Y-%m-%d")
    try:
        file_path=hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"data/{today}.json",
            repo_type="dataset"
        )
        with open(file_path,"r") as f:
            data=f.read()
            return eval(data)
    except Exception as e:
        return {"error":f"Data for {today} not available yet. Please check back later."}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
