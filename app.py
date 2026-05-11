from fastapi import FastAPI
import joblib
import pandas as pd
import traceback

app = FastAPI()

MODEL_PATH = "models/basketball_xgb_calibrated.joblib"

artifact = None
model_error = None

try:
    artifact = joblib.load(MODEL_PATH)
except Exception as e:
    model_error = traceback.format_exc()

@app.get("/")
def home():
    if artifact is None:
        return {
            "status": "API live, but model failed to load",
            "error": model_error
        }

    return {
        "status": "Basketball prediction API is running",
        "model_loaded": True
    }

@app.post("/predict")
def predict(features: dict):
    if artifact is None:
        return {
            "error": "Model not loaded",
            "details": model_error
        }

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    row = pd.DataFrame([features])

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols]

    prob = model.predict_proba(row)[0][1]

    return {
        "home_win_probability": round(float(prob), 4),
        "away_win_probability": round(float(1 - prob), 4),
        "prediction": "Home Team" if prob >= 0.5 else "Away Team"
    }
