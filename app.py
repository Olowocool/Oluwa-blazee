from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

artifact = joblib.load("models/basketball_xgb_calibrated.joblib")

model = artifact["model"]
feature_cols = artifact["feature_cols"]

@app.get("/")
def home():
    return {"status": "Basketball prediction API is running"}

@app.post("/predict")
def predict(features: dict):

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
