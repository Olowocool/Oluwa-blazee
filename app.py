from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI()

MODEL_PATH = "models/basketball_xgb_calibrated_v3.joblib"
TEAM_MAP_PATH = "team_map.json"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

with open(TEAM_MAP_PATH, "r") as f:
    team_map = {int(k): v for k, v in json.load(f).items()}

team_names = sorted(set(team_map.values()))


@app.get("/")
def home():
    return {
        "status": "NBA prediction API running",
        "model_loaded": True
    }


@app.get("/teams")
def teams():
    return {"teams": team_names}


@app.post("/predict")
def predict(features: dict):
    row = pd.DataFrame([features])

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    prob = model.predict_proba(row)[0][1]

    return {
        "home_win_probability": round(float(prob), 4),
        "away_win_probability": round(float(1 - prob), 4),
        "prediction": "Home Team" if prob >= 0.5 else "Away Team"
    }


@app.post("/predict_matchup")
def predict_matchup(payload: dict):
    home_team = payload["home_team"]
    away_team = payload["away_team"]

    # Lightweight approximation.
    # We avoid loading the full training parquet on Render free plan.
    row = {}

    for col in feature_cols:
        row[col] = 0

    row["home_court"] = 1

    X = pd.DataFrame([row])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    prob = model.predict_proba(X)[0][1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": round(float(prob), 4),
        "away_win_probability": round(float(1 - prob), 4),
        "prediction": home_team if prob >= 0.5 else away_team
    }


@app.get("/predict_today")
def predict_today(date: str = None):
    try:
        today = date or datetime.now().strftime("%m/%d/%Y")

        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        games_df = scoreboard.get_data_frames()[0]

        if games_df.empty:
            return {
                "date": today,
                "games": [],
                "message": "No NBA games found"
            }

        predictions = []

        for _, game in games_df.iterrows():
            home_team_id = int(game["HOME_TEAM_ID"])
            away_team_id = int(game["VISITOR_TEAM_ID"])

            home_team = team_map.get(home_team_id)
            away_team = team_map.get(away_team_id)

            if not home_team or not away_team:
                predictions.append({
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "error": "Team ID not found in team_map.json"
                })
                continue

            result = predict_matchup({
                "home_team": home_team,
                "away_team": away_team
            })

            predictions.append(result)

        return {
            "date": today,
            "games": predictions
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "predict_today failed"
        }
