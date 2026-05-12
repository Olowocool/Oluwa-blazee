from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

MODEL_PATH = "models/basketball_xgb_calibrated_v3.joblib"
DATA_PATH = "outputs/training_dataset.parquet"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

history = pd.read_parquet(DATA_PATH)
team_map = {}

for _, row in history.iterrows():
    team_map[int(row["home_team_id"])] = row["home_team_name"]
    team_map[int(row["away_team_id"])] = row["away_team_name"]
@app.get("/")
def home():
    return {"status": "NBA prediction API running", "model_loaded": True}

@app.get("/teams")
def teams():
    team_names = sorted(set(history["home_team_name"]).union(set(history["away_team_name"])))
    return {"teams": team_names}

@app.post("/predict_matchup")
def predict_matchup(payload: dict):
    home_team = payload["home_team"]
    away_team = payload["away_team"]

    home_games = history[
        (history["home_team_name"] == home_team) | 
        (history["away_team_name"] == home_team)
    ]

    away_games = history[
        (history["home_team_name"] == away_team) | 
        (history["away_team_name"] == away_team)
    ]

    if home_games.empty:
        return {"error": f"Home team not found: {home_team}"}

    if away_games.empty:
        return {"error": f"Away team not found: {away_team}"}

    latest_home = home_games.sort_values("date").iloc[-1]
    latest_away = away_games.sort_values("date").iloc[-1]

    row = {}

    for col in feature_cols:
        if col.startswith("home_") and col in latest_home:
            row[col] = latest_home[col]
        elif col.startswith("away_") and col in latest_away:
            row[col] = latest_away[col]
        elif col.startswith("diff_"):
            base = col.replace("diff_", "")
            home_col = "home_" + base
            away_col = "away_" + base
            row[col] = latest_home.get(home_col, 0) - latest_away.get(away_col, 0)
        else:
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

        # Build team ID mapping from historical data
        team_map = {}

        for _, row in history.iterrows():
            if "home_team_id" in row and "home_team_name" in row:
                team_map[row["home_team_id"]] = row["home_team_name"]

            if "away_team_id" in row and "away_team_name" in row:
                team_map[row["away_team_id"]] = row["away_team_name"]

        predictions = []

        for _, game in games_df.iterrows():
            home_team_id = game["HOME_TEAM_ID"]
            away_team_id = game["VISITOR_TEAM_ID"]

            home_team = team_map.get(home_team_id)
            away_team = team_map.get(away_team_id)

            if not home_team or not away_team:
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

    except Exception as e:
        return {
            "error": str(e),
            "message": "predict_today failed"
        }

        if games_df.empty:
            return {
                "date": today,
                "games": [],
                "message": "No NBA games found for this date"
            }

        predictions = []

        for _, game in games_df.iterrows():
            home_team = game["HOME_TEAM_NAME"]
            away_team = game["VISITOR_TEAM_NAME"]

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
