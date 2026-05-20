from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import json
import os

from injury_impact import calculate_matchup_injury_adjustment

from model_quality import (
    calculate_recent_form,
    calculate_home_away_strength,
    calculate_rest_days,
    quality_adjust_probability
)

app = FastAPI(title="NBA Basketball Prediction Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CANDIDATES = [
    "models/basketball_xgb_calibrated_v3.joblib",
    "models/basketball_xgb_calibrated_v2.joblib",
    "basketball_xgb_calibrated_v3.joblib",
    "basketball_xgb_calibrated_v2.joblib",
]

TEAM_MAP_PATH = "team_map.json"
DATA_PATH = "outputs/training_dataset.parquet"

model = None
feature_cols = []
model_status = "not_loaded"
model_error = ""

for path in MODEL_CANDIDATES:
    try:
        if os.path.isfile(path):
            artifact = joblib.load(path)
            model = artifact["model"]
            feature_cols = artifact["feature_cols"]
            model_status = f"loaded: {path}"
            break
    except Exception as e:
        model_error = f"{path}: {str(e)}"

with open(TEAM_MAP_PATH, "r") as f:
    team_map = {int(k): v for k, v in json.load(f).items()}

history = pd.read_parquet(DATA_PATH)


@app.get("/")
def root():
    return {"message": "NBA backend live"}


@app.get("/version")
def version():
    return {
        "version": "basketball-model-v4-safe-quality-upgrade",
        "model_status": model_status,
        "model_error": model_error,
        "message": "backend is live even if model file fails"
    }


@app.get("/teams")
def teams():
    team_names = sorted(
        set(history["home_team_name"]).union(set(history["away_team_name"]))
    )
    return {"teams": team_names}


def build_feature_row(latest_home, latest_away):
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

        elif col == "home_court":
            row[col] = 1

        else:
            row[col] = 0

    row["home_court"] = 1
    return row


def fallback_probability(home_recent_form, away_recent_form, home_rest_days, away_rest_days, injury_adjustment):
    base_prob = 0.5

    return quality_adjust_probability(
        raw_prob=base_prob,
        home_recent_form=home_recent_form,
        away_recent_form=away_recent_form,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        injury_adjustment=injury_adjustment
    )


@app.post("/predict_matchup")
def predict_matchup(payload: dict):
    home_team = payload["home_team"]
    away_team = payload["away_team"]

    home_games = history[
        (history["home_team_name"] == home_team)
        | (history["away_team_name"] == home_team)
    ]

    away_games = history[
        (history["home_team_name"] == away_team)
        | (history["away_team_name"] == away_team)
    ]

    if home_games.empty:
        return {"error": f"Home team not found: {home_team}"}

    if away_games.empty:
        return {"error": f"Away team not found: {away_team}"}

    latest_home = home_games.sort_values("date").iloc[-1]
    latest_away = away_games.sort_values("date").iloc[-1]

    injury_data = calculate_matchup_injury_adjustment(home_team, away_team)

    home_recent_form = calculate_recent_form(home_games, home_team)
    away_recent_form = calculate_recent_form(away_games, away_team)

    home_strength = calculate_home_away_strength(home_games, home_team)
    away_strength = calculate_home_away_strength(away_games, away_team)

    home_rest_days = calculate_rest_days(home_games)
    away_rest_days = calculate_rest_days(away_games)

    injury_adjustment = injury_data["injury_diff"] * 0.004

    if model is not None and feature_cols:
        try:
            row = build_feature_row(latest_home, latest_away)

            X = pd.DataFrame([row])

            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0

            X = X[feature_cols]
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)

            raw_prob = float(model.predict_proba(X)[0][1])

        except Exception:
            raw_prob = 0.5
    else:
        raw_prob = 0.5

    prob = quality_adjust_probability(
        raw_prob=raw_prob,
        home_recent_form=home_recent_form,
        away_recent_form=away_recent_form,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
        injury_adjustment=injury_adjustment
    )

    prob = max(0.05, min(0.95, prob))

    home_probability = round(float(prob), 4)
    away_probability = round(float(1 - prob), 4)

    best_bet = home_team if prob >= 0.5 else away_team

    return {
        "home_team": home_team,
        "away_team": away_team,

        "home_win_probability": home_probability,
        "away_win_probability": away_probability,

        "prediction": best_bet,
        "best_bet": best_bet,
        "confidence": round(float(max(prob, 1 - prob)), 4),

        "model_status": model_status,
        "raw_home_win_probability": round(float(raw_prob), 4),

        "home_recent_win_rate": round(float(home_recent_form["recent_win_rate"]), 4),
        "away_recent_win_rate": round(float(away_recent_form["recent_win_rate"]), 4),

        "home_recent_margin": round(float(home_recent_form["recent_margin"]), 2),
        "away_recent_margin": round(float(away_recent_form["recent_margin"]), 2),

        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,

        "home_strength": round(float(home_strength["home_strength"]), 4),
        "away_strength": round(float(away_strength["away_strength"]), 4),

        "home_injury_penalty": injury_data["home_injury_penalty"],
        "away_injury_penalty": injury_data["away_injury_penalty"],
        "injury_diff": injury_data["injury_diff"],
        "injury_probability_adjustment": round(float(injury_adjustment), 4),

        "home_injuries": injury_data.get("home_injuries", []),
        "away_injuries": injury_data.get("away_injuries", [])
    }


@app.get("/predict_today")
def predict_today(date: str = None):
    try:
        today = date or datetime.now().strftime("%m/%d/%Y")

        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        frames = scoreboard.get_data_frames()

        games_df = None

        for frame in frames:
            if (
                "HOME_TEAM_ID" in frame.columns
                and "VISITOR_TEAM_ID" in frame.columns
            ):
                games_df = frame.fillna("")
                break

        if games_df is None or games_df.empty:
            return {
                "date": today,
                "games": [],
                "message": "No NBA games found from scoreboard frames",
                "frames_available": [list(f.columns) for f in frames]
            }

        predictions = []

            for _, game in games_df.iterrows():
                home_team_id = game.get("HOME_TEAM_ID")
                away_team_id = game.get("VISITOR_TEAM_ID")
    
                if home_team_id == "" or away_team_id == "":
                    continue
    
                home_team = team_map.get(int(home_team_id))
                away_team = team_map.get(int(away_team_id))
    
                if not home_team or not away_team:
                    continue
    
                result = predict_matchup({
                    "home_team": home_team,
                    "away_team": away_team
                })
    
                if "error" not in result:
                    predictions.append(result)
    
            return {
                "date": today,
                "games": predictions,
                "games_found": len(predictions)
            }
    
        except Exception as e:
            return {
                "date": date,
                "games": [],
                "error": str(e),
                "message": "predict_today failed"
            }
    
            predictions = []

        for _, game in games_df.iterrows():
            home_team_id = game.get("HOME_TEAM_ID")
            away_team_id = game.get("VISITOR_TEAM_ID")

            if home_team_id == "" or away_team_id == "":
                continue

            home_team = team_map.get(int(home_team_id))
            away_team = team_map.get(int(away_team_id))

            if not home_team or not away_team:
                continue

            result = predict_matchup(
                {
                    "home_team": home_team,
                    "away_team": away_team
                }
            )

            if "error" not in result:
                predictions.append(result)

        return {
            "date": today,
            "games": predictions
        }

    except Exception as e:
        return {
            "date": date,
            "games": [],
            "error": str(e),
            "message": "predict_today failed"
        }


@app.get("/daily-predictions")
def daily_predictions(date: str = None):
    return predict_today(date)


@app.get("/score_result")
def score_result(
    date: str,
    home_team: str,
    away_team: str,
    best_bet: str
):
    try:
        parsed_date = datetime.strptime(date, "%m/%d/%Y")

        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=parsed_date.strftime("%m/%d/%Y")
        )

        frames = scoreboard.get_data_frames()

        if len(frames) < 2:
            return {
                "status": "pending",
                "message": "No line score data returned yet."
            }

        line_score = frames[1].fillna("")

        if line_score.empty:
            return {
                "status": "pending",
                "message": "No completed games found yet."
            }

        for game_id in line_score["GAME_ID"].unique():
            game_df = line_score[line_score["GAME_ID"] == game_id]

            if len(game_df) != 2:
                continue

            team1 = game_df.iloc[0]
            team2 = game_df.iloc[1]

            t1 = f"{team1['TEAM_CITY_NAME']} {team1['TEAM_NAME']}"
            t2 = f"{team2['TEAM_CITY_NAME']} {team2['TEAM_NAME']}"

            teams_match = sorted([t1.lower(), t2.lower()]) == sorted(
                [home_team.lower(), away_team.lower()]
            )

            if teams_match:
                team1_points = int(team1["PTS"])
                team2_points = int(team2["PTS"])

                winner = t1 if team1_points > team2_points else t2

                result = "Win" if winner.lower() == best_bet.lower() else "Loss"

                return {
                    "status": "completed",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": team1_points,
                    "away_score": team2_points,
                    "winner": winner,
                    "best_bet": best_bet,
                    "result": result
                }

        return {
            "status": "not_found",
            "message": "Game not found."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/debug_injuries")
def debug_injuries():
    sample_teams = [
        "Cleveland Cavaliers",
        "Detroit Pistons",
        "Minnesota Timberwolves",
        "San Antonio Spurs",
        "Denver Nuggets",
        "Oklahoma City Thunder"
    ]

    output = {}

    for team in sample_teams:
        output[team] = calculate_matchup_injury_adjustment(team, team)

    return output
