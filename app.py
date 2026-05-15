from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import json

from injury_impact import calculate_matchup_injury_adjustment

app = FastAPI()

MODEL_PATH = "models/basketball_xgb_calibrated_v3.joblib"
TEAM_MAP_PATH = "team_map.json"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

with open(TEAM_MAP_PATH, "r") as f:
    team_map = {int(k): v for k, v in json.load(f).items()}

history = pd.read_parquet("outputs/training_dataset.parquet")


@app.get("/")
def root():
    return {"message": "backend live"}


@app.get("/teams")
def teams():
    team_names = sorted(
        set(history["home_team_name"]).union(
            set(history["away_team_name"])
        )
    )

    return {"teams": team_names}


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

    injury_data = calculate_matchup_injury_adjustment(
        home_team,
        away_team
    )

    row["home_injury_penalty"] = injury_data["home_injury_penalty"]
    row["away_injury_penalty"] = injury_data["away_injury_penalty"]
    row["injury_diff"] = injury_data["injury_diff"]

    X = pd.DataFrame([row])

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    prob = model.predict_proba(X)[0][1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": round(float(prob), 4),
        "away_win_probability": round(float(1 - prob), 4),
        "prediction": home_team if prob >= 0.5 else away_team,
        "home_injury_penalty": injury_data["home_injury_penalty"],
        "away_injury_penalty": injury_data["away_injury_penalty"],
        "injury_diff": injury_data["injury_diff"]
    }


@app.get("/predict_today")
def predict_today(date: str = None):
    try:
        today = date or datetime.now().strftime("%m/%d/%Y")

        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        frames = scoreboard.get_data_frames()

        if len(frames) == 0:
            return {
                "date": today,
                "games": [],
                "message": "No NBA schedule data returned"
            }

        games_df = frames[0].fillna("")

        if games_df.empty:
            return {
                "date": today,
                "games": [],
                "message": "No NBA games found"
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

        grouped_games = []

        for game_id in line_score["GAME_ID"].unique():
            game_df = line_score[line_score["GAME_ID"] == game_id]

            if len(game_df) != 2:
                continue

            grouped_games.append(game_df)

        for game_df in grouped_games:
            team1 = game_df.iloc[0]
            team2 = game_df.iloc[1]

            t1 = f"{team1['TEAM_CITY_NAME']} {team1['TEAM_NAME']}"
            t2 = f"{team2['TEAM_CITY_NAME']} {team2['TEAM_NAME']}"

            teams_match = sorted([
                t1.lower(),
                t2.lower()
            ]) == sorted([
                home_team.lower(),
                away_team.lower()
            ])

            if teams_match:
                team1_points = int(team1["PTS"])
                team2_points = int(team2["PTS"])

                if team1_points > team2_points:
                    winner = t1
                else:
                    winner = t2

                result = (
                    "Win"
                    if winner.lower() == best_bet.lower()
                    else "Loss"
                )

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
