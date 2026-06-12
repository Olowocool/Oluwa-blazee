from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
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

NBA_TEAM_ID_MAP = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls",
    1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612754: "Indiana Pacers",
    1610612746: "Los Angeles Clippers",
    1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans",
    1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612764: "Washington Wizards",
}

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
        model_error = str(e)

team_map = {}
try:
    with open(TEAM_MAP_PATH, "r") as f:
        team_map = {int(k): v for k, v in json.load(f).items()}
except Exception as e:
    model_error = f"{model_error} | team_map load error: {e}".strip(" |")

try:
    history = pd.read_parquet(DATA_PATH)
except Exception as e:
    history = pd.DataFrame()
    model_error = f"{model_error} | history load error: {e}".strip(" |")


@app.get("/")
def root():
    return {"message": "NBA backend live"}


@app.get("/version")
def version():
    return {
        "version": "basketball-model-v7-scoreboard-date-window-v3",
        "model_status": model_status,
        "model_error": model_error
    }


@app.get("/teams")
def teams():
    if history.empty:
        return {"teams": []}

    team_names = sorted(
        set(history["home_team_name"]).union(
            set(history["away_team_name"])
        )
    )

    return {"teams": team_names}


def parse_selected_date(date_value: str = None):
    if date_value is None or str(date_value).strip() == "":
        return datetime.now()

    date_value = str(date_value).strip()

    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_value, fmt)
        except Exception:
            pass

    raise ValueError("Invalid date format. Use MM/DD/YYYY or YYYY-MM-DD.")


def normalize_team_id(team_id):
    try:
        if pd.isna(team_id):
            return None
        return int(float(team_id))
    except Exception:
        return None


def team_name_from_id(team_id):
    team_id = normalize_team_id(team_id)

    if team_id is None:
        return ""

    if team_id in NBA_TEAM_ID_MAP:
        return NBA_TEAM_ID_MAP[team_id]

    if team_id in team_map:
        return team_map[team_id]

    return ""


def team_name_from_line(line_row, fallback_team_id=None):
    if line_row is not None:
        city = str(line_row.get("TEAM_CITY_NAME", "")).strip()
        name = str(line_row.get("TEAM_NAME", "")).strip()

        if city and name:
            return f"{city} {name}".strip()

        mapped = team_name_from_id(line_row.get("TEAM_ID", fallback_team_id))
        if mapped:
            return mapped

    return team_name_from_id(fallback_team_id)


def safe_int(value, default=0):
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


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


def safe_prediction(home_team, away_team):
    prediction = predict_matchup({
        "home_team": home_team,
        "away_team": away_team
    })

    if "error" not in prediction:
        return prediction

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": 0.5,
        "away_win_probability": 0.5,
        "prediction": home_team,
        "best_bet": home_team,
        "confidence": 0.5,
        "model_status": model_status,
        "warning": prediction.get("error", "Model prediction failed."),
        "home_recent_win_rate": 0,
        "away_recent_win_rate": 0,
        "home_recent_margin": 0,
        "away_recent_margin": 0,
        "home_rest_days": 0,
        "away_rest_days": 0,
        "home_strength": 0,
        "away_strength": 0,
        "home_injury_penalty": 0,
        "away_injury_penalty": 0,
        "injury_diff": 0,
        "injury_probability_adjustment": 0,
        "home_injuries": [],
        "away_injuries": []
    }


@app.post("/predict_matchup")
def predict_matchup(payload: dict):
    home_team = payload["home_team"]
    away_team = payload["away_team"]

    if history.empty:
        return {"error": "Training history is not loaded."}

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

    raw_prob = 0.5

    if model is not None and len(feature_cols) > 0:
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
    prediction = home_team if prob >= 0.5 else away_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
        "prediction": prediction,
        "best_bet": prediction,
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
    """
    Reliable date-window ScoreboardV2 schedule loader.

    Fixes the previous bug where a valid game was found in the date window
    but skipped/failed because ScoreboardV2 sometimes returns only one
    line_score row before full box-score data is populated.
    """
    try:
        try:
            parsed_date = parse_selected_date(date)
        except Exception as e:
            return {
                "date": date,
                "games": [],
                "games_found": 0,
                "mode": "invalid_date",
                "message": str(e)
            }

        selected_date = parsed_date.strftime("%m/%d/%Y")

        search_dates = [
            parsed_date,
            parsed_date - timedelta(days=1),
            parsed_date + timedelta(days=1),
        ]

        games = []
        seen_game_ids = set()
        checked_dates = []

        for search_date in search_dates:
            formatted_date = search_date.strftime("%m/%d/%Y")

            try:
                scoreboard = scoreboardv2.ScoreboardV2(
                    game_date=formatted_date
                )
                frames = scoreboard.get_data_frames()
            except Exception as e:
                checked_dates.append({
                    "date": formatted_date,
                    "frame0_rows": 0,
                    "frame1_rows": 0,
                    "error": str(e)
                })
                continue

            frame0_rows = len(frames[0]) if len(frames) > 0 else 0
            frame1_rows = len(frames[1]) if len(frames) > 1 else 0

            checked_dates.append({
                "date": formatted_date,
                "frame0_rows": frame0_rows,
                "frame1_rows": frame1_rows,
            })

            if len(frames) < 1:
                continue

            game_header = frames[0].fillna("")
            if game_header.empty:
                continue

            line_score = (
                frames[1].fillna("")
                if len(frames) > 1
                else pd.DataFrame()
            )

            for _, game_row in game_header.iterrows():
                game_id = str(game_row.get("GAME_ID", "")).strip()

                if not game_id or game_id in seen_game_ids:
                    continue

                home_team_id = game_row.get("HOME_TEAM_ID", None)
                away_team_id = game_row.get("VISITOR_TEAM_ID", None)

                home_line_row = None
                away_line_row = None
                game_lines = pd.DataFrame()

                if not line_score.empty and "GAME_ID" in line_score.columns:
                    game_lines = line_score[
                        line_score["GAME_ID"].astype(str) == game_id
                    ]

                    if not game_lines.empty and "TEAM_ID" in game_lines.columns:
                        for _, possible_line in game_lines.iterrows():
                            possible_team_id = normalize_team_id(
                                possible_line.get("TEAM_ID", None)
                            )
                    
                            normalized_home_id = normalize_team_id(home_team_id)
                            normalized_away_id = normalize_team_id(away_team_id)
                    
                            if possible_team_id == normalized_home_id:
                                home_line_row = possible_line
                    
                            if normalized_away_id is not None and possible_team_id == normalized_away_id:
                                away_line_row = possible_line
                    
                        if away_line_row is None and len(game_lines) >= 1:
                            for _, possible_line in game_lines.iterrows():
                                possible_team_id = normalize_team_id(
                                    possible_line.get("TEAM_ID", None)
                                )
                    
                                if possible_team_id != normalize_team_id(home_team_id):
                                    away_line_row = possible_line
                                    away_team_id = possible_team_id
                                    break

                # Always resolve team names from the official IDs first.
                home_team = team_name_from_id(home_team_id)
                away_team = team_name_from_id(away_team_id)

                if not home_team:
                    home_team = team_name_from_line(
                        home_line_row,
                        fallback_team_id=home_team_id
                    )

                if not away_team:
                    away_team = team_name_from_line(
                        away_line_row,
                        fallback_team_id=away_team_id
                    )

                if not home_team or not away_team:
                    # Do not drop the game. Return a safe placeholder so the
                    # frontend still shows the schedule issue clearly.
                    home_team = home_team or str(home_team_id)
                    away_team = away_team or str(away_team_id)
                    prediction = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_win_probability": 0.5,
                        "away_win_probability": 0.5,
                        "prediction": "No Bet",
                        "best_bet": "No Bet",
                        "confidence": 0,
                        "model_status": model_status,
                        "warning": "Could not fully resolve team names from NBA team IDs."
                    }
                else:
                    prediction = safe_prediction(
                        home_team,
                        away_team
                    )

                # IMPORTANT: initialize scores before checking line rows.
                # This fixes the bug where a game with only one line_score row
                # caused an exception and made the endpoint return zero games.
                home_score = 0
                away_score = 0

                if home_line_row is not None:
                    home_score = safe_int(
                        home_line_row.get("PTS", 0)
                    )

                if away_line_row is not None:
                    away_score = safe_int(
                        away_line_row.get("PTS", 0)
                    )

                prediction["game_id"] = game_id
                prediction["game_date"] = formatted_date
                prediction["selected_date"] = selected_date
                prediction["schedule_source_date"] = formatted_date
                prediction["home_score"] = home_score
                prediction["away_score"] = away_score
                prediction["game_status"] = str(
                    game_row.get("GAME_STATUS_TEXT", "")
                )

                games.append(prediction)
                seen_game_ids.add(game_id)

        if not games:
            return {
                "date": selected_date,
                "games": [],
                "games_found": 0,
                "mode": "scoreboardv2_date_window_v3",
                "message": "No real NBA games found for this selected date.",
                "checked_dates": checked_dates
            }

        return {
            "date": selected_date,
            "games": games,
            "games_found": len(games),
            "mode": "scoreboardv2_date_window_v3",
            "checked_dates": checked_dates
        }

    except Exception as e:
        return {
            "date": date,
            "games": [],
            "games_found": 0,
            "mode": "scoreboardv2_error",
            "error": str(e)
        }


@app.get("/daily-predictions")
def daily_predictions(date: str = None):
    return predict_today(date)


@app.get("/raw_scoreboard")
def raw_scoreboard(date: str):
    try:
        parsed_date = parse_selected_date(date)
        formatted_date = parsed_date.strftime("%m/%d/%Y")
        board = scoreboardv2.ScoreboardV2(game_date=formatted_date)
        frames = board.get_data_frames()

        frame_info = []
        for index, frame in enumerate(frames):
            frame_info.append({
                "frame": index,
                "rows": len(frame),
                "columns": list(frame.columns)
            })

        sample_frame_0 = []
        sample_frame_1 = []

        if len(frames) > 0 and not frames[0].empty:
            sample_frame_0 = frames[0].head(10).fillna("").to_dict(orient="records")

        if len(frames) > 1 and not frames[1].empty:
            sample_frame_1 = frames[1].head(20).fillna("").to_dict(orient="records")

        return {
            "input_date": date,
            "formatted_date": formatted_date,
            "num_frames": len(frames),
            "frames": frame_info,
            "frame0_rows": len(frames[0]) if len(frames) > 0 else 0,
            "frame1_rows": len(frames[1]) if len(frames) > 1 else 0,
            "sample_frame_0": sample_frame_0,
            "sample_frame_1": sample_frame_1
        }

    except Exception as e:
        return {
            "input_date": date,
            "mode": "raw_scoreboard_error",
            "error": str(e)
        }


@app.get("/score_result")
def score_result(
    date: str,
    home_team: str,
    away_team: str,
    best_bet: str
):
    try:
        parsed_date = parse_selected_date(date)
        scoreboard = scoreboardv2.ScoreboardV2(game_date=parsed_date.strftime("%m/%d/%Y"))
        frames = scoreboard.get_data_frames()

        if len(frames) < 2:
            return {"status": "pending", "message": "No line score data returned yet."}

        line_score = frames[1].fillna("")

        if line_score.empty:
            return {"status": "pending", "message": "No completed games found yet."}

        for game_id in line_score["GAME_ID"].unique():
            game_df = line_score[line_score["GAME_ID"] == game_id]

            if len(game_df) < 2:
                continue

            team1 = game_df.iloc[0]
            team2 = game_df.iloc[1]

            t1 = f"{team1['TEAM_CITY_NAME']} {team1['TEAM_NAME']}"
            t2 = f"{team2['TEAM_CITY_NAME']} {team2['TEAM_NAME']}"

            teams_match = sorted([t1.lower(), t2.lower()]) == sorted([home_team.lower(), away_team.lower()])

            if teams_match:
                team1_points = safe_int(team1["PTS"])
                team2_points = safe_int(team2["PTS"])
                winner = t1 if team1_points > team2_points else t2
                result = "Win" if winner.lower() == best_bet.lower() else "Loss"

                return {
                    "status": "completed",
                    "home_team": home_team,
                    "away_team": away_team,
                    "team1": t1,
                    "team2": t2,
                    "team1_score": team1_points,
                    "team2_score": team2_points,
                    "winner": winner,
                    "best_bet": best_bet,
                    "result": result
                }

        return {"status": "not_found", "message": "Game not found."}

    except Exception as e:
        return {"status": "error", "message": str(e)}


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

