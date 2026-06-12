from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import json
import os
import requests

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

TEAM_NAME_FIXES = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Los Angeles Clippers": "Los Angeles Clippers",
    "New York Knicks": "New York Knicks",
    "San Antonio Spurs": "San Antonio Spurs",
    "Philadelphia 76ers": "Philadelphia 76ers",
}

DISPLAY_TIMEZONE_OFFSET_HOURS = 1  # Nigeria / WAT display date


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
        "version": "basketball-model-v8-espn-window-wat-date",
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


def normalize_team_name(name):
    name = str(name or "").strip()
    return TEAM_NAME_FIXES.get(name, name)


def team_name_from_id(team_id):
    team_id = normalize_team_id(team_id)

    if team_id is None:
        return ""

    if team_id in NBA_TEAM_ID_MAP:
        return NBA_TEAM_ID_MAP[team_id]

    if team_id in team_map:
        return team_map[team_id]

    return ""


def safe_int(value, default=0):
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def event_display_date(event_date_text):
    """
    ESPN returns UTC timestamps like 2026-06-11T00:30Z.
    The user-facing app is using Nigeria/Livescore calendar behavior,
    so convert UTC to WAT before comparing dates.
    """
    try:
        dt = datetime.fromisoformat(
            str(event_date_text).replace("Z", "+00:00")
        )
        local_dt = dt.astimezone(
            timezone(timedelta(hours=DISPLAY_TIMEZONE_OFFSET_HOURS))
        )
        return local_dt.strftime("%m/%d/%Y")
    except Exception:
        return ""


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


def fallback_prediction(home_team, away_team, warning):
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": 0.5,
        "away_win_probability": 0.5,
        "prediction": "No Bet",
        "best_bet": "No Bet",
        "confidence": 0,
        "model_status": model_status,
        "warning": warning,
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


def safe_prediction(home_team, away_team):
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    if not home_team or not away_team:
        return fallback_prediction(
            home_team,
            away_team,
            "Missing home or away team."
        )

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
    home_team = normalize_team_name(payload["home_team"])
    away_team = normalize_team_name(payload["away_team"])

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


def espn_games_for_schedule_date(schedule_date: datetime):
    espn_date = schedule_date.strftime("%Y%m%d")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

    response = requests.get(
        url,
        params={"dates": espn_date},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()
    return response.json().get("events", [])


def convert_espn_event_to_game(event, selected_display_date):
    competition = event.get("competitions", [{}])[0]
    competitors = competition.get("competitors", [])

    home_team = ""
    away_team = ""
    home_score = 0
    away_score = 0

    for competitor in competitors:
        team = competitor.get("team", {})
        display_name = normalize_team_name(team.get("displayName", ""))

        score = safe_int(competitor.get("score", 0))

        if competitor.get("homeAway") == "home":
            home_team = display_name
            home_score = score
        elif competitor.get("homeAway") == "away":
            away_team = display_name
            away_score = score

    if not home_team or not away_team:
        return None

    prediction = safe_prediction(home_team, away_team)

    prediction["game_id"] = str(event.get("id", ""))
    prediction["game_date"] = selected_display_date
    prediction["selected_date"] = selected_display_date
    prediction["home_score"] = home_score
    prediction["away_score"] = away_score
    prediction["game_status"] = event.get("status", {}).get("type", {}).get("description", "")
    prediction["game_time"] = event.get("date", "")
    prediction["schedule_provider"] = "espn_scoreboard"
    prediction["display_date"] = event_display_date(event.get("date", ""))

    return prediction


def get_espn_schedule_window(selected_date: datetime):
    """
    Query ESPN for selected, previous, and next ET schedule dates,
    then keep only games whose WAT display date equals the user's selected date.
    """
    selected_display_date = selected_date.strftime("%m/%d/%Y")

    schedule_dates = [
        selected_date - timedelta(days=1),
        selected_date,
        selected_date + timedelta(days=1),
    ]

    games = []
    seen_ids = set()
    checked = []

    for schedule_date in schedule_dates:
        query_date = schedule_date.strftime("%m/%d/%Y")

        try:
            events = espn_games_for_schedule_date(schedule_date)
        except Exception as e:
            checked.append({
                "source": "espn_scoreboard",
                "query_date": query_date,
                "error": str(e)
            })
            continue

        kept = 0

        for event in events:
            display_date = event_display_date(event.get("date", ""))

            if display_date != selected_display_date:
                continue

            game_id = str(event.get("id", ""))

            if game_id and game_id in seen_ids:
                continue

            game = convert_espn_event_to_game(
                event,
                selected_display_date
            )

            if game is None:
                continue

            game["schedule_source_date"] = query_date

            games.append(game)
            kept += 1

            if game_id:
                seen_ids.add(game_id)

        checked.append({
            "source": "espn_scoreboard",
            "query_date": query_date,
            "events_found": len(events),
            "games_kept_for_selected_display_date": kept,
        })

    return games, checked


def get_nba_scoreboard_games_for_date(target_date: datetime):
    formatted_date = target_date.strftime("%m/%d/%Y")

    scoreboard = scoreboardv2.ScoreboardV2(game_date=formatted_date)
    frames = scoreboard.get_data_frames()

    if len(frames) < 1:
        return [], {"date": formatted_date, "frame0_rows": 0, "frame1_rows": 0}

    game_header = frames[0].fillna("")
    line_score = frames[1].fillna("") if len(frames) > 1 else pd.DataFrame()

    debug = {
        "date": formatted_date,
        "frame0_rows": len(game_header),
        "frame1_rows": len(line_score),
    }

    if game_header.empty:
        return [], debug

    games = []

    for _, game_row in game_header.iterrows():
        game_id = str(game_row.get("GAME_ID", "")).strip()

        if not game_id:
            continue

        home_team_id = game_row.get("HOME_TEAM_ID", None)
        away_team_id = game_row.get("VISITOR_TEAM_ID", None)

        home_team = team_name_from_id(home_team_id)
        away_team = team_name_from_id(away_team_id)

        game_lines = pd.DataFrame()

        if not line_score.empty and "GAME_ID" in line_score.columns:
            game_lines = line_score[line_score["GAME_ID"].astype(str) == game_id]

        if not game_lines.empty and "TEAM_ID" in game_lines.columns:
            for _, line in game_lines.iterrows():
                line_team_id = normalize_team_id(line.get("TEAM_ID", None))
                line_team_name = normalize_team_name(
                    f"{line.get('TEAM_CITY_NAME', '')} {line.get('TEAM_NAME', '')}".strip()
                )

                if line_team_id == normalize_team_id(home_team_id):
                    home_team = line_team_name or home_team
                elif line_team_id == normalize_team_id(away_team_id):
                    away_team = line_team_name or away_team
                elif not away_team and line_team_id != normalize_team_id(home_team_id):
                    away_team = line_team_name

        if not home_team or not away_team:
            continue

        prediction = safe_prediction(home_team, away_team)

        prediction["game_id"] = game_id
        prediction["game_date"] = formatted_date
        prediction["selected_date"] = formatted_date
        prediction["schedule_source_date"] = formatted_date
        prediction["home_score"] = 0
        prediction["away_score"] = 0
        prediction["game_status"] = str(game_row.get("GAME_STATUS_TEXT", ""))
        prediction["schedule_provider"] = "nba_scoreboardv2"

        games.append(prediction)

    return games, debug


@app.get("/predict_today")
def predict_today(date: str = None):
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

        # Primary: ESPN schedule window filtered to WAT display date.
        espn_games, checked_sources = get_espn_schedule_window(parsed_date)

        if espn_games:
            return {
                "date": selected_date,
                "games": espn_games,
                "games_found": len(espn_games),
                "mode": "espn_scoreboard_wat_date_window",
                "checked_sources": checked_sources
            }

        # Fallback: NBA ScoreboardV2 window.
        games = []
        seen_game_ids = set()

        search_dates = [
            parsed_date,
            parsed_date - timedelta(days=1),
            parsed_date + timedelta(days=1),
        ]

        for search_date in search_dates:
            fallback_games, debug = get_nba_scoreboard_games_for_date(search_date)
            checked_sources.append({
                "source": "nba_scoreboardv2",
                **debug
            })

            for game in fallback_games:
                game_id = str(game.get("game_id", ""))
                if game_id and game_id in seen_game_ids:
                    continue

                game["selected_date"] = selected_date
                games.append(game)

                if game_id:
                    seen_game_ids.add(game_id)

        if games:
            return {
                "date": selected_date,
                "games": games,
                "games_found": len(games),
                "mode": "nba_scoreboardv2_fallback_window",
                "checked_sources": checked_sources
            }

        return {
            "date": selected_date,
            "games": [],
            "games_found": 0,
            "mode": "schedule_not_found",
            "message": "No real NBA games found for this selected date.",
            "checked_sources": checked_sources
        }

    except Exception as e:
        return {
            "date": date,
            "games": [],
            "games_found": 0,
            "mode": "predict_today_error",
            "error": str(e)
        }


@app.get("/daily-predictions")
def daily_predictions(date: str = None):
    return predict_today(date)


@app.get("/raw_espn_scoreboard")
def raw_espn_scoreboard(date: str):
    try:
        parsed_date = parse_selected_date(date)

        events = espn_games_for_schedule_date(parsed_date)

        event_rows = []

        for event in events:
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            event_teams = []

            for competitor in competitors:
                team = competitor.get("team", {})
                event_teams.append({
                    "homeAway": competitor.get("homeAway"),
                    "displayName": team.get("displayName"),
                    "score": competitor.get("score")
                })

            event_rows.append({
                "id": event.get("id"),
                "name": event.get("name"),
                "date": event.get("date"),
                "display_date_wat": event_display_date(event.get("date", "")),
                "status": event.get("status", {}).get("type", {}).get("description", ""),
                "teams": event_teams
            })

        return {
            "input_date": date,
            "espn_query_date": parsed_date.strftime("%Y%m%d"),
            "events_found": len(event_rows),
            "events": event_rows
        }

    except Exception as e:
        return {
            "input_date": date,
            "mode": "raw_espn_scoreboard_error",
            "error": str(e)
        }


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

        espn_games, _ = get_espn_schedule_window(parsed_date)

        for game in espn_games:
            teams_match = sorted([
                str(game.get("home_team", "")).lower(),
                str(game.get("away_team", "")).lower()
            ]) == sorted([
                home_team.lower(),
                away_team.lower()
            ])

            if not teams_match:
                continue

            home_score = safe_int(game.get("home_score", 0))
            away_score = safe_int(game.get("away_score", 0))

            if home_score <= 0 and away_score <= 0:
                return {
                    "status": "pending",
                    "message": "Game has no final score yet."
                }

            winner = home_team if home_score > away_score else away_team
            result = "Win" if winner.lower() == best_bet.lower() else "Loss"

            return {
                "status": "completed",
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
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

