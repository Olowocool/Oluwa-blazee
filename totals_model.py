import os
import pandas as pd


DATA_PATH = "outputs/training_dataset.parquet"
LEAGUE_AVG_TOTAL = 224
LEAGUE_AVG_TEAM_POINTS = 112


def safe_float(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_history():
    if not os.path.exists(DATA_PATH):
        return None

    try:
        return pd.read_parquet(DATA_PATH)
    except Exception:
        return None


def team_recent_points(team_name, history, last_n=10):
    if history is None or history.empty:
        return {
            "points_for": 112,
            "points_allowed": 112,
            "pace_score": 224
        }

    games = history[
        (history["home_team_name"] == team_name)
        |
        (history["away_team_name"] == team_name)
    ].copy()

    if games.empty:
        return {
            "points_for": 112,
            "points_allowed": 112,
            "pace_score": 224
        }

    games["date"] = pd.to_datetime(
        games["date"],
        errors="coerce"
    )

    games = games.sort_values("date").tail(last_n)

    scored = []
    allowed = []
    totals = []

    for _, row in games.iterrows():
        home_team = row.get("home_team_name")
        away_team = row.get("away_team_name")

        home_score = safe_float(
            row.get("home_team_score", row.get("home_points", 0))
        )

        away_score = safe_float(
            row.get("away_team_score", row.get("away_points", 0))
        )

        if home_score <= 0 or away_score <= 0:
            continue

        game_total = home_score + away_score

        if home_team == team_name:
            scored.append(home_score)
            allowed.append(away_score)
            totals.append(game_total)

        elif away_team == team_name:
            scored.append(away_score)
            allowed.append(home_score)
            totals.append(game_total)

    if not scored:
        return {
            "points_for": 112,
            "points_allowed": 112,
            "pace_score": 224
        }

    return {
        "points_for": sum(scored) / len(scored),
        "points_allowed": sum(allowed) / len(allowed),
        "pace_score": sum(totals) / len(totals)
    }


def calculate_pace_adjustment(home_pace_score, away_pace_score):
    combined_pace = (
        home_pace_score + away_pace_score
    ) / 2

    pace_gap = combined_pace - LEAGUE_AVG_TOTAL

    pace_adjustment = pace_gap * 0.35

    return {
        "combined_pace_score": round(combined_pace, 1),
        "pace_gap": round(pace_gap, 1),
        "pace_adjustment": round(pace_adjustment, 1)
    }


def predict_game_total(home_team, away_team, bookmaker_total=None):
    history = load_history()

    home_5 = team_recent_points(home_team, history, last_n=5)
    away_5 = team_recent_points(away_team, history, last_n=5)

    home_10 = team_recent_points(home_team, history, last_n=10)
    away_10 = team_recent_points(away_team, history, last_n=10)

    projected_home_points = (
        (home_10["points_for"] * 0.45)
        + (away_10["points_allowed"] * 0.35)
        + (home_5["points_for"] * 0.20)
    )

    projected_away_points = (
        (away_10["points_for"] * 0.45)
        + (home_10["points_allowed"] * 0.35)
        + (away_5["points_for"] * 0.20)
    )

    raw_projected_total = (
        projected_home_points + projected_away_points
    )

    pace_data = calculate_pace_adjustment(
        home_pace_score=home_10["pace_score"],
        away_pace_score=away_10["pace_score"]
    )

    projected_total = (
        raw_projected_total
        + pace_data["pace_adjustment"]
    )

    confidence_note = "No betting line entered."
    recommendation = "No Bet"
    edge = None

    if bookmaker_total is not None:
        bookmaker_total = safe_float(bookmaker_total)

        edge = projected_total - bookmaker_total

        if edge >= 6:
            recommendation = "Over"
            confidence_note = "Strong Over edge"

        elif edge >= 3:
            recommendation = "Lean Over"
            confidence_note = "Small Over edge"

        elif edge <= -6:
            recommendation = "Under"
            confidence_note = "Strong Under edge"

        elif edge <= -3:
            recommendation = "Lean Under"
            confidence_note = "Small Under edge"

        else:
            recommendation = "No Bet"
            confidence_note = "Line is too close to projection"

    return {
        "status": "success",
        "home_team": home_team,
        "away_team": away_team,

        "projected_home_points": round(projected_home_points, 1),
        "projected_away_points": round(projected_away_points, 1),

        "raw_projected_total": round(raw_projected_total, 1),
        "pace_adjustment": pace_data["pace_adjustment"],
        "combined_pace_score": pace_data["combined_pace_score"],
        "pace_gap": pace_data["pace_gap"],

        "projected_total": round(projected_total, 1),
        "raw_projected_total": round(raw_projected_total, 2),
        "pace_adjustment": round(pace_adjustment, 2),
        "combined_pace_score": round(combined_pace_score, 2),
        "pace_gap": round(pace_gap, 2),
        "bookmaker_total": bookmaker_total,
        "edge": round(edge, 1) if edge is not None else None,

        "recommendation": recommendation,
        "confidence_note": confidence_note,

        "home_last_5_ppg": round(home_5["points_for"], 1),
        "away_last_5_ppg": round(away_5["points_for"], 1),

        "home_last_10_ppg": round(home_10["points_for"], 1),
        "away_last_10_ppg": round(away_10["points_for"], 1),

        "home_points_allowed_last_10": round(home_10["points_allowed"], 1),
        "away_points_allowed_last_10": round(away_10["points_allowed"], 1),

        "home_pace_score_last_10": round(home_10["pace_score"], 1),
        "away_pace_score_last_10": round(away_10["pace_score"], 1),
    }
