# totals_model.py

import pandas as pd


NBA_AVG_TEAM_POINTS = 114
PACE_BASELINE = 228


def safe_mean(values, default=114):
    try:
        values = pd.Series(values).dropna()
        if len(values) == 0:
            return default
        return float(values.mean())
    except Exception:
        return default


def get_team_recent_stats(history_df, team_name):
    team_games = history_df[
        (history_df["home_team"] == team_name) |
        (history_df["away_team"] == team_name)
    ].copy()

    if team_games.empty:
        return {
            "last_5_scored": NBA_AVG_TEAM_POINTS,
            "last_10_scored": NBA_AVG_TEAM_POINTS,
            "last_10_allowed": NBA_AVG_TEAM_POINTS,
            "pace_score": NBA_AVG_TEAM_POINTS,
        }

    team_games = team_games.tail(10)

    scored = []
    allowed = []

    for _, row in team_games.iterrows():
        if row["home_team"] == team_name:
            scored.append(row["home_score"])
            allowed.append(row["away_score"])
        else:
            scored.append(row["away_score"])
            allowed.append(row["home_score"])

    last_5_scored = safe_mean(scored[-5:], NBA_AVG_TEAM_POINTS)
    last_10_scored = safe_mean(scored, NBA_AVG_TEAM_POINTS)
    last_10_allowed = safe_mean(allowed, NBA_AVG_TEAM_POINTS)

    pace_score = last_10_scored + last_10_allowed

    return {
        "last_5_scored": last_5_scored,
        "last_10_scored": last_10_scored,
        "last_10_allowed": last_10_allowed,
        "pace_score": pace_score,
    }


def predict_totals(home_team, away_team, sportsbook_total_line, history_df):
    home_stats = get_team_recent_stats(history_df, home_team)
    away_stats = get_team_recent_stats(history_df, away_team)

    projected_home_points = (
        home_stats["last_5_scored"] * 0.35 +
        home_stats["last_10_scored"] * 0.35 +
        away_stats["last_10_allowed"] * 0.30
    )

    projected_away_points = (
        away_stats["last_5_scored"] * 0.35 +
        away_stats["last_10_scored"] * 0.35 +
        home_stats["last_10_allowed"] * 0.30
    )

    raw_projected_total = projected_home_points + projected_away_points

    # -----------------------------
    # PACE ADJUSTMENT
    # -----------------------------

    home_pace_score = home_stats["pace_score"]
    away_pace_score = away_stats["pace_score"]

    combined_pace_score = (home_pace_score + away_pace_score) / 2
    pace_gap = combined_pace_score - PACE_BASELINE

    pace_adjustment = pace_gap * 0.20

    # -----------------------------
    # OFFENSIVE RATING ADJUSTMENT
    # -----------------------------

    home_offensive_rating = home_stats["last_10_scored"]
    away_offensive_rating = away_stats["last_10_scored"]

    home_off_edge = home_offensive_rating - NBA_AVG_TEAM_POINTS
    away_off_edge = away_offensive_rating - NBA_AVG_TEAM_POINTS

    offensive_adjustment = (home_off_edge + away_off_edge) * 0.25

    # -----------------------------
    # FINAL TOTAL
    # -----------------------------

    projected_total = raw_projected_total + pace_adjustment + offensive_adjustment

    edge = projected_total - sportsbook_total_line

    if edge >= 5:
        recommendation = "Strong Over"
    elif edge >= 2.5:
        recommendation = "Lean Over — Small Over edge"
    elif edge <= -5:
        recommendation = "Strong Under"
    elif edge <= -2.5:
        recommendation = "Lean Under — Small Under edge"
    else:
        recommendation = "No Bet — Edge too small"

    return {
        "home_team": home_team,
        "away_team": away_team,

        "projected_total": round(projected_total, 2),
        "raw_projected_total": round(raw_projected_total, 2),
        "bookmaker_line": round(float(sportsbook_total_line), 2),
        "edge": round(edge, 2),
        "recommendation": recommendation,

        "projected_home_points": round(projected_home_points, 2),
        "projected_away_points": round(projected_away_points, 2),

        "home_last_5_scored": round(home_stats["last_5_scored"], 2),
        "away_last_5_scored": round(away_stats["last_5_scored"], 2),
        "home_last_10_scored": round(home_stats["last_10_scored"], 2),
        "away_last_10_scored": round(away_stats["last_10_scored"], 2),
        "home_last_10_allowed": round(home_stats["last_10_allowed"], 2),
        "away_last_10_allowed": round(away_stats["last_10_allowed"], 2),

        "home_pace_score": round(home_pace_score, 2),
        "away_pace_score": round(away_pace_score, 2),
        "combined_pace_score": round(combined_pace_score, 2),
        "pace_gap": round(pace_gap, 2),
        "pace_adjustment": round(pace_adjustment, 2),

        "home_offensive_rating": round(home_offensive_rating, 2),
        "away_offensive_rating": round(away_offensive_rating, 2),
        "offensive_adjustment": round(offensive_adjustment, 2),
    }
