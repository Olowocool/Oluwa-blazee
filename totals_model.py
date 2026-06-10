# totals_model.py

import os
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


def load_history():
    possible_files = [
        "data/historical_nba_scores.csv",
        "historical_nba_scores.csv",
        "historical_games.csv",
        "data/historical_games.csv",
        "historical_training_data.csv",
        "data/historical_training_data.csv",
    ]
    for file in possible_files:
        if os.path.isfile(file):
            try:
                df = pd.read_csv(file)

                print("================================")
                print("FOUND HISTORY FILE:", file)
                print("ROWS:", len(df))
                print("COLUMNS:", list(df.columns))
                print("================================")

                return df

            except Exception as e:
                print("FAILED TO LOAD:", file)
                print(e)

    print("NO HISTORY FILE FOUND")
    return pd.DataFrame()


def default_team_stats():
    return {
        "last_5_scored": NBA_AVG_TEAM_POINTS,
        "last_10_scored": NBA_AVG_TEAM_POINTS,
        "last_10_allowed": NBA_AVG_TEAM_POINTS,
        "pace_score": PACE_BASELINE,
        "home_split": NBA_AVG_TEAM_POINTS,
        "away_split": NBA_AVG_TEAM_POINTS,
    }


def get_team_recent_stats(history_df, team_name):
    if history_df is None or history_df.empty:
        return default_team_stats()

    required_cols = [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    for col in required_cols:
        if col not in history_df.columns:
            return default_team_stats()

    team_games = history_df[
        (history_df["home_team"].astype(str).str.lower() == team_name.lower()) |
        (history_df["away_team"].astype(str).str.lower() == team_name.lower())
    ].copy()

    if team_games.empty:
        return default_team_stats()

    team_games = team_games.tail(10)

    scored = []
    allowed = []
    home_scored = []
    away_scored = []

    for _, row in team_games.iterrows():
        if str(row["home_team"]).lower() == team_name.lower():
            scored.append(row["home_score"])
            allowed.append(row["away_score"])
            home_scored.append(row["home_score"])
        else:
            scored.append(row["away_score"])
            allowed.append(row["home_score"])
            away_scored.append(row["away_score"])

    last_5_scored = safe_mean(scored[-5:], NBA_AVG_TEAM_POINTS)
    last_10_scored = safe_mean(scored, NBA_AVG_TEAM_POINTS)
    last_10_allowed = safe_mean(allowed, NBA_AVG_TEAM_POINTS)

    home_split = safe_mean(home_scored, NBA_AVG_TEAM_POINTS)
    away_split = safe_mean(away_scored, NBA_AVG_TEAM_POINTS)

    pace_score = last_10_scored + last_10_allowed

    return {
        "last_5_scored": last_5_scored,
        "last_10_scored": last_10_scored,
        "last_10_allowed": last_10_allowed,
        "pace_score": pace_score,
        "home_split": home_split,
        "away_split": away_split,
    }


def predict_game_total(home_team, away_team, bookmaker_total):
    history_df = load_history()

    history_rows = len(history_df)
    history_columns = str(list(history_df.columns))

    print("================================")
    print("TOTALS MODEL DEBUG")
    print("History Rows:", history_rows)
    print("History Columns:", history_columns)

    if history_rows > 0:
        print(history_df.head())
    else:
        print("NO HISTORY DATA FOUND")

    print("================================")

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

    home_pace_score = home_stats["pace_score"]
    away_pace_score = away_stats["pace_score"]

    combined_pace_score = (home_pace_score + away_pace_score) / 2
    pace_gap = combined_pace_score - PACE_BASELINE
    pace_adjustment = pace_gap * 0.20

    home_offensive_rating = home_stats["last_10_scored"]
    away_offensive_rating = away_stats["last_10_scored"]

    offensive_adjustment = (
        (home_offensive_rating - NBA_AVG_TEAM_POINTS) +
        (away_offensive_rating - NBA_AVG_TEAM_POINTS)
    ) * 0.25

    home_defensive_rating = home_stats["last_10_allowed"]
    away_defensive_rating = away_stats["last_10_allowed"]

    defensive_adjustment = (
        (home_defensive_rating - NBA_AVG_TEAM_POINTS) +
        (away_defensive_rating - NBA_AVG_TEAM_POINTS)
    ) * 0.25

    home_split_advantage = home_stats["home_split"] - NBA_AVG_TEAM_POINTS
    away_split_advantage = away_stats["away_split"] - NBA_AVG_TEAM_POINTS

    home_away_adjustment = (
        home_split_advantage +
        away_split_advantage
    ) * 0.20

    projected_total = (
        raw_projected_total
        + pace_adjustment
        + offensive_adjustment
        + defensive_adjustment
        + home_away_adjustment
    )

    edge = projected_total - float(bookmaker_total)

    if edge >= 5:
        recommendation = "Strong Over"
        confidence_note = "Strong Over edge"
    elif edge >= 2.5:
        recommendation = "Lean Over"
        confidence_note = "Small Over edge"
    elif edge <= -5:
        recommendation = "Strong Under"
        confidence_note = "Strong Under edge"
    elif edge <= -2.5:
        recommendation = "Lean Under"
        confidence_note = "Small Under edge"
    else:
        recommendation = "No Bet"
        confidence_note = "Edge too small"

    return {
        "history_rows": history_rows,
        "history_columns": history_columns,

        "home_team": home_team,
        "away_team": away_team,

        "projected_total": round(projected_total, 2),
        "bookmaker_total": round(float(bookmaker_total), 2),
        "edge": round(edge, 2),
        "recommendation": recommendation,
        "confidence_note": confidence_note,

        "raw_projected_total": round(raw_projected_total, 2),
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

        "home_defensive_rating": round(home_defensive_rating, 2),
        "away_defensive_rating": round(away_defensive_rating, 2),
        "defensive_adjustment": round(defensive_adjustment, 2),

        "home_split": round(home_stats["home_split"], 2),
        "away_split": round(away_stats["away_split"], 2),
        "home_away_adjustment": round(home_away_adjustment, 2),
    }


def predict_totals(home_team, away_team, sportsbook_total_line, history_df=None):
    return predict_game_total(
        home_team=home_team,
        away_team=away_team,
        bookmaker_total=sportsbook_total_line
    )
