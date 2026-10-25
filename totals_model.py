
import os
import pandas as pd

from injury_impact_engine import get_injury_impact
from pace_engine import calculate_matchup_pace
from points_engine import calculate_matchup_points, load_game_history
from betting_guards import (
    USE_STACKED_ADJUSTMENTS,
    EARLY_SEASON_MIN_GAMES,
    EARLY_SEASON_WINDOW_DAYS,
    classify_totals_edge,
)

NBA_AVG_TEAM_POINTS = 114


def safe_mean(values, default=114):
    try:
        values = pd.Series(values).dropna()
        if len(values) == 0:
            return default
        return float(values.mean())
    except Exception:
        return default


def load_history():
    """
    Load NBA game history from supported CSV files first.
    If none contains usable data, use points_engine's history loader,
    which supports outputs/training_dataset.parquet.
    """
    possible_files = [
        "data/historical_nba_scores.csv",
        "historical_nba_scores.csv",
        "historical_games.csv",
        "data/historical_games.csv",
        "historical_training_data.csv",
        "data/historical_training_data.csv",
    ]

    required_score_cols = [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    for file in possible_files:
        if not os.path.isfile(file):
            continue

        try:
            df = pd.read_csv(file)

            if not all(col in df.columns for col in required_score_cols):
                continue

            df["home_score"] = pd.to_numeric(
                df["home_score"], errors="coerce"
            )
            df["away_score"] = pd.to_numeric(
                df["away_score"], errors="coerce"
            )

            df = df.dropna(
                subset=[
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                ]
            )

            if df.empty:
                continue

            if "total_score" not in df.columns:
                df["total_score"] = df["home_score"] + df["away_score"]

            return df

        except Exception:
            continue

    # Fallback: use the loader that already reads the Parquet dataset.
    try:
        df = load_game_history()

        if df is None or df.empty:
            return pd.DataFrame()

        if not all(col in df.columns for col in required_score_cols):
            return pd.DataFrame()

        df = df.copy()
        df["home_score"] = pd.to_numeric(
            df["home_score"], errors="coerce"
        )
        df["away_score"] = pd.to_numeric(
            df["away_score"], errors="coerce"
        )

        df = df.dropna(
            subset=[
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        )

        if df.empty:
            return pd.DataFrame()

        if "total_score" not in df.columns:
            df["total_score"] = df["home_score"] + df["away_score"]

        return df

    except Exception as exc:
        print(f"Could not load NBA historical data: {exc}")
        return pd.DataFrame()


def games_in_recent_window(history_df, team_name, window_days):
    """Games a team played in the window_days before the latest game in history.
    Returns None if it cannot be computed (caller must then not block)."""
    try:
        d = history_df[["game_date", "home_team", "away_team"]].copy()
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
        d = d.dropna(subset=["game_date"])
        latest = d["game_date"].max()
        t = str(team_name).strip().lower()
        is_team = (
            (d["home_team"].astype(str).str.strip().str.lower() == t)
            | (d["away_team"].astype(str).str.strip().str.lower() == t)
        )
        in_window = d["game_date"] >= latest - pd.Timedelta(days=window_days)
        return int((is_team & in_window).sum())
    except Exception:
        return None


def default_team_stats():
    return {
        "last_5_scored": NBA_AVG_TEAM_POINTS,
        "last_10_scored": NBA_AVG_TEAM_POINTS,
        "last_10_allowed": NBA_AVG_TEAM_POINTS,
        "home_split": NBA_AVG_TEAM_POINTS,
        "away_split": NBA_AVG_TEAM_POINTS,
        "rest_days": 1,
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

    team_name = str(team_name).strip()

    team_games = history_df[
        (
            history_df["home_team"]
            .astype(str)
            .str.strip()
            .str.lower()
            == team_name.lower()
        )
        |
        (
            history_df["away_team"]
            .astype(str)
            .str.strip()
            .str.lower()
            == team_name.lower()
        )
    ].copy()

    if team_games.empty:
        return default_team_stats()

    if "game_date" in team_games.columns:
        team_games["game_date"] = pd.to_datetime(
            team_games["game_date"], errors="coerce"
        )
        team_games = team_games.dropna(subset=["game_date"])
        team_games = team_games.sort_values("game_date")

    if team_games.empty:
        return default_team_stats()

    recent_games = team_games.tail(10)

    rest_days = 1
    if "game_date" in team_games.columns and len(team_games) >= 2:
        try:
            latest_game = team_games.iloc[-1]["game_date"]
            previous_game = team_games.iloc[-2]["game_date"]
            rest_days = max(
                0,
                min(3, int((latest_game - previous_game).days) - 1),
            )
        except Exception:
            rest_days = 1

    scored = []
    allowed = []
    home_scored = []
    away_scored = []

    for _, row in recent_games.iterrows():
        if str(row["home_team"]).strip().lower() == team_name.lower():
            scored.append(row["home_score"])
            allowed.append(row["away_score"])
            home_scored.append(row["home_score"])
        else:
            scored.append(row["away_score"])
            allowed.append(row["home_score"])
            away_scored.append(row["away_score"])

    return {
        "last_5_scored": safe_mean(
            scored[-5:], NBA_AVG_TEAM_POINTS
        ),
        "last_10_scored": safe_mean(
            scored, NBA_AVG_TEAM_POINTS
        ),
        "last_10_allowed": safe_mean(
            allowed, NBA_AVG_TEAM_POINTS
        ),
        "home_split": safe_mean(
            home_scored, NBA_AVG_TEAM_POINTS
        ),
        "away_split": safe_mean(
            away_scored, NBA_AVG_TEAM_POINTS
        ),
        "rest_days": rest_days,
    }


def predict_game_total(home_team, away_team, bookmaker_total):
    history_df = load_history()

    history_rows = len(history_df)
    history_columns = str(list(history_df.columns))

    points_data = calculate_matchup_points(
        home_team=home_team,
        away_team=away_team,
        history_df=history_df,
        bookmaker_total=bookmaker_total,
    )

    projected_home_points = points_data["projected_home_points"]
    projected_away_points = points_data["projected_away_points"]
    raw_projected_total = points_data["projected_total"]

    home_stats = get_team_recent_stats(history_df, home_team)
    away_stats = get_team_recent_stats(history_df, away_team)

    pace_data = calculate_matchup_pace(
        history_df, home_team, away_team
    )

    home_pace_score = pace_data["home_pace"]
    away_pace_score = pace_data["away_pace"]
    combined_pace_score = pace_data["combined_pace"]
    pace_adjustment = pace_data["pace_adjustment"]
    pace_gap = pace_data["pace_gap"]

    home_offensive_rating = home_stats["last_10_scored"]
    away_offensive_rating = away_stats["last_10_scored"]

    offensive_adjustment = (
        (home_offensive_rating - NBA_AVG_TEAM_POINTS)
        + (away_offensive_rating - NBA_AVG_TEAM_POINTS)
    ) * 0.15

    home_defensive_rating = home_stats["last_10_allowed"]
    away_defensive_rating = away_stats["last_10_allowed"]

    defensive_adjustment = (
        (home_defensive_rating - NBA_AVG_TEAM_POINTS)
        + (away_defensive_rating - NBA_AVG_TEAM_POINTS)
    ) * 0.15

    home_split_advantage = (
        home_stats["home_split"] - NBA_AVG_TEAM_POINTS
    )
    away_split_advantage = (
        away_stats["away_split"] - NBA_AVG_TEAM_POINTS
    )

    home_away_adjustment = (
        home_split_advantage + away_split_advantage
    ) * 0.12

    home_rest_days = home_stats["rest_days"]
    away_rest_days = away_stats["rest_days"]

    rest_advantage = (home_rest_days + away_rest_days) - 2
    rest_adjustment = rest_advantage * 0.50

    # PATCH: these four re-count inputs the points engine already uses
    # (rest was applied twice). See USE_STACKED_ADJUSTMENTS in betting_guards.py.
    if not USE_STACKED_ADJUSTMENTS:
        offensive_adjustment = 0.0
        defensive_adjustment = 0.0
        home_away_adjustment = 0.0
        rest_adjustment = 0.0

    injury_data = get_injury_impact(home_team, away_team)

    home_injury_penalty = injury_data["home_injury_penalty"]
    away_injury_penalty = injury_data["away_injury_penalty"]
    injury_adjustment = injury_data["injury_adjustment"]

    projected_total = (
        raw_projected_total
        + pace_adjustment
        + offensive_adjustment
        + defensive_adjustment
        + home_away_adjustment
        + rest_adjustment
        + injury_adjustment
    )

    edge = projected_total - float(bookmaker_total)

    recommendation, confidence_note = classify_totals_edge(edge)

    # PATCH: early in a season the "last 10 games" are mostly last season's games.
    home_recent_games = games_in_recent_window(
        history_df, home_team, EARLY_SEASON_WINDOW_DAYS
    )
    away_recent_games = games_in_recent_window(
        history_df, away_team, EARLY_SEASON_WINDOW_DAYS
    )
    early_season = (
        EARLY_SEASON_MIN_GAMES > 0
        and home_recent_games is not None
        and away_recent_games is not None
        and min(home_recent_games, away_recent_games) < EARLY_SEASON_MIN_GAMES
    )
    if early_season:
        recommendation = "No Bet"
        confidence_note = (
            f"Early season: only {home_recent_games}/{away_recent_games} recent games "
            f"in the data (need {EARLY_SEASON_MIN_GAMES}); form inputs are mostly last season"
        )

    return {
        "history_rows": history_rows,
        "history_columns": history_columns,
        "early_season": early_season,
        "home_recent_games": home_recent_games,
        "away_recent_games": away_recent_games,
        "stacked_adjustments_active": USE_STACKED_ADJUSTMENTS,
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
        "points_engine_home_points": points_data["projected_home_points"],
        "points_engine_away_points": points_data["projected_away_points"],
        "points_engine_total": points_data["projected_total"],
        "points_engine_h2h_average_total": points_data["h2h_average_total"],
        "points_engine_h2h_adjustment": points_data["h2h_adjustment"],
        "points_engine_h2h_games_used": points_data["h2h_games_used"],
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
        "home_last_10_pace": pace_data["home_last_10_pace"],
        "home_last_20_pace": pace_data["home_last_20_pace"],
        "home_season_pace": pace_data["home_season_pace"],
        "home_venue_pace": pace_data["home_venue_pace"],
        "away_last_10_pace": pace_data["away_last_10_pace"],
        "away_last_20_pace": pace_data["away_last_20_pace"],
        "away_season_pace": pace_data["away_season_pace"],
        "away_venue_pace": pace_data["away_venue_pace"],
        "home_pace_games_used": pace_data["home_pace_games_used"],
        "away_pace_games_used": pace_data["away_pace_games_used"],
        "home_offensive_rating": round(home_offensive_rating, 2),
        "away_offensive_rating": round(away_offensive_rating, 2),
        "offensive_adjustment": round(offensive_adjustment, 2),
        "home_defensive_rating": round(home_defensive_rating, 2),
        "away_defensive_rating": round(away_defensive_rating, 2),
        "defensive_adjustment": round(defensive_adjustment, 2),
        "home_split": round(home_stats["home_split"], 2),
        "away_split": round(away_stats["away_split"], 2),
        "home_away_adjustment": round(home_away_adjustment, 2),
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_adjustment": round(rest_adjustment, 2),
        "home_injury_penalty": round(home_injury_penalty, 2),
        "away_injury_penalty": round(away_injury_penalty, 2),
        "injury_adjustment": round(injury_adjustment, 2),
        "home_missing_players": injury_data["home_missing_players"],
        "away_missing_players": injury_data["away_missing_players"],
    }


def predict_totals(
    home_team,
    away_team,
    sportsbook_total_line,
    history_df=None,
):
    return predict_game_total(
        home_team=home_team,
        away_team=away_team,
        bookmaker_total=sportsbook_total_line,
    )
