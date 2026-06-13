# pace_engine.py

import pandas as pd


NBA_LEAGUE_AVG_TOTAL = 225.0
PACE_BASELINE = 100.0


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _prepare_history(history_df):
    """
    Normalizes the historical scores file for pace calculations.

    Expected useful columns:
    - game_date
    - home_team
    - away_team
    - home_score
    - away_score
    - total_score

    This is compatible with historical_backfill_engine.py.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()

    required = [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    df["home_score"] = pd.to_numeric(
        df["home_score"],
        errors="coerce"
    )

    df["away_score"] = pd.to_numeric(
        df["away_score"],
        errors="coerce"
    )

    if "total_score" not in df.columns:
        df["total_score"] = (
            df["home_score"] +
            df["away_score"]
        )
    else:
        df["total_score"] = pd.to_numeric(
            df["total_score"],
            errors="coerce"
        )

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(
            df["game_date"],
            errors="coerce"
        )

        df = df.dropna(subset=["game_date"])

        df = df.sort_values("game_date")

    df = df.dropna(
        subset=[
            "home_score",
            "away_score",
            "total_score",
        ]
    )

    return df


def _team_games(df, team_name):
    if df.empty:
        return pd.DataFrame()

    team_name = str(team_name).strip().lower()

    games = df[
        (df["home_team"].astype(str).str.lower() == team_name)
        |
        (df["away_team"].astype(str).str.lower() == team_name)
    ].copy()

    if "game_date" in games.columns:
        games = games.sort_values("game_date")

    return games


def _average_total_as_pace(games, default=PACE_BASELINE):
    """
    Converts recent game total environment into a pace-like score.

    True NBA pace requires possessions. Since our current historical file is
    score-only, this estimates pace pressure from total game environment.
    It is intentionally centered around 100 so the totals model can use it
    cleanly as a pace adjustment.
    """
    if games is None or games.empty:
        return default

    avg_total = _safe_float(
        games["total_score"].mean(),
        NBA_LEAGUE_AVG_TOTAL
    )

    if avg_total <= 0:
        avg_total = NBA_LEAGUE_AVG_TOTAL

    pace = PACE_BASELINE + (
        (avg_total - NBA_LEAGUE_AVG_TOTAL)
        / NBA_LEAGUE_AVG_TOTAL
        * 20
    )

    return round(float(pace), 2)


def _venue_games(games, team_name, venue):
    """
    venue:
    - "home" means games where team_name was home
    - "away" means games where team_name was away
    """
    if games is None or games.empty:
        return pd.DataFrame()

    team_name = str(team_name).strip().lower()

    if venue == "home":
        return games[
            games["home_team"].astype(str).str.lower() == team_name
        ].copy()

    if venue == "away":
        return games[
            games["away_team"].astype(str).str.lower() == team_name
        ].copy()

    return pd.DataFrame()


def calculate_team_advanced_pace(
    history_df,
    team_name,
    venue="neutral"
):
    """
    Advanced Pace Engine v2.

    Uses score-based historical environment to estimate:
    - last 10 pace
    - last 20 pace
    - season pace
    - venue pace

    This is not possession-level pace yet. It is a strong upgrade over the
    previous basic pace because it separates recent trend, longer trend,
    season environment, and home/away environment.
    """

    df = _prepare_history(history_df)

    if df.empty:
        return {
            "team": team_name,
            "last_10_pace": PACE_BASELINE,
            "last_20_pace": PACE_BASELINE,
            "season_pace": PACE_BASELINE,
            "venue_pace": PACE_BASELINE,
            "weighted_pace": PACE_BASELINE,
            "games_used": 0,
        }

    games = _team_games(df, team_name)

    if games.empty:
        return {
            "team": team_name,
            "last_10_pace": PACE_BASELINE,
            "last_20_pace": PACE_BASELINE,
            "season_pace": PACE_BASELINE,
            "venue_pace": PACE_BASELINE,
            "weighted_pace": PACE_BASELINE,
            "games_used": 0,
        }

    last_10 = games.tail(10)
    last_20 = games.tail(20)

    last_10_pace = _average_total_as_pace(last_10)
    last_20_pace = _average_total_as_pace(last_20)
    season_pace = _average_total_as_pace(games)

    venue_games = _venue_games(
        games,
        team_name,
        venue
    )

    if venue_games.empty:
        venue_pace = season_pace
    else:
        venue_pace = _average_total_as_pace(
            venue_games.tail(20)
        )

    weighted_pace = (
        last_10_pace * 0.40
        + last_20_pace * 0.25
        + season_pace * 0.20
        + venue_pace * 0.15
    )

    return {
        "team": team_name,
        "last_10_pace": round(float(last_10_pace), 2),
        "last_20_pace": round(float(last_20_pace), 2),
        "season_pace": round(float(season_pace), 2),
        "venue_pace": round(float(venue_pace), 2),
        "weighted_pace": round(float(weighted_pace), 2),
        "games_used": int(len(games)),
    }


def calculate_matchup_pace(
    history_df,
    home_team,
    away_team
):
    """
    Returns matchup-level pace information for the totals model.
    """

    home_pace = calculate_team_advanced_pace(
        history_df,
        home_team,
        venue="home"
    )

    away_pace = calculate_team_advanced_pace(
        history_df,
        away_team,
        venue="away"
    )

    combined_pace = (
        home_pace["weighted_pace"]
        + away_pace["weighted_pace"]
    ) / 2

    pace_gap = combined_pace - PACE_BASELINE

    # Each pace point above/below 100 moves the total by about 1.25 points.
    # Example: combined pace 102.0 -> +2.5 total points.
    pace_adjustment = pace_gap * 2.5

    return {
        "home_pace": round(float(home_pace["weighted_pace"]), 2),
        "away_pace": round(float(away_pace["weighted_pace"]), 2),
        "combined_pace": round(float(combined_pace), 2),
        "pace_gap": round(float(pace_gap), 2),
        "pace_adjustment": round(float(pace_adjustment), 2),

        "home_last_10_pace": home_pace["last_10_pace"],
        "home_last_20_pace": home_pace["last_20_pace"],
        "home_season_pace": home_pace["season_pace"],
        "home_venue_pace": home_pace["venue_pace"],

        "away_last_10_pace": away_pace["last_10_pace"],
        "away_last_20_pace": away_pace["last_20_pace"],
        "away_season_pace": away_pace["season_pace"],
        "away_venue_pace": away_pace["venue_pace"],

        "home_pace_games_used": home_pace["games_used"],
        "away_pace_games_used": away_pace["games_used"],
    }

