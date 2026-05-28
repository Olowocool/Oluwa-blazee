import pandas as pd
from datetime import datetime


STAR_PLAYER_PENALTIES = {
    "Boston Celtics": 0.06,
    "Denver Nuggets": 0.07,
    "Oklahoma City Thunder": 0.06,
    "Minnesota Timberwolves": 0.05,
    "Cleveland Cavaliers": 0.05,
    "San Antonio Spurs": 0.08,
    "New York Knicks": 0.05,
    "Los Angeles Lakers": 0.07,
    "Golden State Warriors": 0.07,
}


def safe_date(value):
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def calculate_rest_days(team, game_date, df):
    game_date = safe_date(game_date)

    if game_date is None:
        return 2

    previous_games = df[
        (
            (df["home_team"] == team) |
            (df["away_team"] == team)
        )
        &
        (pd.to_datetime(df["game_date"], errors="coerce") < game_date)
    ]

    if previous_games.empty:
        return 2

    previous_date = pd.to_datetime(
        previous_games["game_date"],
        errors="coerce"
    ).max()

    rest_days = (game_date - previous_date).days - 1

    return max(rest_days, 0)


def estimate_injury_penalty(team, is_star_out=False):
    if is_star_out:
        return STAR_PLAYER_PENALTIES.get(team, 0.04)

    return 0


def add_injury_rest_features(df):
    df = df.copy()

    if "game_date" not in df.columns:
        return df

    for idx, row in df.iterrows():
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        game_date = row.get("game_date", "")

        home_rest = calculate_rest_days(
            home_team,
            game_date,
            df
        )

        away_rest = calculate_rest_days(
            away_team,
            game_date,
            df
        )

        df.loc[idx, "home_rest_days"] = home_rest
        df.loc[idx, "away_rest_days"] = away_rest

        df.loc[idx, "home_back_to_back"] = 1 if home_rest == 0 else 0
        df.loc[idx, "away_back_to_back"] = 1 if away_rest == 0 else 0

        df.loc[idx, "home_injury_penalty"] = estimate_injury_penalty(
            home_team,
            row.get("home_star_out", False)
        )

        df.loc[idx, "away_injury_penalty"] = estimate_injury_penalty(
            away_team,
            row.get("away_star_out", False)
        )

        df.loc[idx, "rest_advantage"] = home_rest - away_rest

        df.loc[idx, "fatigue_edge"] = (
            df.loc[idx, "away_back_to_back"]
            - df.loc[idx, "home_back_to_back"]
        )

    return df
