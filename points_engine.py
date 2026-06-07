import os
import pandas as pd


DATA_PATH = "outputs/training_dataset.parquet"


def safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_game_history():
    if not os.path.exists(DATA_PATH):
        return None

    try:
        return pd.read_parquet(DATA_PATH)
    except Exception:
        return None


def calculate_team_points_features(team_name, history_df, last_n=10):
    if history_df is None or history_df.empty:
        return {
            "ppg": 112,
            "points_allowed": 112,
            "net_points": 0,
            "avg_margin": 0,
        }

    team_games = history_df[
        (history_df["home_team_name"] == team_name)
        |
        (history_df["away_team_name"] == team_name)
    ].copy()

    if team_games.empty:
        return {
            "ppg": 112,
            "points_allowed": 112,
            "net_points": 0,
            "avg_margin": 0,
        }

    if "date" in team_games.columns:
        team_games["date"] = pd.to_datetime(
            team_games["date"],
            errors="coerce"
        )

        team_games = team_games.sort_values(
            "date"
        )

    recent_games = team_games.tail(last_n)

    points_for = []
    points_allowed = []
    margins = []

    for _, row in recent_games.iterrows():

        home_team = row.get("home_team_name")
        away_team = row.get("away_team_name")

        home_points = safe_number(
            row.get("home_team_score", row.get("home_points", 0))
        )

        away_points = safe_number(
            row.get("away_team_score", row.get("away_points", 0))
        )

        if home_team == team_name:
            scored = home_points
            allowed = away_points
        elif away_team == team_name:
            scored = away_points
            allowed = home_points
        else:
            continue

        if scored <= 0 or allowed <= 0:
            continue

        points_for.append(scored)
        points_allowed.append(allowed)
        margins.append(scored - allowed)

    if not points_for:
        return {
            "ppg": 112,
            "points_allowed": 112,
            "net_points": 0,
            "avg_margin": 0,
        }

    ppg = sum(points_for) / len(points_for)
    papg = sum(points_allowed) / len(points_allowed)
    avg_margin = sum(margins) / len(margins)

    return {
        "ppg": round(ppg, 2),
        "points_allowed": round(papg, 2),
        "net_points": round(ppg - papg, 2),
        "avg_margin": round(avg_margin, 2),
    }


def add_points_features(df):
    df = df.copy()

    history_df = load_game_history()

    for idx, row in df.iterrows():

        home_team = row.get("home_team", row.get("home_team_name", ""))
        away_team = row.get("away_team", row.get("away_team_name", ""))

        home_last_5 = calculate_team_points_features(
            home_team,
            history_df,
            last_n=5
        )

        away_last_5 = calculate_team_points_features(
            away_team,
            history_df,
            last_n=5
        )

        home_last_10 = calculate_team_points_features(
            home_team,
            history_df,
            last_n=10
        )

        away_last_10 = calculate_team_points_features(
            away_team,
            history_df,
            last_n=10
        )

        df.loc[idx, "home_ppg_last_5"] = home_last_5["ppg"]
        df.loc[idx, "away_ppg_last_5"] = away_last_5["ppg"]

        df.loc[idx, "home_ppg_last_10"] = home_last_10["ppg"]
        df.loc[idx, "away_ppg_last_10"] = away_last_10["ppg"]

        df.loc[idx, "home_points_allowed_last_5"] = home_last_5["points_allowed"]
        df.loc[idx, "away_points_allowed_last_5"] = away_last_5["points_allowed"]

        df.loc[idx, "home_points_allowed_last_10"] = home_last_10["points_allowed"]
        df.loc[idx, "away_points_allowed_last_10"] = away_last_10["points_allowed"]

        df.loc[idx, "home_net_points_last_10"] = home_last_10["net_points"]
        df.loc[idx, "away_net_points_last_10"] = away_last_10["net_points"]

        df.loc[idx, "home_avg_margin_last_10"] = home_last_10["avg_margin"]
        df.loc[idx, "away_avg_margin_last_10"] = away_last_10["avg_margin"]

        df.loc[idx, "ppg_diff_last_5"] = (
            home_last_5["ppg"] - away_last_5["ppg"]
        )

        df.loc[idx, "ppg_diff_last_10"] = (
            home_last_10["ppg"] - away_last_10["ppg"]
        )

        df.loc[idx, "points_allowed_diff_last_10"] = (
            away_last_10["points_allowed"]
            - home_last_10["points_allowed"]
        )

        df.loc[idx, "net_points_diff_last_10"] = (
            home_last_10["net_points"]
            - away_last_10["net_points"]
        )

        df.loc[idx, "avg_margin_diff_last_10"] = (
            home_last_10["avg_margin"]
            - away_last_10["avg_margin"]
        )

    return df
