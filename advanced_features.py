import pandas as pd
from injury_rest_engine import add_injury_rest_features
from nba_stats_api import get_team_stats
from market_intelligence_engine import generate_market_intelligence

def safe_float(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_advanced_features(df):
    df = df.copy()
    df = add_injury_rest_features(df)
    df = generate_market_intelligence(df)

    for idx, row in df.iterrows():
        home_stats = get_team_stats(row.get("home_team", ""))
        away_stats = get_team_stats(row.get("away_team", ""))

        df.loc[idx, "home_off_rating"] = home_stats["off_rating"]
        df.loc[idx, "away_off_rating"] = away_stats["off_rating"]

        df.loc[idx, "home_def_rating"] = home_stats["def_rating"]
        df.loc[idx, "away_def_rating"] = away_stats["def_rating"]

        df.loc[idx, "home_pace"] = home_stats["pace"]
        df.loc[idx, "away_pace"] = away_stats["pace"]

        df.loc[idx, "home_recent_wins"] = home_stats["recent_wins"]
        df.loc[idx, "away_recent_wins"] = away_stats["recent_wins"]

        df.loc[idx, "home_rest_days"] = safe_float(row.get("home_rest_days", 2), 2)
        df.loc[idx, "away_rest_days"] = safe_float(row.get("away_rest_days", 2), 2)

        df.loc[idx, "home_injury_penalty"] = safe_float(row.get("home_injury_penalty", 0), 0)
        df.loc[idx, "away_injury_penalty"] = safe_float(row.get("away_injury_penalty", 0), 0)

        df.loc[idx, "home_line_move_pct"] = safe_float(row.get("home_line_move_pct", 0), 0)
        df.loc[idx, "away_line_move_pct"] = safe_float(row.get("away_line_move_pct", 0), 0)

        df.loc[idx, "sharp_books_support"] = safe_float(row.get("sharp_books_support", 0), 0)
        df.loc[idx, "total_books"] = safe_float(row.get("total_books", 1), 1)

        df.loc[idx, "home_home_win_pct"] = safe_float(row.get("home_home_win_pct", 0.6), 0.6)
        df.loc[idx, "away_away_win_pct"] = safe_float(row.get("away_away_win_pct", 0.4), 0.4)

    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["off_rating_diff"] = df["home_off_rating"] - df["away_off_rating"]
    df["def_rating_diff"] = df["away_def_rating"] - df["home_def_rating"]
    df["pace_diff"] = df["home_pace"] - df["away_pace"]
    df["recent_form_diff"] = df["home_recent_wins"] - df["away_recent_wins"]
    df["injury_diff"] = df["away_injury_penalty"] - df["home_injury_penalty"]
    df["line_movement_diff"] = df["home_line_move_pct"] - df["away_line_move_pct"]

    df["sharp_support_pct"] = df.apply(
        lambda row: safe_float(row["sharp_books_support"]) / max(safe_float(row["total_books"], 1), 1),
        axis=1
    )

    df["home_venue_edge"] = df["home_home_win_pct"] - df["away_away_win_pct"]

    return df
