import pandas as pd


def safe_float(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_advanced_features(df):
    df = df.copy()

    defaults = {
        "home_rest_days": 2,
        "away_rest_days": 2,
        "home_off_rating": 112,
        "away_off_rating": 112,
        "home_def_rating": 112,
        "away_def_rating": 112,
        "home_pace": 100,
        "away_pace": 100,
        "home_recent_wins": 5,
        "away_recent_wins": 5,
        "home_injury_penalty": 0,
        "away_injury_penalty": 0,
        "home_line_move_pct": 0,
        "away_line_move_pct": 0,
        "sharp_books_support": 0,
        "total_books": 1,
        "home_home_win_pct": 0.5,
        "away_away_win_pct": 0.5,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

        df[col] = df[col].apply(lambda x: safe_float(x, default))

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
