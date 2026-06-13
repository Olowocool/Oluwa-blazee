# points_engine.py
# Points Engine V2

import os
import pandas as pd

TRAINING_DATA_PATH = "outputs/training_dataset.parquet"

HISTORICAL_SCORE_FILES = [
    "data/historical_nba_scores.csv",
    "historical_nba_scores.csv",
    "historical_games.csv",
    "data/historical_games.csv",
]

NBA_AVG_TEAM_POINTS = 114.0
NBA_AVG_TOTAL = 228.0


def safe_number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_team_name(name):
    return str(name or "").strip()


def standardize_score_history(df):
    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(
        subset=[
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )

    df["total_score"] = df["home_score"] + df["away_score"]

    return df.sort_values("game_date")


def standardize_training_history(df):
    df = df.copy()

    rename_map = {
        "date": "game_date",
        "home_team_name": "home_team",
        "away_team_name": "away_team",
        "home_team_score": "home_score",
        "away_team_score": "away_score",
        "home_points": "home_score",
        "away_points": "away_score",
    }

    df = df.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in df.columns
        }
    )

    required = [
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    return standardize_score_history(df)


def load_game_history():
    for path in HISTORICAL_SCORE_FILES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)

                required = [
                    "game_date",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                ]

                if all(col in df.columns for col in required):
                    return standardize_score_history(df)
            except Exception:
                pass

    if os.path.exists(TRAINING_DATA_PATH):
        try:
            return standardize_training_history(
                pd.read_parquet(TRAINING_DATA_PATH)
            )
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def default_points_features(team_name):
    return {
        "team": team_name,
        "games_used": 0,
        "ppg": NBA_AVG_TEAM_POINTS,
        "points_allowed": NBA_AVG_TEAM_POINTS,
        "net_points": 0.0,
        "avg_margin": 0.0,
        "home_split": NBA_AVG_TEAM_POINTS,
        "away_split": NBA_AVG_TEAM_POINTS,
        "avg_total_environment": NBA_AVG_TOTAL,
    }


def get_team_games(history_df, team_name):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    team_name = normalize_team_name(team_name).lower()

    games = history_df[
        (history_df["home_team"].astype(str).str.lower() == team_name)
        |
        (history_df["away_team"].astype(str).str.lower() == team_name)
    ].copy()

    if not games.empty:
        games = games.sort_values("game_date")

    return games


def get_head_to_head_games(history_df, home_team, away_team, limit=10):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    home_team = normalize_team_name(home_team).lower()
    away_team = normalize_team_name(away_team).lower()

    h2h = history_df[
        (
            (history_df["home_team"].astype(str).str.lower() == home_team)
            &
            (history_df["away_team"].astype(str).str.lower() == away_team)
        )
        |
        (
            (history_df["home_team"].astype(str).str.lower() == away_team)
            &
            (history_df["away_team"].astype(str).str.lower() == home_team)
        )
    ].copy()

    if not h2h.empty:
        h2h = h2h.sort_values("game_date").tail(limit)

    return h2h


def calculate_team_points_features(team_name, history_df=None, last_n=10):
    if history_df is None:
        history_df = load_game_history()

    if history_df is None or history_df.empty:
        return default_points_features(team_name)

    team_games = get_team_games(history_df, team_name)

    if team_games.empty:
        return default_points_features(team_name)

    recent_games = team_games.tail(last_n)

    points_for = []
    points_allowed = []
    margins = []
    home_points = []
    away_points = []
    totals = []

    team_name_lc = normalize_team_name(team_name).lower()

    for _, row in recent_games.iterrows():
        home_team = str(row.get("home_team", "")).lower()
        away_team = str(row.get("away_team", "")).lower()

        home_score = safe_number(row.get("home_score"), 0)
        away_score = safe_number(row.get("away_score"), 0)

        if home_score <= 0 or away_score <= 0:
            continue

        if home_team == team_name_lc:
            scored = home_score
            allowed = away_score
            home_points.append(scored)
        elif away_team == team_name_lc:
            scored = away_score
            allowed = home_score
            away_points.append(scored)
        else:
            continue

        points_for.append(scored)
        points_allowed.append(allowed)
        margins.append(scored - allowed)
        totals.append(home_score + away_score)

    if not points_for:
        return default_points_features(team_name)

    ppg = sum(points_for) / len(points_for)
    papg = sum(points_allowed) / len(points_allowed)
    avg_margin = sum(margins) / len(margins)
    avg_total_environment = sum(totals) / len(totals)

    home_split = sum(home_points) / len(home_points) if home_points else ppg
    away_split = sum(away_points) / len(away_points) if away_points else ppg

    return {
        "team": team_name,
        "games_used": len(points_for),
        "ppg": round(ppg, 2),
        "points_allowed": round(papg, 2),
        "net_points": round(ppg - papg, 2),
        "avg_margin": round(avg_margin, 2),
        "home_split": round(home_split, 2),
        "away_split": round(away_split, 2),
        "avg_total_environment": round(avg_total_environment, 2),
    }


def calculate_team_rest_days(team_name, history_df):
    games = get_team_games(history_df, team_name)

    if games.empty or len(games) < 2:
        return 1

    try:
        latest_date = games.iloc[-1]["game_date"]
        previous_date = games.iloc[-2]["game_date"]
        rest_days = int((latest_date - previous_date).days) - 1
        return max(0, min(3, rest_days))
    except Exception:
        return 1


def calculate_projected_team_points(
    team_name,
    opponent_name,
    history_df,
    venue="neutral"
):
    team_last_5 = calculate_team_points_features(
        team_name,
        history_df,
        last_n=5
    )

    team_last_10 = calculate_team_points_features(
        team_name,
        history_df,
        last_n=10
    )

    opponent_last_10 = calculate_team_points_features(
        opponent_name,
        history_df,
        last_n=10
    )

    if venue == "home":
        venue_points = team_last_10["home_split"]
        venue_adjustment = 1.5
    elif venue == "away":
        venue_points = team_last_10["away_split"]
        venue_adjustment = -1.0
    else:
        venue_points = team_last_10["ppg"]
        venue_adjustment = 0.0

    projected_points = (
        team_last_5["ppg"] * 0.30
        + team_last_10["ppg"] * 0.25
        + opponent_last_10["points_allowed"] * 0.25
        + venue_points * 0.20
        + venue_adjustment
    )

    return {
        "team": team_name,
        "opponent": opponent_name,
        "venue": venue,
        "projected_points": round(projected_points, 2),
        "last_5_ppg": team_last_5["ppg"],
        "last_10_ppg": team_last_10["ppg"],
        "opponent_last_10_allowed": opponent_last_10["points_allowed"],
        "venue_points": venue_points,
        "venue_adjustment": venue_adjustment,
        "net_points": team_last_10["net_points"],
        "avg_margin": team_last_10["avg_margin"],
        "games_used": team_last_10["games_used"],
    }


def calculate_matchup_points(
    home_team,
    away_team,
    history_df=None,
    bookmaker_total=None
):
    if history_df is None:
        history_df = load_game_history()

    if history_df is None:
        history_df = pd.DataFrame()

    home_projection = calculate_projected_team_points(
        home_team,
        away_team,
        history_df,
        venue="home"
    )

    away_projection = calculate_projected_team_points(
        away_team,
        home_team,
        history_df,
        venue="away"
    )

    raw_home_points = home_projection["projected_points"]
    raw_away_points = away_projection["projected_points"]
    raw_total = raw_home_points + raw_away_points

    h2h_games = get_head_to_head_games(
        history_df,
        home_team,
        away_team,
        limit=10
    )

    if h2h_games.empty:
        h2h_average_total = raw_total
        h2h_adjustment = 0.0
        h2h_games_used = 0
    else:
        h2h_average_total = safe_number(
            h2h_games["total_score"].mean(),
            raw_total
        )
        h2h_adjustment = (h2h_average_total - raw_total) * 0.15
        h2h_games_used = len(h2h_games)

    home_rest = calculate_team_rest_days(home_team, history_df)
    away_rest = calculate_team_rest_days(away_team, history_df)

    rest_adjustment = ((home_rest + away_rest) - 2) * 0.50

    projected_home_points = raw_home_points + (h2h_adjustment / 2)
    projected_away_points = raw_away_points + (h2h_adjustment / 2)

    projected_total = (
        projected_home_points
        + projected_away_points
        + rest_adjustment
    )

    edge = None
    recommendation = "No Bet"
    confidence_note = "No bookmaker total available"

    if bookmaker_total is not None:
        try:
            bookmaker_total = float(bookmaker_total)
            edge = projected_total - bookmaker_total

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
        except Exception:
            bookmaker_total = None
            edge = None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "projected_home_points": round(projected_home_points, 2),
        "projected_away_points": round(projected_away_points, 2),
        "projected_total": round(projected_total, 2),
        "raw_home_points": round(raw_home_points, 2),
        "raw_away_points": round(raw_away_points, 2),
        "raw_total": round(raw_total, 2),
        "bookmaker_total": round(float(bookmaker_total), 2) if bookmaker_total is not None else None,
        "edge": round(float(edge), 2) if edge is not None else None,
        "recommendation": recommendation,
        "confidence_note": confidence_note,
        "home_last_5_ppg": home_projection["last_5_ppg"],
        "away_last_5_ppg": away_projection["last_5_ppg"],
        "home_last_10_ppg": home_projection["last_10_ppg"],
        "away_last_10_ppg": away_projection["last_10_ppg"],
        "home_opponent_allowed": home_projection["opponent_last_10_allowed"],
        "away_opponent_allowed": away_projection["opponent_last_10_allowed"],
        "home_venue_points": home_projection["venue_points"],
        "away_venue_points": away_projection["venue_points"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_adjustment": round(rest_adjustment, 2),
        "h2h_average_total": round(h2h_average_total, 2),
        "h2h_adjustment": round(h2h_adjustment, 2),
        "h2h_games_used": h2h_games_used,
        "home_games_used": home_projection["games_used"],
        "away_games_used": away_projection["games_used"],
        "history_rows": int(len(history_df)) if history_df is not None else 0,
    }


def add_points_features(df):
    df = df.copy()
    history_df = load_game_history()

    for idx, row in df.iterrows():
        home_team = row.get("home_team", row.get("home_team_name", ""))
        away_team = row.get("away_team", row.get("away_team_name", ""))

        home_last_5 = calculate_team_points_features(home_team, history_df, last_n=5)
        away_last_5 = calculate_team_points_features(away_team, history_df, last_n=5)
        home_last_10 = calculate_team_points_features(home_team, history_df, last_n=10)
        away_last_10 = calculate_team_points_features(away_team, history_df, last_n=10)

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
        df.loc[idx, "ppg_diff_last_5"] = home_last_5["ppg"] - away_last_5["ppg"]
        df.loc[idx, "ppg_diff_last_10"] = home_last_10["ppg"] - away_last_10["ppg"]
        df.loc[idx, "points_allowed_diff_last_10"] = away_last_10["points_allowed"] - home_last_10["points_allowed"]
        df.loc[idx, "net_points_diff_last_10"] = home_last_10["net_points"] - away_last_10["net_points"]
        df.loc[idx, "avg_margin_diff_last_10"] = home_last_10["avg_margin"] - away_last_10["avg_margin"]

    return df

