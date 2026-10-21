
import os
import pandas as pd
import numpy as np

from points_engine import calculate_matchup_points, load_game_history


DATA_PATH = "data/historical_nba_scores.csv"
OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "totals_backtest_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "totals_backtest_summary.csv")

MIN_PRIOR_GAMES = 10
MAX_GAMES_TO_TEST = 2000


def load_backtest_history():
    """Load real completed games and normalize required fields."""
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = load_game_history()

    if df is None or df.empty:
        raise ValueError("No historical NBA games were loaded.")

    date_col = next(
        (c for c in ["game_date", "date", "GAME_DATE"] if c in df.columns),
        None,
    )
    if date_col is None:
        raise ValueError("History has no recognizable game-date column.")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce")

    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=required)
    df = df[
        (df["home_score"] > 0)
        & (df["away_score"] > 0)
        & (df["home_team"].astype(str).str.strip() != "")
        & (df["away_team"].astype(str).str.strip() != "")
    ]

    df["total_score"] = df["home_score"] + df["away_score"]
    df = df.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"],
        keep="last",
    )
    return df.reset_index(drop=True)


def predict_with_past_only(prior_games, home_team, away_team):
    """
    Use the existing points engine, passing only games before the target date.
    The full totals_model is deliberately not called because its other
    adjustments have not yet been verified for historical, date-aware inputs.
    """
    return calculate_matchup_points(
        home_team=home_team,
        away_team=away_team,
        history_df=prior_games,
        bookmaker_total=None,
    )


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_backtest_history()

    if len(history) <= MIN_PRIOR_GAMES:
        raise ValueError("Not enough historical games to run the backtest.")

    # Test the most recent completed games; all prior history remains available
    # to each prediction, but no game on or after the prediction date is used.
    unique_dates = sorted(history["game_date"].unique())
    test_dates = set(unique_dates[-MAX_GAMES_TO_TEST:])

    records = []
    tested = 0
    skipped = 0

    for date in sorted(test_dates):
        day_games = history[history["game_date"] == date]
        prior_games = history[history["game_date"] < date]

        if len(prior_games) < MIN_PRIOR_GAMES:
            skipped += len(day_games)
            continue

        for _, game in day_games.iterrows():
            home_team = str(game["home_team"]).strip()
            away_team = str(game["away_team"]).strip()
            actual_home = float(game["home_score"])
            actual_away = float(game["away_score"])
            actual_total = actual_home + actual_away

            try:
                prediction = predict_with_past_only(
                    prior_games, home_team, away_team
                )
                projected_total = float(prediction["projected_total"])
                projected_home = float(prediction["projected_home_points"])
                projected_away = float(prediction["projected_away_points"])
            except Exception as exc:
                skipped += 1
                print(
                    f"Skipped {date}: {away_team} at {home_team}: {exc}"
                )
                continue

            records.append({
                "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "actual_home_points": actual_home,
                "actual_away_points": actual_away,
                "actual_total": actual_total,
                "projected_home_points": projected_home,
                "projected_away_points": projected_away,
                "projected_total": projected_total,
                "total_error": projected_total - actual_total,
                "absolute_error": abs(projected_total - actual_total),
                "prior_history_rows": len(prior_games),
            })
            tested += 1

        if tested and tested % 500 == 0:
            print(f"Predictions completed: {tested}")

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No predictions completed. Check team names and history.")

    results.to_csv(RESULTS_PATH, index=False)

    errors = results["total_error"]
    summary = pd.DataFrame([{
        "history_rows": len(history),
        "history_start": history["game_date"].min().date(),
        "history_end": history["game_date"].max().date(),
        "test_games": len(results),
        "skipped_games": skipped,
        "mean_absolute_error": results["absolute_error"].mean(),
        "root_mean_squared_error": np.sqrt((errors ** 2).mean()),
        "mean_error_bias": errors.mean(),
        "within_5_points_pct": (results["absolute_error"] <= 5).mean(),
        "within_10_points_pct": (results["absolute_error"] <= 10).mean(),
    }])
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n===== NBA TOTALS BACKTEST SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"\nDetailed predictions: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(
        "\nNote: This evaluates score prediction only. "
        "It does not establish betting ROI or use sportsbook totals."
    )


if __name__ == "__main__":
    run_backtest()
