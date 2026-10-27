"""Chronological walk-forward evaluation for the production NBA totals model.

This backtest evaluates score prediction only. It does not estimate betting ROI,
because historical sportsbook totals/odds are not available for each test game.
Every prediction receives only games strictly earlier than its target date.
"""

import os
import numpy as np
import pandas as pd

from points_engine import calculate_matchup_points
from totals_model import load_history, predict_game_total


OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "totals_full_model_backtest_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "totals_full_model_backtest_summary.csv")

MIN_PRIOR_GAMES = 100
MAX_GAMES_TO_TEST = 1000
TRAILING_MEAN_GAMES = 1000
SAFE_BOOKMAKER_PLACEHOLDER = 228.0


def prepare_backtest_history(history):
    """Normalize, validate, sort, and deduplicate a score-only history table."""
    if history is None or history.empty:
        raise ValueError("No historical NBA games were loaded.")

    df = history.copy()
    if "game_date" not in df.columns:
        date_col = next((c for c in ["date", "GAME_DATE", "gameDate"] if c in df.columns), None)
        if date_col is None:
            raise ValueError("History is missing a game_date/date column.")
        df = df.rename(columns={date_col: "game_date"})

    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=required)
    df = df[
        (df["home_score"] > 0)
        & (df["away_score"] > 0)
        & df["home_team"].ne("")
        & df["away_team"].ne("")
    ].copy()

    df["total_score"] = df["home_score"] + df["away_score"]
    df = df.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"],
        keep="last",
    )
    return df.reset_index(drop=True)


def history_strictly_before(history, target_date):
    """Return only completed games dated before the target calendar date."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    target = target.normalize()

    prior = history.loc[history["game_date"] < target].copy()
    if not prior.empty and not (prior["game_date"] < target).all():
        raise AssertionError("Chronological cutoff admitted a target-day/future game.")
    return prior


def regression_metrics(actual, predicted):
    """Return basic point-prediction metrics while ignoring invalid pairs."""
    actual = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted = pd.to_numeric(pd.Series(predicted), errors="coerce")
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid].astype(float)
    predicted = predicted.loc[valid].astype(float)

    if len(actual) == 0:
        return {
            "games": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "within_5_pct": np.nan,
            "within_10_pct": np.nan,
        }

    errors = predicted - actual
    absolute_errors = errors.abs()
    return {
        "games": int(len(actual)),
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "bias": float(errors.mean()),
        "within_5_pct": float((absolute_errors <= 5).mean() * 100),
        "within_10_pct": float((absolute_errors <= 10).mean() * 100),
    }


def run_backtest(max_games=MAX_GAMES_TO_TEST, min_prior_games=MIN_PRIOR_GAMES):
    history = prepare_backtest_history(load_history())

    if len(history) <= min_prior_games:
        raise ValueError(
            f"Need more than {min_prior_games} valid games; found {len(history)}."
        )

    test_history = history.tail(int(max_games)).copy()
    if test_history.empty:
        raise ValueError("The selected chronological test window is empty.")

    records = []
    skipped = 0
    tested_dates = sorted(test_history["game_date"].unique())

    # All games on a date share one strictly earlier history snapshot.
    for target_date in tested_dates:
        target_games = test_history.loc[test_history["game_date"] == target_date]
        prior = history_strictly_before(history, target_date)

        if len(prior) < min_prior_games:
            skipped += len(target_games)
            continue

        trailing_prior = prior.tail(TRAILING_MEAN_GAMES)
        trailing_mean_total = float(trailing_prior["total_score"].mean())

        for _, game in target_games.iterrows():
            home_team = str(game["home_team"]).strip()
            away_team = str(game["away_team"]).strip()
            actual_home = float(game["home_score"])
            actual_away = float(game["away_score"])
            actual_total = actual_home + actual_away

            try:
                # The production API needs a line to populate edge/recommendation.
                # This constant placeholder is not an actual market line and is
                # not used to calculate the score projection.
                full_prediction = predict_game_total(
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_total=SAFE_BOOKMAKER_PLACEHOLDER,
                    game_date=target_date,
                    history_df=prior,
                )

                points_prediction = calculate_matchup_points(
                    home_team=home_team,
                    away_team=away_team,
                    history_df=prior,
                    bookmaker_total=None,
                    game_date=target_date,
                )

                full_total = float(full_prediction["projected_total"])
                points_engine_total = float(points_prediction["projected_total"])

                records.append({
                    "game_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "actual_home_points": actual_home,
                    "actual_away_points": actual_away,
                    "actual_total": actual_total,
                    "full_model_projected_home": float(full_prediction["projected_home_points"]),
                    "full_model_projected_away": float(full_prediction["projected_away_points"]),
                    "full_model_projected_total": full_total,
                    "points_engine_projected_total": points_engine_total,
                    "trailing_mean_baseline": trailing_mean_total,
                    "full_model_error": full_total - actual_total,
                    "points_engine_error": points_engine_total - actual_total,
                    "trailing_mean_error": trailing_mean_total - actual_total,
                    "prior_history_rows": len(prior),
                    "prior_max_date": prior["game_date"].max().strftime("%Y-%m-%d"),
                })
            except Exception as exc:
                skipped += 1
                print(f"Skipped {target_date}: {away_team} at {home_team}: {exc}")

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No predictions completed. Check the score-history data and team names.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    comparisons = [
        ("Full totals model", "full_model_projected_total"),
        ("Points Engine baseline", "points_engine_projected_total"),
        ("Trailing 1000-game mean baseline", "trailing_mean_baseline"),
    ]

    summary_rows = []
    for model_name, prediction_column in comparisons:
        metrics = regression_metrics(results["actual_total"], results[prediction_column])
        summary_rows.append({
            "model": model_name,
            **metrics,
            "history_rows": len(history),
            "history_start": history["game_date"].min().date().isoformat(),
            "history_end": history["game_date"].max().date().isoformat(),
            "test_window_start": pd.to_datetime(results["game_date"]).min().date().isoformat(),
            "test_window_end": pd.to_datetime(results["game_date"]).max().date().isoformat(),
            "test_window_requested_games": len(test_history),
            "skipped_games": skipped,
            "test_results_file": RESULTS_PATH,
            "sportsbook_line_used": "none; constant placeholder used only by prediction API",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n===== CHRONOLOGICAL FULL TOTALS MODEL BACKTEST =====")
    print(summary[["model", "games", "mae", "rmse", "bias", "within_5_pct", "within_10_pct"]].to_string(index=False))
    print(f"\nHistory rows: {len(history)} | Completed test predictions: {len(results)} | Skipped: {skipped}")
    print(f"Detailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(
        "\nThis is a score-prediction evaluation, not a betting ROI backtest. "
        "It uses no future game data and does not validate AI probabilities."
    )

    return summary, results


if __name__ == "__main__":
    run_backtest()
