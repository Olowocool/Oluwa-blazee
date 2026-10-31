"""Chronological V2-raw vs V3 candidate comparison.

Run from the project root:
    python totals_backtest_v3.py

Uses the existing historical score CSV; does not alter the production engine.
The comparison deliberately uses V2's raw_total (before H2H/rest), not its
projected_total, so the baseline is component-pure.
"""

import os
import numpy as np
import pandas as pd

from points_engine import load_game_history, calculate_matchup_points
from points_engine_v3 import calculate_matchup_points_v3


DATA_PATH = "data/historical_nba_scores.csv"
OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "points_engine_v3_comparison.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "points_engine_v3_summary.csv")
MAX_GAMES_TO_TEST = 2000
MIN_PRIOR_GAMES = 10


def load_history():
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = load_game_history()
    if df is None or df.empty:
        raise ValueError("No historical NBA score data was loaded.")
    date_col = next((c for c in ["game_date", "date", "GAME_DATE"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("Historical data has no game-date column.")
    df = df.copy()
    if date_col != "game_date":
        df = df.rename(columns={date_col: "game_date"})
    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Historical data is missing columns: {missing}")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df["home_score"] > 0) & (df["away_score"] > 0)]
    df["total_score"] = df["home_score"] + df["away_score"]
    df = df.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )
    return df.reset_index(drop=True)


def metrics(actual, predicted):
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    absolute = np.abs(errors)
    return {
        "games": int(len(errors)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(errors.mean()),
        "within_5_pct": float((absolute <= 5).mean() * 100),
        "within_10_pct": float((absolute <= 10).mean() * 100),
    }


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_history()
    # Select the dates covering the most recent N games (not N calendar dates).
    # The old implementation accidentally selected up to 2,000 dates, which
    # could mean testing tens of thousands of games.
    cutoff_date = history.tail(MAX_GAMES_TO_TEST)["game_date"].min()
    dates = sorted(history.loc[history["game_date"] >= cutoff_date, "game_date"].unique())
    records = []
    skipped = 0
    # One cleaned history + one ratings fit per target date, reused for every
    # matchup on that date. This avoids repeating the expensive fit per game.
    prepared_cache = {}

    for date in dates:
        prior = history[history["game_date"] < date].copy()
        if len(prior) < MIN_PRIOR_GAMES:
            skipped += int((history["game_date"] == date).sum())
            continue

        day = history[history["game_date"] == date]
        for row in day.itertuples(index=False):
            try:
                v2 = calculate_matchup_points(
                    row.home_team, row.away_team, prior, game_date=date,
                    _prepared_cache=prepared_cache
                )
                v3 = calculate_matchup_points_v3(
                    row.home_team, row.away_team, prior, game_date=date
                )
                actual = float(row.home_score) + float(row.away_score)
                records.append({
                    "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "season_start_year": v3["season_start_year"],
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_total": actual,
                    "v2_raw_total": float(v2["raw_total"]),
                    "v3_raw_total": float(v3["raw_total"]),
                    "v2_error": float(v2["raw_total"]) - actual,
                    "v3_error": float(v3["raw_total"]) - actual,
                    "v3_league_avg": v3["league_avg_team_points"],
                    "v3_season_games": v3["season_games_used"],
                })
            except Exception as exc:
                skipped += 1
                print(f"Skipped {date} {row.away_team} at {row.home_team}: {exc}")

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No V2/V3 comparison predictions completed.")
    results.to_csv(RESULTS_PATH, index=False)

    overall = []
    for model, col in [("V2 raw baseline", "v2_raw_total"), ("V3 candidate", "v3_raw_total")]:
        item = {"scope": "overall", "season_start_year": "all", "model": model}
        item.update(metrics(results["actual_total"], results[col]))
        overall.append(item)

    seasonal = []
    for season, group in results.groupby("season_start_year"):
        for model, col in [("V2 raw baseline", "v2_raw_total"), ("V3 candidate", "v3_raw_total")]:
            item = {"scope": "season", "season_start_year": int(season), "model": model}
            item.update(metrics(group["actual_total"], group[col]))
            seasonal.append(item)

    summary = pd.DataFrame(overall + seasonal)
    summary["history_rows"] = len(history)
    summary["history_start"] = history["game_date"].min().date().isoformat()
    summary["history_end"] = history["game_date"].max().date().isoformat()
    summary["skipped_games"] = skipped
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n===== POINTS ENGINE V2 RAW vs V3 CANDIDATE =====")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("\nV3 is a research candidate only. No production prediction path is changed.")


if __name__ == "__main__":
    run_backtest()
