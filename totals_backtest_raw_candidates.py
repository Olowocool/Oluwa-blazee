"""Chronological raw-total model selection: V2 vs V3 vs V4 candidates.

Run from the project root:
    python totals_backtest_raw_candidates.py

Uses score history only. V4 candidates use recency-weighted, opponent-adjusted
team scoring ratings and a learned home-court effect. No pace, H2H, rest,
odds, or market lines enter these raw-total predictions.

The first 70% of test dates is the development period for selecting settings.
The final 30% is an untouched chronological holdout. Every game prediction
uses only games strictly before that game's calendar date.
"""
import os
import numpy as np
import pandas as pd

from points_engine import (
    load_game_history,
    standardize_score_history,
    calculate_matchup_points,
)
from points_engine_v3 import calculate_matchup_points_v3
from points_engine_v4 import calculate_matchup_points_v4


DATA_PATH = "data/historical_nba_scores.csv"
OUTPUT_DIR = "outputs"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "raw_total_candidate_predictions.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "raw_total_model_selection.csv")
MAX_GAMES_TO_TEST = 2000
MIN_PRIOR_GAMES = 10
DEVELOPMENT_DATE_FRACTION = 0.70

V4_CONFIGS = [
    {"name": "V4_HL90_P8", "half_life_days": 90, "prior_games": 8},
    {"name": "V4_HL180_P8", "half_life_days": 180, "prior_games": 8},
    {"name": "V4_HL365_P8", "half_life_days": 365, "prior_games": 8},
    {"name": "V4_HL730_P8", "half_life_days": 730, "prior_games": 8},
    {"name": "V4_HL365_P20", "half_life_days": 365, "prior_games": 20},
]


def load_history():
    if os.path.isfile(DATA_PATH):
        source = pd.read_csv(DATA_PATH)
        history = standardize_score_history(source)
    else:
        history = load_game_history()
    if history is None or history.empty:
        raise ValueError("No historical NBA score data was loaded.")

    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [col for col in required if col not in history.columns]
    if missing:
        raise ValueError(f"Historical score data is missing columns: {missing}")

    history = history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=required)
    history = history[
        (history["home_score"] > 0) & (history["away_score"] > 0)
    ]
    history["total_score"] = history["home_score"] + history["away_score"]
    history = history.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )
    return history.reset_index(drop=True)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    absolute = np.abs(errors)
    return {
        "games": int(len(errors)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "bias_pred_minus_actual": float(np.mean(errors)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
    }


def summarize_predictions(results):
    model_columns = [
        ("V2 raw", "v2_raw_total"),
        ("V3 current candidate", "v3_raw_total"),
    ] + [(cfg["name"], cfg["name"]) for cfg in V4_CONFIGS]

    rows = []
    for period in ["development", "holdout", "all"]:
        subset = results if period == "all" else results[results["period"] == period]
        for model_name, column in model_columns:
            item = {"period": period, "model": model_name}
            item.update(metrics(subset["actual_total"], subset[column]))
            rows.append(item)
    return pd.DataFrame(rows)


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_history()
    if len(history) < MAX_GAMES_TO_TEST + MIN_PRIOR_GAMES:
        raise ValueError(
            f"Only {len(history)} valid rows found; insufficient for "
            f"{MAX_GAMES_TO_TEST} test games plus prior history."
        )

    cutoff_date = history.tail(MAX_GAMES_TO_TEST)["game_date"].min()
    test_dates = sorted(
        history.loc[history["game_date"] >= cutoff_date, "game_date"].unique()
    )
    if len(test_dates) < 10:
        raise ValueError("Too few unique test dates for chronological validation.")

    split_index = int(len(test_dates) * DEVELOPMENT_DATE_FRACTION)
    split_index = min(max(split_index, 1), len(test_dates) - 1)
    holdout_start = pd.Timestamp(test_dates[split_index])
    print("===== RAW-TOTAL MODEL SELECTION =====")
    print(f"History rows: {len(history):,}")
    print(f"Test window: {cutoff_date.date()} to {history['game_date'].max().date()}")
    print(f"Holdout starts: {holdout_start.date()}")
    print("V4 candidates: " + ", ".join(cfg["name"] for cfg in V4_CONFIGS))
    print("No pace/H2H/rest/market adjustments are included.")

    v3_cache = {}
    v4_cache = {}
    records = []
    skipped = 0

    for date_number, date in enumerate(test_dates, start=1):
        prior = history.loc[history["game_date"] < date].copy()
        if len(prior) < MIN_PRIOR_GAMES:
            skipped += int((history["game_date"] == date).sum())
            continue

        day = history.loc[history["game_date"] == date]
        for row in day.itertuples(index=False):
            try:
                v2 = calculate_matchup_points(
                    row.home_team, row.away_team, prior, game_date=date
                )
                v3 = calculate_matchup_points_v3(
                    row.home_team, row.away_team, prior,
                    game_date=date, _prepared_cache=v3_cache
                )
                item = {
                    "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "season_start_year": int(
                        pd.Timestamp(date).year
                        if pd.Timestamp(date).month >= 10
                        else pd.Timestamp(date).year - 1
                    ),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_total": float(row.home_score) + float(row.away_score),
                    "v2_raw_total": float(v2["raw_total"]),
                    "v3_raw_total": float(v3["raw_total"]),
                    "period": "development" if pd.Timestamp(date) < holdout_start else "holdout",
                }
                for cfg in V4_CONFIGS:
                    prediction = calculate_matchup_points_v4(
                        row.home_team,
                        row.away_team,
                        history,
                        game_date=date,
                        half_life_days=cfg["half_life_days"],
                        prior_games=cfg["prior_games"],
                        _prepared_cache=v4_cache,
                    )
                    item[cfg["name"]] = float(prediction["raw_total"])
                records.append(item)
            except Exception as exc:
                skipped += 1
                print(
                    f"Skipped {date} {row.away_team} at {row.home_team}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if date_number % 20 == 0 or date_number == len(test_dates):
            print(
                f"Processed {date_number}/{len(test_dates)} dates; "
                f"predictions={len(records)}; skipped={skipped}"
            )

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No candidate model predictions completed.")
    results.to_csv(PREDICTIONS_PATH, index=False)

    summary = summarize_predictions(results)
    summary["history_rows"] = len(history)
    summary["test_start"] = results["game_date"].min()
    summary["test_end"] = results["game_date"].max()
    summary["holdout_start"] = holdout_start.date().isoformat()
    summary["skipped_games"] = skipped
    summary.to_csv(SUMMARY_PATH, index=False)

    development = summary[summary["period"] == "development"].copy()
    # Model selection uses development dates only. Holdout metrics are reported
    # separately and are not used to choose the winning configuration.
    winner = development.sort_values(["mae", "rmse"]).iloc[0]
    winner_name = winner["model"]
    holdout_winner = summary[
        (summary["period"] == "holdout") & (summary["model"] == winner_name)
    ].iloc[0]

    print("\nDEVELOPMENT PERIOD RESULTS (used for model selection)")
    print(
        summary[summary["period"] == "development"][
            ["model", "games", "mae", "rmse", "bias_pred_minus_actual",
             "within_5_pct", "within_10_pct"]
        ].sort_values(["mae", "rmse"]).to_string(index=False)
    )
    print("\nUNTOUCHED CHRONOLOGICAL HOLDOUT RESULTS")
    print(
        summary[summary["period"] == "holdout"][
            ["model", "games", "mae", "rmse", "bias_pred_minus_actual",
             "within_5_pct", "within_10_pct"]
        ].sort_values(["mae", "rmse"]).to_string(index=False)
    )
    print(f"\nSelected on development only: {winner_name}")
    print(
        "Its holdout results: "
        f"MAE={holdout_winner['mae']:.4f}, "
        f"RMSE={holdout_winner['rmse']:.4f}, "
        f"bias={holdout_winner['bias_pred_minus_actual']:.4f}"
    )
    print(f"Detailed predictions: {PREDICTIONS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Skipped games: {skipped}")
    print("Research-only comparison. No production prediction path was changed.")


if __name__ == "__main__":
    run_backtest()
