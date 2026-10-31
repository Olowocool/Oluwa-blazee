"""Multi-season chronological raw-total evaluation and fixed V3/V4 blends.

Run from the project root:
    python totals_backtest_raw_multiseason.py

The script evaluates the five most recent NBA seasons with at least 500 games
in the local historical scores file. It uses only strictly prior games for
each prediction. It compares V2, V3, V4 90/180-day candidates, and three
predefined V3/V4-90 blends.

The first three selected seasons are the development period; the final two
seasons are the later validation period. Blend weight selection uses only
development seasons. This is research-only and does not touch production.
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
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "raw_total_multiseason_predictions.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "raw_total_multiseason_summary.csv")
MIN_GAMES_PER_SEASON = 500
MAX_SEASONS = 5
DEVELOPMENT_SEASONS = 3
MIN_PRIOR_GAMES = 10

BASE_MODELS = [
    ("V2 raw", "v2_raw_total"),
    ("V3 candidate", "v3_raw_total"),
    ("V4 90-day", "v4_90_total"),
    ("V4 180-day", "v4_180_total"),
]
BLEND_MODELS = [
    ("V5 blend V3 75% / V4-90 25%", "blend_v3_75_v4_90_25"),
    ("V5 blend V3 50% / V4-90 50%", "blend_v3_50_v4_90_50"),
    ("V5 blend V3 25% / V4-90 75%", "blend_v3_25_v4_90_75"),
]


def season_start_year(dates):
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1).astype(int)


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
        raise ValueError(f"Historical scores are missing columns: {missing}")

    history = history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=required)
    history = history[
        (history["home_score"] > 0) & (history["away_score"] > 0)
    ].copy()
    history["total_score"] = history["home_score"] + history["away_score"]
    history = history.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)
    history["season_start_year"] = season_start_year(history["game_date"])
    return history


def score_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    absolute = np.abs(errors)
    return {
        "games": int(len(errors)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias_pred_minus_actual": float(errors.mean()),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
    }


def make_summary(predictions):
    model_columns = BASE_MODELS + BLEND_MODELS
    rows = []

    for period in ["development", "later_seasons", "all"]:
        subset = predictions if period == "all" else predictions[
            predictions["period"] == period
        ]
        for model, column in model_columns:
            row = {"scope": period, "season_start_year": "all", "model": model}
            row.update(score_metrics(subset["actual_total"], subset[column]))
            rows.append(row)

    for season, group in predictions.groupby("season_start_year", sort=True):
        for model, column in model_columns:
            row = {"scope": "season", "season_start_year": int(season), "model": model}
            row.update(score_metrics(group["actual_total"], group[column]))
            rows.append(row)

    return pd.DataFrame(rows)


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_history()
    season_counts = history.groupby("season_start_year").size()
    eligible = [
        int(season) for season, count in season_counts.items()
        if int(count) >= MIN_GAMES_PER_SEASON
    ]
    selected_seasons = eligible[-MAX_SEASONS:]
    if len(selected_seasons) < 4:
        raise ValueError(
            f"Need at least four seasons with {MIN_GAMES_PER_SEASON}+ games; "
            f"found {selected_seasons}."
        )
    dev_seasons = selected_seasons[:DEVELOPMENT_SEASONS]
    later_seasons = selected_seasons[DEVELOPMENT_SEASONS:]
    if not later_seasons:
        raise ValueError("No later validation seasons were selected.")

    target = history[history["season_start_year"].isin(selected_seasons)].copy()
    target = target.sort_values("game_date").reset_index(drop=True)
    print("===== MULTI-SEASON RAW-TOTAL EVALUATION =====")
    print(f"History rows: {len(history):,}")
    print("Season games: " + ", ".join(
        f"{int(s)}={int(season_counts.loc[s])}" for s in selected_seasons
    ))
    print(f"Development seasons: {dev_seasons}")
    print(f"Later validation seasons: {later_seasons}")
    print(f"Target games: {len(target):,}")
    print("Predictions use only games strictly before the target calendar date.")
    print("Models: V2, V3, V4-90, V4-180 and fixed V3/V4-90 blends.")

    v3_cache = {}
    v4_cache = {}
    records = []
    skipped = 0
    dates = sorted(target["game_date"].unique())

    for date_number, date in enumerate(dates, start=1):
        prior = history.loc[history["game_date"] < date].copy()
        if len(prior) < MIN_PRIOR_GAMES:
            skipped += int((target["game_date"] == date).sum())
            continue

        day = target[target["game_date"] == date]
        for row in day.itertuples(index=False):
            try:
                v2 = calculate_matchup_points(
                    row.home_team, row.away_team, prior, game_date=date
                )
                v3 = calculate_matchup_points_v3(
                    row.home_team, row.away_team, prior,
                    game_date=date, _prepared_cache=v3_cache
                )
                v4_90 = calculate_matchup_points_v4(
                    row.home_team, row.away_team, history,
                    game_date=date, half_life_days=90, prior_games=8,
                    _prepared_cache=v4_cache
                )
                v4_180 = calculate_matchup_points_v4(
                    row.home_team, row.away_team, history,
                    game_date=date, half_life_days=180, prior_games=8,
                    _prepared_cache=v4_cache
                )

                p3 = float(v3["raw_total"])
                p90 = float(v4_90["raw_total"])
                p180 = float(v4_180["raw_total"])
                actual = float(row.home_score) + float(row.away_score)
                row_season = int(row.season_start_year)
                period = "development" if row_season in dev_seasons else "later_seasons"

                records.append({
                    "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "season_start_year": row_season,
                    "period": period,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_total": actual,
                    "v2_raw_total": float(v2["raw_total"]),
                    "v3_raw_total": p3,
                    "v4_90_total": p90,
                    "v4_180_total": p180,
                    "blend_v3_75_v4_90_25": 0.75 * p3 + 0.25 * p90,
                    "blend_v3_50_v4_90_50": 0.50 * p3 + 0.50 * p90,
                    "blend_v3_25_v4_90_75": 0.25 * p3 + 0.75 * p90,
                })
            except Exception as exc:
                skipped += 1
                print(
                    f"Skipped {date} {row.away_team} at {row.home_team}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if date_number % 20 == 0 or date_number == len(dates):
            print(
                f"Processed {date_number}/{len(dates)} dates; "
                f"predictions={len(records)}; skipped={skipped}"
            )

    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise ValueError("No multi-season predictions completed.")
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    summary = make_summary(predictions)
    summary["history_rows"] = len(history)
    summary["selected_seasons"] = ",".join(map(str, selected_seasons))
    summary["development_seasons"] = ",".join(map(str, dev_seasons))
    summary["later_seasons"] = ",".join(map(str, later_seasons))
    summary["skipped_games"] = skipped
    summary.to_csv(SUMMARY_PATH, index=False)

    development = summary[summary["scope"] == "development"].copy()
    selected = development.sort_values(["mae", "rmse"]).iloc[0]
    selected_model = selected["model"]
    later_result = summary[
        (summary["scope"] == "later_seasons")
        & (summary["model"] == selected_model)
    ].iloc[0]

    print("\nDEVELOPMENT RESULTS (used to select blend/model)")
    print(
        development[["model", "games", "mae", "rmse", "bias_pred_minus_actual",
                     "within_5_pct", "within_10_pct"]]
        .sort_values(["mae", "rmse"]).to_string(index=False)
    )
    print("\nLATER-SEASON VALIDATION RESULTS")
    print(
        summary[summary["scope"] == "later_seasons"][
            ["model", "games", "mae", "rmse", "bias_pred_minus_actual",
             "within_5_pct", "within_10_pct"]
        ].sort_values(["mae", "rmse"]).to_string(index=False)
    )
    print("\nSEASON-BY-SEASON RESULTS")
    print(
        summary[summary["scope"] == "season"][
            ["season_start_year", "model", "games", "mae", "rmse",
             "bias_pred_minus_actual", "within_5_pct", "within_10_pct"]
        ].to_string(index=False)
    )
    print(f"\nBest development model: {selected_model}")
    print(
        f"Later-season validation for selected model: "
        f"MAE={later_result['mae']:.4f}, RMSE={later_result['rmse']:.4f}, "
        f"bias={later_result['bias_pred_minus_actual']:.4f}"
    )
    print(f"Detailed predictions: {PREDICTIONS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Skipped games: {skipped}")
    print("Research-only. No production model or prediction path was changed.")


if __name__ == "__main__":
    run_backtest()
