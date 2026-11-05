"""Self-contained integration backtest for the research-only V5 raw-total engine.

Run from the project root:
    python totals_backtest_v5.py

Uses data/historical_nba_scores.csv directly. It selects the five most recent
NBA seasons with at least 500 games, uses the first three for development and
the next two for later validation, and checks:
- all eligible target games receive V5 predictions;
- V5 reproduces its 55:45 V3/V4-90 component blend within 0.03 points;
- full-history and pre-truncated-history predictions are identical on a
  25-date leakage-invariance sample.

No external/local reference prediction CSV is required. No production file is
modified.
"""
import os
import numpy as np
import pandas as pd

from points_engine import standardize_score_history, load_game_history
from points_engine_v5 import calculate_matchup_points_v5


DATA_PATH = "data/historical_nba_scores.csv"
OUTPUT_DIR = "outputs"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "raw_total_v5_engine_predictions.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "raw_total_v5_engine_summary.csv")
LEAKAGE_PATH = os.path.join(OUTPUT_DIR, "raw_total_v5_leakage_checks.csv")
MIN_GAMES_PER_SEASON = 500
MAX_SEASONS = 5
DEVELOPMENT_SEASONS = 3
MIN_PRIOR_GAMES = 10
LEAKAGE_SAMPLE_COUNT = 25
MAX_FORMULA_DIFFERENCE = 0.03


def season_start_year(dates):
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1).astype(int)


def load_history():
    if os.path.isfile(DATA_PATH):
        history = standardize_score_history(pd.read_csv(DATA_PATH))
    else:
        history = load_game_history()
    if history is None or history.empty:
        raise ValueError("No historical NBA score data was loaded.")

    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(f"Historical score file is missing columns: {missing}")

    history = history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    for column in ["home_score", "away_score"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
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


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    absolute = np.abs(errors)
    if len(errors) == 0:
        raise ValueError("Cannot calculate metrics for an empty prediction set.")
    return {
        "games": int(len(errors)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias_pred_minus_actual": float(errors.mean()),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
    }


def calculate_summary(results):
    models = [
        ("V3 component", "v3_raw_total"),
        ("V4 90-day component", "v4_90_raw_total"),
        ("V5 implemented 55/45", "v5_raw_total"),
    ]
    rows = []
    for scope in ["development", "later_validation", "all"]:
        subset = results if scope == "all" else results[results["period"] == scope]
        if subset.empty:
            continue
        for name, column in models:
            row = {"scope": scope, "model": name}
            row.update(metrics(subset["actual_total"], subset[column]))
            rows.append(row)

    for season, subset in results.groupby("season_start_year", sort=True):
        for name, column in models:
            row = {
                "scope": "season",
                "season_start_year": int(season),
                "model": name,
            }
            row.update(metrics(subset["actual_total"], subset[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def run_leakage_invariance_checks(history, targets):
    """Compare full input history with an explicitly pre-truncated history."""
    if targets.empty:
        return pd.DataFrame()

    positions = np.linspace(
        0,
        len(targets) - 1,
        num=min(LEAKAGE_SAMPLE_COUNT, len(targets)),
        dtype=int,
    )
    checks = []
    for position in positions:
        row = targets.iloc[int(position)]
        date = pd.Timestamp(row["game_date"]).normalize()
        prior = history.loc[history["game_date"] < date].copy()

        full_result = calculate_matchup_points_v5(
            row["home_team"], row["away_team"], history, date,
            _prepared_cache={},
        )
        truncated_result = calculate_matchup_points_v5(
            row["home_team"], row["away_team"], prior, date,
            _prepared_cache={},
        )
        difference = abs(
            float(full_result["raw_total"])
            - float(truncated_result["raw_total"])
        )
        checks.append({
            "game_date": date.date().isoformat(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "full_history_prediction": float(full_result["raw_total"]),
            "pre_truncated_history_prediction": float(truncated_result["raw_total"]),
            "absolute_difference": difference,
            "passed": bool(difference <= 0.01),
        })
    return pd.DataFrame(checks)


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_history()

    season_counts = history.groupby("season_start_year").size()
    eligible_seasons = [
        int(season)
        for season, count in season_counts.items()
        if int(count) >= MIN_GAMES_PER_SEASON
    ]
    selected_seasons = eligible_seasons[-MAX_SEASONS:]
    if len(selected_seasons) < MAX_SEASONS:
        raise ValueError(
            f"Need {MAX_SEASONS} seasons with {MIN_GAMES_PER_SEASON}+ games; "
            f"found {selected_seasons}."
        )

    development_seasons = selected_seasons[:DEVELOPMENT_SEASONS]
    validation_seasons = selected_seasons[DEVELOPMENT_SEASONS:]
    if not validation_seasons:
        raise ValueError("No later validation seasons were selected.")

    targets = history[
        history["season_start_year"].isin(selected_seasons)
    ].copy()
    targets = targets.sort_values("game_date").reset_index(drop=True)
    if targets.empty:
        raise ValueError("No target games were selected.")

    print("===== V5 ENGINE INTEGRATION BACKTEST =====")
    print(f"Historical rows: {len(history):,}")
    print("Season game counts: " + ", ".join(
        f"{int(season)}={int(season_counts.loc[season])}"
        for season in selected_seasons
    ))
    print(f"Development seasons: {development_seasons}")
    print(f"Later validation seasons: {validation_seasons}")
    print(f"Reference target games: {len(targets):,}")
    print(
        f"Target dates: {targets['game_date'].min().date()} to "
        f"{targets['game_date'].max().date()}"
    )
    print("V5 formula: 55% V3 + 45% V4-90.")
    print("Predictions use only games strictly before each target calendar date.")

    cache = {}
    records = []
    skipped = 0
    dates = sorted(targets["game_date"].drop_duplicates())

    for date_number, date in enumerate(dates, start=1):
        prior_count = int((history["game_date"] < date).sum())
        day = targets[targets["game_date"] == date]
        if prior_count < MIN_PRIOR_GAMES:
            skipped += len(day)
            continue

        for row in day.itertuples(index=False):
            try:
                prediction = calculate_matchup_points_v5(
                    row.home_team,
                    row.away_team,
                    history,
                    game_date=date,
                    _prepared_cache=cache,
                )
                expected_blend = (
                    float(prediction["v3_weight"])
                    * float(prediction["v3_raw_total"])
                    + float(prediction["v4_weight"])
                    * float(prediction["v4_90_raw_total"])
                )
                records.append({
                    "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "season_start_year": int(row.season_start_year),
                    "period": (
                        "development"
                        if int(row.season_start_year) in development_seasons
                        else "later_validation"
                    ),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_total": float(row.home_score) + float(row.away_score),
                    "v3_raw_total": float(prediction["v3_raw_total"]),
                    "v4_90_raw_total": float(prediction["v4_90_raw_total"]),
                    "v5_raw_total": float(prediction["raw_total"]),
                    "v5_home_points": float(prediction["projected_home_points"]),
                    "v5_away_points": float(prediction["projected_away_points"]),
                    "expected_blend_55_45": expected_blend,
                    "formula_difference": float(prediction["raw_total"]) - expected_blend,
                    "v5_error": float(prediction["raw_total"])
                    - (float(row.home_score) + float(row.away_score)),
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

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("V5 did not produce any raw-total predictions.")
    results.to_csv(PREDICTIONS_PATH, index=False)

    summary = calculate_summary(results)
    summary["history_rows"] = len(history)
    summary["selected_seasons"] = ",".join(map(str, selected_seasons))
    summary["development_seasons"] = ",".join(map(str, development_seasons))
    summary["validation_seasons"] = ",".join(map(str, validation_seasons))
    summary["skipped_games"] = skipped
    summary.to_csv(SUMMARY_PATH, index=False)

    formula_differences = results["formula_difference"].abs()
    max_difference = float(formula_differences.max())
    mean_difference = float(formula_differences.mean())

    checks = run_leakage_invariance_checks(history, targets)
    checks.to_csv(LEAKAGE_PATH, index=False)
    leakage_passed = bool(not checks.empty and checks["passed"].all())
    leakage_max_difference = (
        float(checks["absolute_difference"].max()) if not checks.empty else float("inf")
    )

    print("\nMODEL PERFORMANCE")
    print(summary.to_string(index=False))
    print("\nV5 FORMULA REPRODUCTION")
    print(f"Maximum absolute deviation from 55:45 formula: {max_difference:.4f}")
    print(f"Mean absolute deviation from 55:45 formula: {mean_difference:.4f}")
    print("\nFUTURE-DATA INVARIANCE CHECK")
    print(checks.to_string(index=False))
    print(f"All sampled checks passed: {leakage_passed}")
    print(f"Maximum sampled prediction difference: {leakage_max_difference:.4f}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print(f"Leakage checks saved: {LEAKAGE_PATH}")
    print(f"Skipped games: {skipped}")
    print("Research-only. Production files and live prediction paths are unchanged.")

    if skipped:
        raise RuntimeError(f"V5 integration backtest skipped {skipped} games.")
    if max_difference > MAX_FORMULA_DIFFERENCE:
        raise RuntimeError(
            f"V5 differs from its 55:45 formula by more than "
            f"{MAX_FORMULA_DIFFERENCE:.2f} points."
        )
    if not leakage_passed:
        raise RuntimeError("V5 failed the future-data invariance check.")


if __name__ == "__main__":
    run_backtest()
