"""Integration backtest for the frozen V5 55:45 raw-total engine.

Run from the project root:
    python totals_backtest_v5.py

Requires:
- data/historical_nba_scores.csv
- outputs/raw_total_multiseason_predictions.csv
- points_engine_v5.py, points_engine_v3.py, points_engine_v4.py

Replays the same 6,140 historical target games from the multi-season test.
Checks V5 against the previously evaluated 55% V3 + 45% V4-90 formula and
runs a future-data invariance test on sampled historical dates.

No production file is modified.
"""
import os
import numpy as np
import pandas as pd

from points_engine import standardize_score_history, load_game_history
from points_engine_v5 import calculate_matchup_points_v5


DATA_PATH = "data/historical_nba_scores.csv"
REFERENCE_PATH = "outputs/raw_total_multiseason_predictions.csv"
OUTPUT_DIR = "outputs"
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "raw_total_v5_engine_predictions.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "raw_total_v5_engine_summary.csv")
MIN_PRIOR_GAMES = 10
LEAKAGE_SAMPLE_COUNT = 25


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
    return history


def load_reference():
    if not os.path.isfile(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Missing {REFERENCE_PATH}. Run totals_backtest_raw_multiseason.py first."
        )
    df = pd.read_csv(REFERENCE_PATH)
    required = [
        "game_date", "home_team", "away_team", "actual_total", "period",
        "season_start_year", "v2_raw_total", "v3_raw_total", "v4_90_total",
        "v4_180_total", "blend_v3_50_v4_90_50",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Reference predictions are missing columns: {missing}")
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in [
        "actual_total", "season_start_year", "v2_raw_total", "v3_raw_total",
        "v4_90_total", "v4_180_total", "blend_v3_50_v4_90_50",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).sort_values("game_date").reset_index(drop=True)
    return df


def metrics(actual, predicted):
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


def calculate_summary(results):
    models = [
        ("V2 raw", "v2_raw_total"),
        ("V3 candidate", "v3_raw_total"),
        ("V4 90-day", "v4_90_total"),
        ("V4 180-day", "v4_180_total"),
        ("Previously tested 50/50 blend", "blend_v3_50_v4_90_50"),
        ("V5 implemented 55/45", "v5_raw_total"),
    ]
    rows = []
    # The preceding multi-season run labels the holdout period
    # "later_validation". Accept that exact label (and the earlier
    # "later_seasons" spelling) rather than silently producing an empty group.
    available_periods = set(results["period"].astype(str).str.casefold())
    validation_label = next(
        (label for label in ("later_validation", "later_seasons", "validation")
         if label in available_periods),
        None,
    )
    for period, source_label in [
        ("development", "development"),
        ("later_validation", validation_label),
        ("all", None),
    ]:
        if period == "all":
            subset = results
        elif source_label is None:
            continue
        else:
            subset = results[
                results["period"].astype(str).str.casefold() == source_label
            ]
        if subset.empty:
            continue
        for name, column in models:
            row = {"scope": period, "model": name}
            row.update(metrics(subset["actual_total"], subset[column]))
            rows.append(row)
    for season, subset in results.groupby("season_start_year", sort=True):
        for name, column in models:
            row = {"scope": "season", "season_start_year": int(season), "model": name}
            row.update(metrics(subset["actual_total"], subset[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def run_leakage_invariance_checks(history, reference):
    """Same forecast with full history vs pre-truncated history must agree."""
    if len(reference) < 1:
        return pd.DataFrame()

    positions = np.linspace(0, len(reference) - 1, num=min(
        LEAKAGE_SAMPLE_COUNT, len(reference)
    ), dtype=int)
    checks = []
    for pos in positions:
        row = reference.iloc[int(pos)]
        date = pd.Timestamp(row["game_date"]).normalize()
        prior = history.loc[history["game_date"] < date].copy()

        full_result = calculate_matchup_points_v5(
            row["home_team"], row["away_team"], history, date,
            _prepared_cache={},
        )
        prior_result = calculate_matchup_points_v5(
            row["home_team"], row["away_team"], prior, date,
            _prepared_cache={},
        )
        difference = abs(
            float(full_result["raw_total"]) - float(prior_result["raw_total"])
        )
        checks.append({
            "game_date": date.date().isoformat(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "full_history_prediction": float(full_result["raw_total"]),
            "pre_truncated_history_prediction": float(prior_result["raw_total"]),
            "absolute_difference": difference,
            "passed": bool(difference <= 0.01),
        })
    return pd.DataFrame(checks)


def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = load_history()
    reference = load_reference()

    if len(reference) != 6140:
        print(
            f"NOTE: Reference contains {len(reference)} games rather than the "
            "expected 6,140. The run will use every valid reference row."
        )

    cache = {}
    records = []
    skipped = 0
    dates = sorted(reference["game_date"].unique())

    print("===== V5 ENGINE INTEGRATION BACKTEST =====")
    print(f"Historical rows: {len(history):,}")
    print(f"Reference target games: {len(reference):,}")
    print(f"Target dates: {reference['game_date'].min().date()} to "
          f"{reference['game_date'].max().date()}")
    print("V5 formula: 55% V3 + 45% V4-90.")
    print("Strict pre-target calendar-date cutoff is enforced by both engines.")

    for date_number, date in enumerate(dates, start=1):
        prior_count = int((history["game_date"] < date).sum())
        day = reference[reference["game_date"] == date]

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
                    0.55 * float(row.v3_raw_total)
                    + 0.45 * float(row.v4_90_total)
                )
                records.append({
                    "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "season_start_year": int(row.season_start_year),
                    "period": row.period,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual_total": float(row.actual_total),
                    "v2_raw_total": float(row.v2_raw_total),
                    "v3_raw_total": float(row.v3_raw_total),
                    "v4_90_total": float(row.v4_90_total),
                    "v4_180_total": float(row.v4_180_total),
                    "blend_v3_50_v4_90_50": float(row.blend_v3_50_v4_90_50),
                    "expected_blend_55_45": expected_blend,
                    "v5_raw_total": float(prediction["raw_total"]),
                    "v5_home_points": float(prediction["projected_home_points"]),
                    "v5_away_points": float(prediction["projected_away_points"]),
                    "formula_difference": float(prediction["raw_total"]) - expected_blend,
                    "v5_error": float(prediction["raw_total"]) - float(row.actual_total),
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
    summary["skipped_games"] = skipped
    summary.to_csv(SUMMARY_PATH, index=False)

    max_difference = float(results["formula_difference"].abs().max())
    mean_difference = float(results["formula_difference"].abs().mean())
    checks = run_leakage_invariance_checks(history, reference)
    leakage_passed = bool(not checks.empty and checks["passed"].all())

    print("\nMODEL PERFORMANCE")
    print(summary.to_string(index=False))
    print("\nV5 FORMULA REPRODUCTION")
    print(f"Maximum absolute deviation from stored 55:45 formula: {max_difference:.4f}")
    print(f"Mean absolute deviation from stored 55:45 formula: {mean_difference:.4f}")
    print("\nFUTURE-DATA INVARIANCE CHECK")
    print(checks.to_string(index=False))
    print(f"All sampled checks passed: {leakage_passed}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print(f"Skipped games: {skipped}")
    print("Research-only. Production files and live prediction paths are unchanged.")

    if skipped:
        raise RuntimeError(f"V5 integration backtest skipped {skipped} games.")
    if max_difference > 0.03:
        raise RuntimeError(
            "V5 engine differs from the tested 55:45 formula by more than 0.03 "
            "points for at least one game."
        )
    if not leakage_passed:
        raise RuntimeError("V5 failed the future-data invariance check.")


if __name__ == "__main__":
    run_backtest()
