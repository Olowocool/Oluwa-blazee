"""Chronological score backtests for the baseline and experimental totals model.

The latest window is kept as the final holdout. Component ablation runs on the
1,000 games immediately before that window and must not be used as a final
unbiased estimate after model tuning. Historical predictions use only games
strictly earlier than each target date.

This evaluates point prediction only; it does not estimate betting ROI.
"""

import os
import numpy as np
import pandas as pd

from points_engine import calculate_matchup_points
from team_name_utils import normalize_history_team_columns
from totals_model import load_history, predict_game_total


OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "totals_full_model_backtest_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "totals_full_model_backtest_summary.csv")
ABLATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "totals_development_ablation_results.csv")
ABLATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "totals_development_ablation_summary.csv")

MIN_PRIOR_GAMES = 100
MAX_GAMES_TO_TEST = 1000
DEV_ABLATION_GAMES = 1000
TRAILING_MEAN_GAMES = 1000
SAFE_BOOKMAKER_PLACEHOLDER = 228.0

PREDICTION_COLUMNS = {
    "Points Engine baseline": "points_engine_projected_total",
    "Experimental full totals model": "full_model_projected_total",
    "Full model minus pace": "full_without_pace_total",
    "Full model minus offense": "full_without_offense_total",
    "Full model minus defense": "full_without_defense_total",
    "Full model minus venue": "full_without_venue_total",
    "Baseline plus pace only": "baseline_plus_pace_total",
    "Baseline plus offense only": "baseline_plus_offense_total",
    "Baseline plus defense only": "baseline_plus_defense_total",
    "Baseline plus venue only": "baseline_plus_venue_total",
}


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

    # Recompute totals from final scores; do not trust stale cached totals.
    df["total_score"] = df["home_score"] + df["away_score"]
    df = df.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"],
        keep="last",
    )
    df = normalize_history_team_columns(df.reset_index(drop=True))
    return df


def history_strictly_before(history, target_date):
    """Return only completed games dated before the target calendar date."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    target = target.normalize()

    prior = history.loc[history["game_date"] < target].copy()
    if not prior.empty and not (prior["game_date"] < target).all():
        raise AssertionError("Chronological cutoff admitted a target-day/future game.")
    prior.attrs["_strict_history_cutoff_date"] = target.date().isoformat()
    prior.attrs["_team_names_normalized"] = True
    return prior


def regression_metrics(actual, predicted):
    """Return point-prediction metrics while ignoring invalid pairs."""
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


def build_ablation_predictions(baseline_total, full_prediction):
    """Construct one-factor removals and one-factor additions to the baseline."""
    baseline = float(baseline_total)
    full = float(full_prediction["projected_total"])
    components = {
        "pace": float(full_prediction["pace_adjustment"]),
        "offense": float(full_prediction["offensive_adjustment"]),
        "defense": float(full_prediction["defensive_adjustment"]),
        "venue": float(full_prediction["home_away_adjustment"]),
    }

    predictions = {
        "points_engine_projected_total": baseline,
        "full_model_projected_total": full,
        "full_without_pace_total": full - components["pace"],
        "full_without_offense_total": full - components["offense"],
        "full_without_defense_total": full - components["defense"],
        "full_without_venue_total": full - components["venue"],
        "baseline_plus_pace_total": baseline + components["pace"],
        "baseline_plus_offense_total": baseline + components["offense"],
        "baseline_plus_defense_total": baseline + components["defense"],
        "baseline_plus_venue_total": baseline + components["venue"],
    }
    return predictions, components


def summarize_predictions(results, prediction_columns, test_window_start, test_window_end,
                          requested_games, skipped_games, history):
    summary_rows = []
    for model_name, prediction_column in prediction_columns.items():
        metrics = regression_metrics(results["actual_total"], results[prediction_column])
        summary_rows.append({
            "model": model_name,
            **metrics,
            "mean_delta_vs_points_engine": float(
                (results[prediction_column] - results["points_engine_projected_total"]).mean()
            ),
            "history_rows": len(history),
            "history_start": history["game_date"].min().date().isoformat(),
            "history_end": history["game_date"].max().date().isoformat(),
            "test_window_start": pd.to_datetime(test_window_start).date().isoformat(),
            "test_window_end": pd.to_datetime(test_window_end).date().isoformat(),
            "test_window_requested_games": int(requested_games),
            "skipped_games": int(skipped_games),
            "sportsbook_line_used": "none; constant placeholder used only by prediction API",
        })
    return pd.DataFrame(summary_rows)


def _evaluate_full_model_window(history, test_history, output_path, min_prior_games,
                                include_ablation=False):
    """Evaluate a date-bounded window; all games on a date share prior-only history."""
    records = []
    skipped = 0
    tested_dates = sorted(test_history["game_date"].unique())

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
                full_prediction = predict_game_total(
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_total=SAFE_BOOKMAKER_PLACEHOLDER,
                    game_date=target_date,
                    history_df=prior,
                    apply_experimental_adjustments=True,
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
                record = {
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
                }

                if include_ablation:
                    predictions, components = build_ablation_predictions(
                        points_engine_total, full_prediction
                    )
                    record.update(predictions)
                    record.update({
                        "pace_adjustment": components["pace"],
                        "offensive_adjustment": components["offense"],
                        "defensive_adjustment": components["defense"],
                        "home_away_adjustment": components["venue"],
                        "injury_adjustment": float(full_prediction.get("injury_adjustment", 0.0)),
                    })

                records.append(record)
            except Exception as exc:
                skipped += 1
                print(f"Skipped {target_date}: {away_team} at {home_team}: {exc}")

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError(f"No predictions completed for {output_path}. Check history and team names.")
    results.to_csv(output_path, index=False)
    return results, skipped


def run_backtest(max_games=MAX_GAMES_TO_TEST, min_prior_games=MIN_PRIOR_GAMES):
    """Evaluate the final, latest holdout window. Do not tune on these results."""
    history = prepare_backtest_history(load_history())
    if len(history) <= min_prior_games:
        raise ValueError(f"Need more than {min_prior_games} valid games; found {len(history)}.")

    test_history = history.tail(int(max_games)).copy()
    if test_history.empty:
        raise ValueError("The selected chronological test window is empty.")

    results, skipped = _evaluate_full_model_window(
        history, test_history, RESULTS_PATH, min_prior_games, include_ablation=False
    )
    comparisons = {
        "Full totals model": "full_model_projected_total",
        "Points Engine baseline": "points_engine_projected_total",
        "Trailing 1000-game mean baseline": "trailing_mean_baseline",
    }
    summary = summarize_predictions(
        results,
        comparisons,
        pd.to_datetime(results["game_date"]).min(),
        pd.to_datetime(results["game_date"]).max(),
        len(test_history),
        skipped,
        history,
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n===== FINAL HOLDOUT: CHRONOLOGICAL FULL TOTALS MODEL BACKTEST =====")
    print(summary[["model", "games", "mae", "rmse", "bias", "within_5_pct", "within_10_pct"]].to_string(index=False))
    print(f"History rows: {len(history)} | Completed predictions: {len(results)} | Skipped: {skipped}")
    print("This final holdout must not be used to choose or tune model adjustments.")
    return summary, results


def run_development_ablation(max_games=DEV_ABLATION_GAMES,
                             min_prior_games=MIN_PRIOR_GAMES):
    """Compare model components on the 1,000 games immediately before the holdout."""
    history = prepare_backtest_history(load_history())
    holdout_rows = history.tail(MAX_GAMES_TO_TEST)
    if holdout_rows.empty:
        raise ValueError("Cannot define the reserved final holdout window.")
    holdout_start = holdout_rows["game_date"].min()

    # Reserve all dates in the final holdout; the development window ends strictly before it.
    development_pool = history.loc[history["game_date"] < holdout_start].copy()
    test_history = development_pool.tail(int(max_games)).copy()
    if test_history.empty:
        raise ValueError("No development games exist before the reserved final holdout.")
    if test_history["game_date"].max() >= holdout_start:
        raise AssertionError("Development ablation overlaps the final holdout dates.")

    results, skipped = _evaluate_full_model_window(
        history, test_history, ABLATION_RESULTS_PATH, min_prior_games, include_ablation=True
    )
    summary = summarize_predictions(
        results,
        PREDICTION_COLUMNS,
        pd.to_datetime(results["game_date"]).min(),
        pd.to_datetime(results["game_date"]).max(),
        len(test_history),
        skipped,
        history,
    )
    summary["reserved_holdout_start"] = holdout_start.date().isoformat()
    summary.to_csv(ABLATION_SUMMARY_PATH, index=False)

    print("\n===== DEVELOPMENT WINDOW: COMPONENT ABLATION (NOT FINAL VALIDATION) =====")
    print(summary[["model", "games", "mae", "rmse", "bias", "mean_delta_vs_points_engine",
                   "within_5_pct", "within_10_pct"]].to_string(index=False))
    print(
        f"Development window: {summary.iloc[0]['test_window_start']} through "
        f"{summary.iloc[0]['test_window_end']} | Reserved holdout starts {holdout_start.date()}"
    )
    print("Use this development window to diagnose components only; rerun on untouched future data after any changes.")
    return summary, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_backtest()
    run_development_ablation()
