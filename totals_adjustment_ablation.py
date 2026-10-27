"""Exploratory chronological ablation of totals-model adjustments.

This evaluates adjustment variants only on the development window immediately
before the most recent 1,000-game holdout window. It intentionally does NOT
score candidate variants on the recent holdout, to avoid tuning on that window.

This is score-prediction research, not a betting ROI evaluation. No candidate
selected here should be called validated until independently tested after it is
frozen.
"""
import os

import numpy as np
import pandas as pd

from totals_full_model_backtest import (
    MIN_PRIOR_GAMES,
    SAFE_BOOKMAKER_PLACEHOLDER,
    history_strictly_before,
    prepare_backtest_history,
    regression_metrics,
)
from totals_model import load_history, predict_game_total


OUTPUT_DIR = "outputs"
DETAILS_PATH = os.path.join(OUTPUT_DIR, "totals_adjustment_ablation_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "totals_adjustment_ablation_summary.csv")
HOLDOUT_GAMES = 1000
DEVELOPMENT_GAMES = 1000

COMPONENT_COLUMNS = {
    "pace": "pace_adjustment",
    "offense": "offensive_adjustment",
    "defense": "defensive_adjustment",
    "venue": "home_away_adjustment",
    "injury": "injury_adjustment",
}

# All component variants are authored in advance. Selection is based on the
# older development window only, never on the recent holdout window.
VARIANTS = {
    "Points Engine baseline": {},
    "Pace only": {"pace": 1.0},
    "Offense only": {"offense": 1.0},
    "Defense only": {"defense": 1.0},
    "Venue split only": {"venue": 1.0},
    "Pace + offense": {"pace": 1.0, "offense": 1.0},
    "Pace + defense": {"pace": 1.0, "defense": 1.0},
    "Pace + venue": {"pace": 1.0, "venue": 1.0},
    "Offense + defense + venue (no pace)": {
        "offense": 1.0, "defense": 1.0, "venue": 1.0
    },
    "Full model current": {
        "pace": 1.0, "offense": 1.0, "defense": 1.0, "venue": 1.0,
        "injury": 1.0,
    },
    "Full minus pace": {
        "offense": 1.0, "defense": 1.0, "venue": 1.0, "injury": 1.0
    },
    "Full minus offense": {
        "pace": 1.0, "defense": 1.0, "venue": 1.0, "injury": 1.0
    },
    "Full minus defense": {
        "pace": 1.0, "offense": 1.0, "venue": 1.0, "injury": 1.0
    },
    "Full minus venue": {
        "pace": 1.0, "offense": 1.0, "defense": 1.0, "injury": 1.0
    },
    "Full adjustments at 25%": {
        "pace": 0.25, "offense": 0.25, "defense": 0.25,
        "venue": 0.25, "injury": 0.25,
    },
    "Full adjustments at 50%": {
        "pace": 0.50, "offense": 0.50, "defense": 0.50,
        "venue": 0.50, "injury": 0.50,
    },
    "Full adjustments at 75%": {
        "pace": 0.75, "offense": 0.75, "defense": 0.75,
        "venue": 0.75, "injury": 0.75,
    },
}


def _candidate_total(baseline, components, weights):
    return round(
        float(baseline)
        + sum(float(components[name]) * weight for name, weight in weights.items()),
        2,
    )


def run_ablation(
    holdout_games=HOLDOUT_GAMES,
    development_games=DEVELOPMENT_GAMES,
    min_prior_games=MIN_PRIOR_GAMES,
):
    history = prepare_backtest_history(load_history())
    if len(history) <= min_prior_games + holdout_games:
        raise ValueError(
            f"Need more than {min_prior_games + holdout_games} valid games "
            f"to keep a prior-only development window and holdout; found {len(history)}."
        )

    # Reserve the recent window exclusively as holdout. This script does not
    # calculate candidate metrics for it.
    holdout_tail = history.tail(int(holdout_games))
    holdout_start = holdout_tail["game_date"].min().normalize()
    pre_holdout = history.loc[history["game_date"] < holdout_start].copy()
    if pre_holdout.empty:
        raise ValueError("No historical games precede the reserved holdout window.")

    dev_tail = pre_holdout.tail(int(development_games))
    dev_start = dev_tail["game_date"].min().normalize()
    development = pre_holdout.loc[pre_holdout["game_date"] >= dev_start].copy()
    if development.empty:
        raise ValueError("The chronological development window is empty.")

    dev_dates = sorted(development["game_date"].unique())
    records = []
    skipped = 0
    for target_date in dev_dates:
        target_games = development.loc[development["game_date"] == target_date]
        prior = history_strictly_before(history, target_date)
        if len(prior) < min_prior_games:
            skipped += len(target_games)
            continue
        if not (prior["game_date"] < pd.Timestamp(target_date)).all():
            raise AssertionError("Development backtest leaked target-day/future data.")

        prior_max_date = prior["game_date"].max().normalize()
        for _, game in target_games.iterrows():
            home_team = str(game["home_team"]).strip()
            away_team = str(game["away_team"]).strip()
            actual_total = float(game["home_score"]) + float(game["away_score"])
            try:
                prediction = predict_game_total(
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker_total=SAFE_BOOKMAKER_PLACEHOLDER,
                    game_date=target_date,
                    history_df=prior,
                )
                baseline = float(prediction["points_engine_total"])
                components = {
                    name: float(prediction[column])
                    for name, column in COMPONENT_COLUMNS.items()
                }

                row = {
                    "game_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "actual_total": actual_total,
                    "points_engine_baseline": baseline,
                    "prior_history_rows": int(len(prior)),
                    "prior_max_date": prior_max_date.strftime("%Y-%m-%d"),
                    **components,
                }
                for variant_name, weights in VARIANTS.items():
                    row[variant_name] = _candidate_total(baseline, components, weights)

                if row["prior_max_date"] >= row["game_date"]:
                    raise AssertionError("Prior-history date cutoff was violated.")
                records.append(row)
            except Exception as exc:
                skipped += 1
                print(f"Skipped {target_date}: {away_team} at {home_team}: {exc}")

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError("No development predictions completed.")
    if not (pd.to_datetime(results["prior_max_date"]) < pd.to_datetime(results["game_date"])).all():
        raise AssertionError("At least one development row has a date-cutoff violation.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results.to_csv(DETAILS_PATH, index=False)

    summary_rows = []
    for variant_name in VARIANTS:
        metrics = regression_metrics(results["actual_total"], results[variant_name])
        summary_rows.append({
            "variant": variant_name,
            **metrics,
            "development_games": len(results),
            "skipped_games": skipped,
            "development_start": results["game_date"].min(),
            "development_end": results["game_date"].max(),
            "reserved_holdout_start": holdout_start.date().isoformat(),
            "reserved_holdout_end": history["game_date"].max().date().isoformat(),
            "reserved_holdout_rows_at_least": int(len(holdout_tail)),
            "selection_window": "development only; no candidate metrics calculated on reserved holdout",
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["mae", "rmse"], ascending=[True, True]
    ).reset_index(drop=True)
    summary.insert(0, "development_rank_by_mae", np.arange(1, len(summary) + 1))
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n===== CHRONOLOGICAL TOTALS ADJUSTMENT ABLATION (DEVELOPMENT ONLY) =====")
    print(summary[[
        "development_rank_by_mae", "variant", "games", "mae", "rmse",
        "bias", "within_5_pct", "within_10_pct"
    ]].to_string(index=False))
    print(
        f"\nDevelopment: {results['game_date'].min()} to {results['game_date'].max()} "
        f"({len(results)} predictions); skipped={skipped}"
    )
    print(
        f"Reserved recent holdout starts {holdout_start.date().isoformat()} "
        f"and ends {history['game_date'].max().date().isoformat()}. "
        "No candidate metrics were computed on this window."
    )
    print(f"Detailed results: {DETAILS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("This is adjustment ablation, not betting ROI or final independent validation.")

    return summary, results


if __name__ == "__main__":
    run_ablation()
