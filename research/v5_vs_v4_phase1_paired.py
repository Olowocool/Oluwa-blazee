#!/usr/bin/env python3
"""Paired point-in-time V5 control vs V4 pace/efficiency challenger.

Research only. Reuses the repository's validated NBA Stats game-ID and team-name
crosswalk. V5 receives the complete score history and applies its own strict
pre-target-date cutoff. V4 features are built date-by-date by v4_phase1_core.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from points_engine_v5 import calculate_matchup_points_v5
from research.v4_paired_v2_vs_v4_backtest import (
    build_paired_predictions,
    load_current_team_names,
    prepare_v2_score_history,
)
from research.v4_phase1_core import connect_and_load

OUT = Path("outputs/research/v5_vs_v4_phase1")
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261103


def date_cluster_bootstrap(results: pd.DataFrame) -> dict:
    """Resample whole game dates, preserving within-date dependence."""
    daily = results.groupby("game_date", sort=True).agg(
        v5_abs=("v5_absolute_error", "mean"),
        v4_abs=("v4_absolute_error", "mean"),
        games=("game_id", "count"),
    ).reset_index()
    if len(daily) < 10:
        raise RuntimeError(f"Need at least 10 distinct dates for cluster bootstrap; got {len(daily)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = (daily["v4_abs"] - daily["v5_abs"]).to_numpy(float)
    samples = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(differences), size=len(differences))
        samples[i] = differences[idx].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    observed = float(results["v4_absolute_error"].mean() - results["v5_absolute_error"].mean())
    return {
        "difference": "V4 MAE minus V5 MAE; negative favours V4",
        "distinct_dates": int(len(daily)),
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "observed_paired_mae_difference": observed,
        "cluster_bootstrap_95pct_interval": [float(lo), float(hi)],
        "interpretation": "date-cluster bootstrap; descriptive uncertainty interval",
    }


def metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    err = prediction.to_numpy(float) - actual.to_numpy(float)
    ae = np.abs(err)
    return {
        "games": int(len(err)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_prediction_minus_actual": float(err.mean()),
        "within_5_points_pct": float((ae <= 5).mean() * 100),
        "within_10_points_pct": float((ae <= 10).mean() * 100),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _ = connect_and_load()
    team_names = load_current_team_names()
    full_history, history_diagnostics = prepare_v2_score_history()

    def v5_predictor(*, home_team, away_team, history_df, game_date, **_ignored):
        # Deliberately pass the full canonical history. V5/V3/V4 enforce
        # game_date strictly; do not truncate history to V2's sufficient subset.
        prediction = calculate_matchup_points_v5(
            home_team=home_team,
            away_team=away_team,
            history_df=full_history,
            game_date=game_date,
            _prepared_cache=v5_cache,
        )
        return {"projected_total": float(prediction["raw_total"])}

    v5_cache = {}
    paired, diagnostics = build_paired_predictions(
        raw=raw,
        team_names=team_names,
        predictor=v5_predictor,
        v2_history=full_history,
    )
    results = paired.rename(columns={
        "v2_points_engine_total": "v5_total",
        "v4_pace_efficiency_total": "v4_total",
    }).copy()
    results["v5_error"] = results["v5_total"] - results["actual_total"]
    results["v4_error"] = results["v4_total"] - results["actual_total"]
    results["v5_absolute_error"] = results["v5_error"].abs()
    results["v4_absolute_error"] = results["v4_error"].abs()

    if results.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate paired game keys.")
    if not np.isfinite(results[["v5_total", "v4_total", "actual_total"]].to_numpy()).all():
        raise RuntimeError("Non-finite prediction or target in paired sample.")
    if not (pd.to_datetime(results["v5_max_history_date"]) < pd.to_datetime(results["game_date"])).all():
        raise RuntimeError("V5 history cutoff is not strictly before target date.")
    if not (pd.to_datetime(results["v4_max_history_date"]) < pd.to_datetime(results["game_date"])).all():
        raise RuntimeError("V4 history cutoff is not strictly before target date.")

    overall = {
        "V5_frozen_control": metrics(results["actual_total"], results["v5_total"]),
        "V4_pace_efficiency_challenger": metrics(results["actual_total"], results["v4_total"]),
    }
    by_season = {}
    for season, part in results.groupby("season", sort=True):
        by_season[str(int(season))] = {
            "V5": metrics(part["actual_total"], part["v5_total"]),
            "V4": metrics(part["actual_total"], part["v4_total"]),
            "paired_games": int(len(part)),
        }
    comparison = {
        "v4_lower_absolute_error_games": int((results["v4_absolute_error"] < results["v5_absolute_error"]).sum()),
        "v5_lower_absolute_error_games": int((results["v5_absolute_error"] < results["v4_absolute_error"]).sum()),
        "ties": int((results["v5_absolute_error"] == results["v4_absolute_error"]).sum()),
        "date_cluster_bootstrap": date_cluster_bootstrap(results),
    }
    results.to_csv(OUT / "paired_predictions.csv", index=False)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "control": "Frozen V5 = 0.55*V3 + 0.45*V4-90",
        "challenger": "V4 Phase 1 true-possession pace + opponent-adjusted scoring efficiency",
        "evaluation": "same NBA Stats regular-season game IDs, same actual totals, non-neutral games only",
        "overall": overall,
        "by_season": by_season,
        "paired_comparison": comparison,
        "data_diagnostics": {**history_diagnostics, **diagnostics},
        "strict_pre_game_history": True,
        "same_date_results_excluded": True,
        "production_modified": False,
        "paper_ledger_modified": False,
        "profitability_claim": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Evidence: {OUT.resolve()}")


if __name__ == "__main__":
    main()
