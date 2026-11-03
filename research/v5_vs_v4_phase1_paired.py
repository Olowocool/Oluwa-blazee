#!/usr/bin/env python3
"""Paired point-in-time V5 control vs V4 pace/efficiency challenger.

Research only. Uses exact NBA Stats season/game IDs and the repository's team
crosswalk. V5 receives only score-history rows strictly before each target date.
The research score patches are appended to the base history without modifying it.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("outputs/research/v5_vs_v4_phase1")
BASE_HISTORY = ROOT / "data" / "historical_nba_scores.csv"
PATCH_FILES = [
    ROOT / "research" / "data" / "score_history_patch_2026-10-21_to_2026-10-29.csv",
    ROOT / "research" / "data" / "score_history_patch_2026-10-30.csv",
    ROOT / "research" / "data" / "score_history_patch_2026-10-31_to_2026-11-01.csv",
]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261103


def load_research_score_history() -> tuple[pd.DataFrame, dict]:
    """Load base scores plus committed research-only patches, validating keys."""
    if not BASE_HISTORY.is_file():
        raise FileNotFoundError(f"Base score history is missing: {BASE_HISTORY}")
    frames = [pd.read_csv(BASE_HISTORY)]
    patch_rows = 0
    loaded_patches = []
    for path in PATCH_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Required research score patch is missing: {path}")
        patch = pd.read_csv(path)
        patch_rows += len(patch)
        loaded_patches.append(path.name)
        frames.append(patch)
    combined = pd.concat(frames, ignore_index=True)
    history, diagnostics = prepare_v2_score_history(combined)
    diagnostics.update({
        "research_score_patch_files": loaded_patches,
        "research_score_patch_rows": int(patch_rows),
        "combined_score_history_rows": int(len(history)),
        "combined_score_history_latest": history["game_date"].max().date().isoformat(),
    })
    return history, diagnostics


def date_cluster_bootstrap(results: pd.DataFrame) -> dict:
    """Resample whole dates and retain the game-weighted MAE estimand."""
    daily = results.groupby("game_date", sort=True).agg(
        v5_abs=("v5_absolute_error", "mean"),
        v4_abs=("v4_absolute_error", "mean"),
        games=("game_id", "count"),
    ).reset_index()
    if len(daily) < 10:
        raise RuntimeError(
            f"Need at least 10 distinct dates for cluster bootstrap; got {len(daily)}"
        )
    differences = (daily["v4_abs"] - daily["v5_abs"]).to_numpy(float)
    weights = daily["games"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(differences), size=len(differences))
        samples[i] = np.average(differences[idx], weights=weights[idx])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    observed = float(
        results["v4_absolute_error"].mean() - results["v5_absolute_error"].mean()
    )
    return {
        "difference": "V4 MAE minus V5 MAE; negative favours V4",
        "distinct_dates": int(len(daily)),
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "observed_game_weighted_paired_mae_difference": observed,
        "date_cluster_bootstrap_95pct_interval": [float(lo), float(hi)],
        "interpretation": (
            "Whole dates are resampled; each replicate retains the game-weighted "
            "MAE estimand. Descriptive uncertainty, not a profitability test."
        ),
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
    raw, _metadata = connect_and_load()
    team_names = load_current_team_names()
    full_history, history_diagnostics = load_research_score_history()
    v5_cache: dict = {}
    v5_cutoff_by_date: dict[str, str] = {}

    def v5_predictor(*, home_team, away_team, history_df, game_date, **_ignored):
        # Enforce the cutoff before calling V5, rather than relying solely on
        # the component engines' internal cutoff. No same-day or future row can
        # reach either V3 or V4-90 through this benchmark wrapper.
        target = pd.Timestamp(game_date).normalize()
        prior = full_history.loc[full_history["game_date"] < target].copy()
        if prior.empty:
            raise RuntimeError(f"No strictly prior V5 score history for {target.date()}.")
        if not (prior["game_date"] < target).all():
            raise AssertionError("V5 wrapper admitted same-date/future score history.")
        v5_cutoff_by_date[target.date().isoformat()] = (
            prior["game_date"].max().date().isoformat()
        )
        prediction = calculate_matchup_points_v5(
            home_team=home_team,
            away_team=away_team,
            history_df=prior,
            game_date=target,
            _prepared_cache=v5_cache,
        )
        total = float(prediction["raw_total"])
        if not np.isfinite(total):
            raise RuntimeError("V5 returned a non-finite raw total.")
        return {"projected_total": total}

    paired, diagnostics = build_paired_predictions(
        raw=raw,
        team_names=team_names,
        predictor=v5_predictor,
        v2_history=full_history,
    )
    results = paired.rename(columns={
        "v2_points_engine_total": "v5_total",
        "v4_pace_efficiency_total": "v4_total",
        "v2_max_history_date": "paired_helper_max_history_date",
    }).copy()
    results["v5_max_history_date"] = results["game_date"].map(v5_cutoff_by_date)
    if results["v5_max_history_date"].isna().any():
        raise RuntimeError("A paired target date has no recorded V5 history cutoff.")

    results["v5_error"] = results["v5_total"] - results["actual_total"]
    results["v4_error"] = results["v4_total"] - results["actual_total"]
    results["v5_absolute_error"] = results["v5_error"].abs()
    results["v4_absolute_error"] = results["v4_error"].abs()

    if results.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate paired game keys.")
    if not np.isfinite(
        results[["v5_total", "v4_total", "actual_total"]].to_numpy()
    ).all():
        raise RuntimeError("Non-finite prediction or target in paired sample.")
    if not (
        pd.to_datetime(results["v5_max_history_date"])
        < pd.to_datetime(results["game_date"])
    ).all():
        raise RuntimeError("V5 history cutoff is not strictly before target date.")
    if not (
        pd.to_datetime(results["v4_max_history_date"])
        < pd.to_datetime(results["game_date"])
    ).all():
        raise RuntimeError("V4 history cutoff is not strictly before target date.")

    overall = {
        "V5_frozen_control": metrics(results["actual_total"], results["v5_total"]),
        "V4_pace_efficiency_challenger": metrics(
            results["actual_total"], results["v4_total"]
        ),
    }
    by_season = {}
    for season, part in results.groupby("season", sort=True):
        by_season[str(int(season))] = {
            "V5": metrics(part["actual_total"], part["v5_total"]),
            "V4": metrics(part["actual_total"], part["v4_total"]),
            "paired_games": int(len(part)),
        }
    comparison = {
        "v4_lower_absolute_error_games": int(
            (results["v4_absolute_error"] < results["v5_absolute_error"]).sum()
        ),
        "v5_lower_absolute_error_games": int(
            (results["v5_absolute_error"] < results["v4_absolute_error"]).sum()
        ),
        "ties": int(
            (results["v5_absolute_error"] == results["v4_absolute_error"]).sum()
        ),
        "date_cluster_bootstrap": date_cluster_bootstrap(results),
    }
    results.to_csv(OUT / "paired_predictions.csv", index=False)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "control": "Frozen V5 = 0.55*V3 + 0.45*V4-90",
        "challenger": (
            "V4 Phase 1 true-possession pace + opponent-adjusted scoring efficiency"
        ),
        "evaluation": (
            "Same NBA Stats regular-season season/game IDs and actual totals; "
            "non-neutral games only."
        ),
        "overall": overall,
        "by_season": by_season,
        "paired_comparison": comparison,
        "data_diagnostics": {**history_diagnostics, **diagnostics},
        "strict_pre_game_history": True,
        "same_date_results_excluded": True,
        "v5_cutoff_dates_recorded": True,
        "production_modified": False,
        "paper_ledger_modified": False,
        "profitability_claim": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Evidence: {OUT.resolve()}")


if __name__ == "__main__":
    main()
