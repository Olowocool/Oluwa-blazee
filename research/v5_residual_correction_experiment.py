#!/usr/bin/env python3
"""Prequential V5 residual-correction experiment using point-in-time team features.

V5 remains frozen. This script tests whether prior-game scoring-mechanism and
pace/efficiency features can predict V5's error on future dates. Research only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from points_engine_v5 import calculate_matchup_points_v5
from research.v4_phase1_core import build_point_in_time, connect_and_load
from research.v4_paired_v2_vs_v4_backtest import (
    build_paired_predictions,
    load_current_team_names,
    prepare_v2_score_history,
)

OUT = ROOT / "outputs" / "research" / "v5_residual_correction"
BASE_HISTORY = ROOT / "data" / "historical_nba_scores.csv"
PATCH_FILES = [
    ROOT / "research" / "data" / "score_history_patch_2026-10-21_to_2026-10-29.csv",
    ROOT / "research" / "data" / "score_history_patch_2026-10-30.csv",
    ROOT / "research" / "data" / "score_history_patch_2026-10-31_to_2026-11-01.csv",
]
TEST_SEASONS = [2024, 2025, 2026]
MIN_TRAIN_GAMES = 1000
RIDGE_ALPHA = 25.0
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261103

PACE_EFFICIENCY_FEATURES = [
    "expected_possessions", "home_pace_est", "away_pace_est",
    "league_pace_prior", "league_off_eff_prior",
    "expected_home_efficiency", "expected_away_efficiency",
    "season_blend_weight", "home_games_prior", "away_games_prior",
]
SCORING_MECHANISM_FEATURES = [
    "home_recent_off_eff", "away_recent_off_eff",
    "home_recent_fga_per_poss", "away_recent_fga_per_poss",
    "home_recent_fta_per_poss", "away_recent_fta_per_poss",
    "home_recent_oreb_per_poss", "away_recent_oreb_per_poss",
    "home_recent_tov_per_poss", "away_recent_tov_per_poss",
    "home_recent_points_per_fga", "away_recent_points_per_fga",
]
FEATURE_FAMILIES = {
    "scoring_mechanism": ["v5_total", *SCORING_MECHANISM_FEATURES],
    "pace_efficiency": ["v5_total", *PACE_EFFICIENCY_FEATURES],
    "combined": ["v5_total", *PACE_EFFICIENCY_FEATURES, *SCORING_MECHANISM_FEATURES],
}


def load_research_score_history() -> tuple[pd.DataFrame, dict]:
    if not BASE_HISTORY.is_file():
        raise FileNotFoundError(f"Missing score history: {BASE_HISTORY}")
    frames = [pd.read_csv(BASE_HISTORY)]
    patch_rows = 0
    for path in PATCH_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required score patch: {path}")
        patch = pd.read_csv(path)
        patch_rows += len(patch)
        frames.append(patch)
    history, diagnostics = prepare_v2_score_history(pd.concat(frames, ignore_index=True))
    diagnostics.update({
        "patch_rows": int(patch_rows),
        "patch_files": [p.name for p in PATCH_FILES],
        "combined_rows": int(len(history)),
        "latest_game_date": history.game_date.max().date().isoformat(),
    })
    return history, diagnostics


def model_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    error = prediction.to_numpy(float) - actual.to_numpy(float)
    absolute = np.abs(error)
    return {
        "games": int(len(error)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_prediction_minus_actual": float(error.mean()),
        "within_5_points_pct": float((absolute <= 5).mean() * 100),
        "within_10_points_pct": float((absolute <= 10).mean() * 100),
    }


def date_cluster_bootstrap(results: pd.DataFrame, candidate_col: str) -> dict:
    daily = results.assign(
        baseline_ae=(results.v5_total - results.actual_total).abs(),
        candidate_ae=(results[candidate_col] - results.actual_total).abs(),
    ).groupby("game_date", sort=True).agg(
        baseline_ae=("baseline_ae", "mean"),
        candidate_ae=("candidate_ae", "mean"),
        games=("game_id", "count"),
    )
    diff = (daily.candidate_ae - daily.baseline_ae).to_numpy(float)
    weights = daily.games.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(diff), size=len(diff))
        draws[i] = np.average(diff[idx], weights=weights[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    observed = float(
        (results[candidate_col] - results.actual_total).abs().mean()
        - (results.v5_total - results.actual_total).abs().mean()
    )
    return {
        "candidate_minus_v5_mae": observed,
        "date_cluster_bootstrap_95pct_interval": [float(lo), float(hi)],
        "distinct_dates": int(len(daily)),
        "replicates": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "negative_favours_candidate": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _metadata = connect_and_load()
    full_history, history_diagnostics = load_research_score_history()
    team_names = load_current_team_names()

    # Build all target-date features once. The V4 feature builder freezes each
    # date's team history before appending any results from that date.
    features = build_point_in_time(raw)
    v5_cache: dict = {}
    v5_cutoffs: dict[str, str] = {}

    def v5_predictor(*, home_team, away_team, history_df, game_date, **_ignored):
        target = pd.Timestamp(game_date).normalize()
        prior = full_history.loc[full_history.game_date < target].copy()
        if prior.empty or not (prior.game_date < target).all():
            raise RuntimeError(f"Invalid V5 history cutoff for {target.date()}")
        v5_cutoffs[target.date().isoformat()] = prior.game_date.max().date().isoformat()
        prediction = calculate_matchup_points_v5(
            home_team=home_team, away_team=away_team,
            history_df=prior, game_date=target, _prepared_cache=v5_cache,
        )
        total = float(prediction["raw_total"])
        if not np.isfinite(total):
            raise RuntimeError("V5 produced a non-finite prediction")
        return {"projected_total": total}

    paired, pair_diagnostics = build_paired_predictions(
        raw=raw, team_names=team_names, v4_features=features,
        predictor=v5_predictor, v2_history=full_history,
    )
    feature_columns = sorted(set(
        PACE_EFFICIENCY_FEATURES + SCORING_MECHANISM_FEATURES
    ))
    feature_frame = features[
        ["season", "game_id", "game_date", "max_history_date", *feature_columns]
    ].copy().rename(columns={"max_history_date": "feature_max_history_date"})
    data = paired.merge(
        feature_frame, on=["season", "game_id"], how="left",
        validate="one_to_one", suffixes=("", "_feature"),
    )
    # build_paired_predictions may already carry some V4 feature columns.
    # Prefer the explicit point-in-time feature-frame values, while verifying
    # any overlapping columns agree before removing merge suffixes.
    for column in feature_columns:
        feature_column = f"{column}_feature"
        if feature_column in data.columns:
            if column in data.columns:
                left = pd.to_numeric(data[column], errors="coerce")
                right = pd.to_numeric(data[feature_column], errors="coerce")
                both = left.notna() & right.notna()
                if both.any() and not np.allclose(
                    left[both].to_numpy(float), right[both].to_numpy(float),
                    rtol=1e-10, atol=1e-10,
                ):
                    raise RuntimeError(f"Conflicting point-in-time feature values: {column}")
                data[column] = right.combine_first(left)
                data = data.drop(columns=[feature_column])
            else:
                data = data.rename(columns={feature_column: column})
    if data.game_date_feature.notna().any():
        mismatch = pd.to_datetime(data.game_date).dt.normalize().ne(
            pd.to_datetime(data.game_date_feature).dt.normalize()
        )
        if mismatch.any():
            raise RuntimeError("Paired target dates do not match feature target dates")
        data = data.drop(columns=["game_date_feature"])
    data = data.rename(columns={
        "v2_points_engine_total": "v5_total",
        "v2_max_history_date": "paired_helper_max_history_date",
        "v4_max_history_date": "v4_max_history_date",
    })
    data["game_date"] = pd.to_datetime(data.game_date).dt.normalize()
    data["actual_total"] = pd.to_numeric(data.actual_total, errors="raise")
    data["v5_total"] = pd.to_numeric(data.v5_total, errors="raise")
    data["v5_error"] = data.v5_total - data.actual_total
    data["v5_max_history_date"] = data.game_date.map(
        lambda d: v5_cutoffs.get(d.date().isoformat())
    )
    if data.v5_max_history_date.isna().any():
        raise RuntimeError("Missing recorded V5 cutoff")
    for cutoff in ("v5_max_history_date", "v4_max_history_date", "feature_max_history_date"):
        if not (pd.to_datetime(data[cutoff]) < data.game_date).all():
            raise RuntimeError(f"Non-point-in-time cutoff detected: {cutoff}")
    if data.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate paired game keys")
    if data[feature_columns].isna().all(axis=1).any():
        raise RuntimeError("At least one game has no scoring-mechanism feature data")

    test = data.loc[data.season.isin(TEST_SEASONS)].copy()
    if test.empty:
        raise RuntimeError("No games in the predeclared 2024-2026 test seasons")
    predictions = []
    # Prequential walk-forward: every test date is trained only on earlier
    # dates. Previous test-season outcomes may update later dates, as in live use.
    for target_date, day in test.groupby("game_date", sort=True):
        train = data.loc[data.game_date < target_date].copy()
        if len(train) < MIN_TRAIN_GAMES:
            continue
        day_predictions = day[["season", "game_id", "game_date", "actual_total", "v5_total", *feature_columns]].copy()
        for family, cols in FEATURE_FAMILIES.items():
            model = model_pipeline()
            model.fit(train[cols], train.v5_error)
            predicted_error = model.predict(day[cols])
            day_predictions[f"{family}_total"] = day.v5_total.to_numpy(float) - predicted_error
        day_predictions["training_games"] = int(len(train))
        day_predictions["training_latest_date"] = train.game_date.max().date().isoformat()
        if not (pd.to_datetime(day_predictions.training_latest_date) < target_date).all():
            raise RuntimeError("Same-day or future target leaked into residual training")
        predictions.append(day_predictions)

    if not predictions:
        raise RuntimeError("No out-of-sample residual-correction predictions were produced")
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)
    if not (pd.to_datetime(result.training_latest_date) < result.game_date).all():
        raise RuntimeError("Residual model training cutoff is not strictly pregame")

    # Robustness check: fit once at the beginning of each test season,
    # then freeze coefficients for the entire season (no within-season updates).
    season_frozen_families = {}
    season_frozen_cutoffs = {}
    for season in TEST_SEASONS:
        season_part = result.loc[result.season.eq(season)].copy()
        required_frozen_features = set().union(*[set(cols) for cols in FEATURE_FAMILIES.values()])
        missing_frozen_features = sorted(required_frozen_features - set(season_part.columns))
        if missing_frozen_features:
            raise RuntimeError(
                f"Season-frozen predictions missing feature columns for {season}: "
                f"{missing_frozen_features}"
            )
        season_start = season_part.game_date.min()
        season_train = data.loc[data.game_date < season_start].copy()
        if len(season_train) < MIN_TRAIN_GAMES:
            raise RuntimeError(
                f"Insufficient pre-season history for frozen validation {season}: {len(season_train)}"
            )
        season_frozen_cutoffs[str(season)] = {
            "first_test_date": season_start.date().isoformat(),
            "training_games": int(len(season_train)),
            "training_latest_date": season_train.game_date.max().date().isoformat(),
        }
        if not (season_train.game_date.max() < season_start):
            raise RuntimeError(f"Season-frozen training cutoff failed for {season}")
        for family, cols in FEATURE_FAMILIES.items():
            frozen_model = model_pipeline()
            frozen_model.fit(season_train[cols], season_train.v5_error)
            result.loc[result.season.eq(season), f"season_frozen_{family}_total"] = (
                result.loc[result.season.eq(season), "v5_total"].to_numpy(float)
                - frozen_model.predict(season_part[cols])
            )

    overall = {"V5_frozen_control": metrics(result.actual_total, result.v5_total)}
    by_family = {}
    for family in FEATURE_FAMILIES:
        col = f"{family}_total"
        by_family[family] = {
            "metrics": metrics(result.actual_total, result[col]),
            "paired_date_cluster_bootstrap": date_cluster_bootstrap(result, col),
        }
    by_season = {}
    season_frozen_by_family = {}
    season_frozen_by_season = {}
    for family in FEATURE_FAMILIES:
        col = f"season_frozen_{family}_total"
        season_frozen_by_family[family] = {
            "metrics": metrics(result.actual_total, result[col]),
            "paired_date_cluster_bootstrap": date_cluster_bootstrap(result, col),
        }
    for season, part in result.groupby("season", sort=True):
        by_season[str(int(season))] = {
            "V5": metrics(part.actual_total, part.v5_total),
            **{
                family: metrics(part.actual_total, part[f"{family}_total"])
                for family in FEATURE_FAMILIES
            },
        }
        season_frozen_by_season[str(int(season))] = {
            "V5": metrics(part.actual_total, part.v5_total),
            **{
                family: metrics(part.actual_total, part[f"season_frozen_{family}_total"])
                for family in FEATURE_FAMILIES
            },
        }

    result.to_csv(OUT / "prequential_predictions.csv", index=False)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V5 residual correction from point-in-time team scoring and pace/efficiency features",
        "frozen_control": "V5 = 0.55*V3 + 0.45*V4-90; no V5 formula or production changes",
        "test_seasons": TEST_SEASONS,
        "test_games": int(len(result)),
        "test_dates": int(result.game_date.nunique()),
        "minimum_prior_training_games": MIN_TRAIN_GAMES,
        "training_policy": "Expanding prequential; strictly earlier game dates only; prior test outcomes may update later dates",
        "residual_target": "V5 prediction minus actual total; candidate subtracts predicted V5 error from frozen V5",
        "model": f"Median imputation + standardization + Ridge(alpha={RIDGE_ALPHA}); fixed before test",
        "feature_families": FEATURE_FAMILIES,
        "overall": overall,
        "candidate_results": by_family,
        "by_season": by_season,
        "season_frozen_validation": {
            "policy": "One model fit per test season using only games dated before that season's first eligible prediction; coefficients remain frozen for the entire season",
            "season_training_cutoffs": season_frozen_cutoffs,
            "candidate_results": season_frozen_by_family,
            "by_season": season_frozen_by_season,
        },
        "point_in_time_cutoffs_verified": True,
        "same_date_games_excluded_from_training": True,
        "data_diagnostics": {**history_diagnostics, **pair_diagnostics},
        "limitations": [
            "Retrospective team-game statistics are cut off by game date but historical data vintages are not archived.",
            "This evaluates score-prediction correction, not sportsbook market beating or betting ROI.",
            "No player availability, player impact, market data, travel, or positional matchup inputs are used.",
        ],
        "production_modified": False,
        "paper_ledger_modified": False,
        "V5_modified": False,
        "profitability_claim": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Evidence: {OUT.resolve()}")


if __name__ == "__main__":
    main()
