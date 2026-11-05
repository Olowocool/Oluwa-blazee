#!/usr/bin/env python3
"""Point-in-time Stage C player-feature challenger against frozen V5.

Consumes the immutable Stage C artifact from the successful Phase 2 run.
V5 is recomputed on identical NBA Stats games using strictly pregame score
history. Candidate residual models are expanding-window and date-prequential.
Research only: no production, V5, or paper-ledger changes.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/research/stage_c_v5_comparison"
ARTIFACT = Path(os.environ.get("STAGE_C_ARTIFACT_DIR", "outputs/research/stage_c_artifact"))
PHASE2_DIR = ARTIFACT / "v4_phase2"
STAGEC_DIR = ARTIFACT / "player_intelligence_stage_c"
PROFILE_FILE = ARTIFACT / "player_intelligence_profiles/player_game_profiles.parquet"
SCORE_FILE = ROOT / "data/historical_nba_scores.csv"
PATCHES = [
    ROOT / "research/data/score_history_patch_2026-10-21_to_2026-10-29.csv",
    ROOT / "research/data/score_history_patch_2026-10-30.csv",
    ROOT / "research/data/score_history_patch_2026-10-31_to_2026-11-01.csv",
]
TEST_SEASONS = [2025, 2026]
MIN_TRAIN_GAMES = 300
RIDGE_ALPHA = 25.0
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261104

RAPM_INJURY_FEATURES = [
    "home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta",
    "home_minutes_changed", "away_minutes_changed",
    "home_reported_out_minutes", "away_reported_out_minutes",
    "home_questionable_minutes", "away_questionable_minutes",
    "home_reported_players_without_prior_team_history",
    "away_reported_players_without_prior_team_history",
    "home_players_excluded_by_opposing_report",
    "away_players_excluded_by_opposing_report",
]
PROFILE_METRICS = [
    "expected_pts", "expected_ast", "expected_tov", "expected_oreb",
    "expected_dreb", "expected_stl", "expected_blk", "minute_weighted_ts",
    "minute_weighted_usage", "minute_weighted_three_pa_share",
    "minute_weighted_three_pct", "minute_weighted_free_throw_rate",
    "profiled_minutes_share",
]
PROFILE_FEATURES = [f"{side}_profile_{metric}" for side in ("home", "away") for metric in PROFILE_METRICS]
FAMILIES = {
    "rapm_injury": RAPM_INJURY_FEATURES,
    "player_profiles": PROFILE_FEATURES,
    "combined": RAPM_INJURY_FEATURES + PROFILE_FEATURES,
}


def load_score_history():
    from research.v4_paired_v2_vs_v4_backtest import prepare_v2_score_history
    if not SCORE_FILE.is_file():
        raise FileNotFoundError(f"Missing score history: {SCORE_FILE}")
    frames = [pd.read_csv(SCORE_FILE)]
    for path in PATCHES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required research score patch: {path}")
        frames.append(pd.read_csv(path))
    history, diagnostics = prepare_v2_score_history(pd.concat(frames, ignore_index=True))
    diagnostics["patch_files"] = [p.name for p in PATCHES]
    diagnostics["patch_rows"] = int(sum(len(x) for x in frames[1:]))
    diagnostics["combined_rows"] = int(len(history))
    diagnostics["latest_game_date"] = history.game_date.max().date().isoformat()
    return history, diagnostics


def build_profile_features(candidate_path: Path) -> tuple[pd.DataFrame, dict]:
    """Recreate Stage C team aggregates from projected, PIT-joined players."""
    p = pd.read_parquet(candidate_path)
    required = {
        "season", "game_id", "game_date", "side", "player_id", "exp_min",
        "profile_source_date", "profile_through_date", "prior_game_days",
        "l20_min", "l20_pts", "l20_ast", "l20_tov", "l20_oreb", "l20_reb",
        "l20_stl", "l20_blk", "l20_fga", "l20_fg3a", "l20_fg3m", "l20_fta",
        "l20_ts_pct", "l20_usg_pct",
    }
    missing = sorted(required - set(p.columns))
    if missing:
        raise RuntimeError(f"Stage C candidate profile file missing columns: {missing}")
    p["game_id"] = p.game_id.astype(str)
    p["game_date"] = pd.to_datetime(p.game_date).dt.normalize()
    for c in required - {"season", "game_id", "game_date", "side", "player_id",
                         "profile_source_date", "profile_through_date"}:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    source_date = pd.to_datetime(p.profile_source_date, errors="coerce")
    through_date = pd.to_datetime(p.profile_through_date, errors="coerce")
    source_bad = int((source_date.notna() & source_date.ge(p.game_date)).sum())
    history_bad = int((through_date.notna() & through_date.ge(p.game_date)).sum())
    if source_bad or history_bad:
        raise RuntimeError(f"Player profile PIT violations: source={source_bad}, history={history_bad}")

    eligible = p.l20_min.ge(60) & p.prior_game_days.ge(5)
    p["pts_per36"] = np.where(eligible, 36*p.l20_pts/p.l20_min, np.nan)
    p["ast_per36"] = np.where(eligible, 36*p.l20_ast/p.l20_min, np.nan)
    p["tov_per36"] = np.where(eligible, 36*p.l20_tov/p.l20_min, np.nan)
    p["oreb_per36"] = np.where(eligible, 36*p.l20_oreb/p.l20_min, np.nan)
    p["dreb_per36"] = np.where(eligible, 36*(p.l20_reb-p.l20_oreb)/p.l20_min, np.nan)
    p["stl_per36"] = np.where(eligible, 36*p.l20_stl/p.l20_min, np.nan)
    p["blk_per36"] = np.where(eligible, 36*p.l20_blk/p.l20_min, np.nan)
    p["ts_eligible"] = p.l20_ts_pct.where(p.l20_fga.ge(20))
    p["usage_eligible"] = p.l20_usg_pct.where(p.l20_fga.ge(20))
    p["three_pa_share"] = (p.l20_fg3a/p.l20_fga).where(p.l20_fga.ge(20))
    p["three_pct"] = (p.l20_fg3m/p.l20_fg3a).where(p.l20_fg3a.ge(10))
    p["free_throw_rate"] = (p.l20_fta/p.l20_fga).where(p.l20_fga.ge(20))
    p["profile_eligible"] = eligible

    def team_agg(g):
        w = g.exp_min.fillna(0).clip(lower=0)
        out = {
            "expected_pts": float((w*g.pts_per36.fillna(0)/36).sum()),
            "expected_ast": float((w*g.ast_per36.fillna(0)/36).sum()),
            "expected_tov": float((w*g.tov_per36.fillna(0)/36).sum()),
            "expected_oreb": float((w*g.oreb_per36.fillna(0)/36).sum()),
            "expected_dreb": float((w*g.dreb_per36.fillna(0)/36).sum()),
            "expected_stl": float((w*g.stl_per36.fillna(0)/36).sum()),
            "expected_blk": float((w*g.blk_per36.fillna(0)/36).sum()),
            "profiled_minutes_share": float(w[g.profile_eligible].sum()/240),
        }
        for source, target in [
            ("ts_eligible","minute_weighted_ts"),
            ("usage_eligible","minute_weighted_usage"),
            ("three_pa_share","minute_weighted_three_pa_share"),
            ("three_pct","minute_weighted_three_pct"),
            ("free_throw_rate","minute_weighted_free_throw_rate"),
        ]:
            valid = g[source].notna() & w.gt(0)
            out[target] = float((w[valid]*g.loc[valid,source]).sum()/w[valid].sum()) if valid.any() else np.nan
        return pd.Series(out)

    keys = ["season", "game_id", "game_date", "side"]
    team = p.groupby(keys, sort=False).apply(team_agg, include_groups=False).reset_index()
    pieces = []
    for side in ("home", "away"):
        x = team.loc[team.side.eq(side), ["season","game_id","game_date",*PROFILE_METRICS]].copy()
        x = x.rename(columns={m:f"{side}_profile_{m}" for m in PROFILE_METRICS})
        pieces.append(x)
    wide = pieces[0].merge(pieces[1], on=["season","game_id","game_date"], how="inner", validate="one_to_one")
    diagnostics = {
        "candidate_player_rows": int(len(p)),
        "candidate_games": int(p.game_id.nunique()),
        "profile_source_date_violations": source_bad,
        "profile_history_date_violations": history_bad,
        "profiled_candidate_rows": int(eligible.sum()),
        "team_game_profile_rows": int(len(wide)),
    }
    return wide, diagnostics


def metric(actual, prediction):
    e = np.asarray(prediction, float)-np.asarray(actual, float)
    ae = np.abs(e)
    return {
        "games": int(len(e)), "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias_prediction_minus_actual": float(e.mean()),
        "within_5_points_pct": float((ae<=5).mean()*100),
        "within_10_points_pct": float((ae<=10).mean()*100),
    }


def cluster_bootstrap(frame, candidate_col):
    daily = frame.assign(
        base_ae=(frame.v5_total-frame.actual_total).abs(),
        candidate_ae=(frame[candidate_col]-frame.actual_total).abs(),
    ).groupby("game_date", sort=True).agg(
        base_ae=("base_ae","mean"), candidate_ae=("candidate_ae","mean"),
        games=("game_id","count"),
    )
    diff = (daily.candidate_ae-daily.base_ae).to_numpy(float)
    weights = daily.games.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0,len(diff),size=len(diff))
        draws[i] = np.average(diff[idx],weights=weights[idx])
    lo,hi = np.quantile(draws,[.025,.975])
    return {
        "candidate_minus_v5_mae": float((frame[candidate_col]-frame.actual_total).abs().mean()-(frame.v5_total-frame.actual_total).abs().mean()),
        "95pct_date_cluster_ci": [float(lo),float(hi)],
        "dates": int(len(daily)), "replicates": BOOTSTRAP_REPS,
        "negative_favours_candidate": True,
    }


def pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def export_live_player_strength_calibration(data: pd.DataFrame) -> dict:
    """Fit a frozen live RAPM/injury correction using only completed pre-2026-27 games."""
    cutoff = pd.Timestamp("2026-07-01")
    train = data.loc[pd.to_datetime(data.game_date).dt.normalize() < cutoff].copy()
    train = train.dropna(subset=["v5_total", "actual_total"])
    if len(train) < MIN_TRAIN_GAMES:
        raise RuntimeError(f"Insufficient pre-2026-27 calibration rows: {len(train)}")
    if train.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate historical game rows in player-strength calibration")
    if not (pd.to_datetime(train.game_date).dt.normalize() < cutoff).all():
        raise RuntimeError("Player-strength calibration includes same-day/future rows")

    features = list(RAPM_INJURY_FEATURES)
    rapm_columns = features[:4]
    rapm_values = train[rapm_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rapm_nonzero = {
        col: int(rapm_values[col].abs().gt(1e-9).sum())
        for col in rapm_columns
    }
    if not any(rapm_nonzero.values()):
        raise RuntimeError(
            "RAPM/injury total-impact features are all zero in calibration data; "
            f"refusing to export a misleading player model. nonzero_rows={rapm_nonzero}"
        )
    model = pipeline()
    target = train.v5_total.astype(float) - train.actual_total.astype(float)
    model.fit(train[features], target)
    imputer, scaler, ridge = (
        model.named_steps["imputer"], model.named_steps["scale"], model.named_steps["ridge"]
    )
    payload = {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "research_shadow_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_cutoff_exclusive": cutoff.date().isoformat(),
        "training_first_date": pd.to_datetime(train.game_date).min().date().isoformat(),
        "training_last_date": pd.to_datetime(train.game_date).max().date().isoformat(),
        "training_games": int(len(train)),
        "training_dates": int(pd.to_datetime(train.game_date).dt.normalize().nunique()),
        "feature_family": "Stage C RAPM + injury + full-rotation features",
        "features": features,
        "rapm_feature_nonzero_rows": rapm_nonzero,
        "imputer": {"strategy": "constant", "fill_value": 0.0,
                    "statistics": [float(x) for x in imputer.statistics_]},
        "standardizer": {"mean": [float(x) for x in scaler.mean_],
                         "scale": [float(x) for x in scaler.scale_]},
        "ridge": {"alpha": float(RIDGE_ALPHA), "intercept": float(ridge.intercept_),
                  "coefficients": [float(x) for x in ridge.coef_]},
        "target": "frozen_v5_total_minus_actual_total",
        "prediction_formula": "player_adjusted_total = frozen_v5_total - predicted_v5_error",
        "point_in_time_policy": "training date strictly before target; live player/RAPM history strictly before target",
        "betting_enabled": False,
        "probabilities_calibrated": False,
        "profitability_claim": False,
        "v5_modified": False,
        "production_modified": False,
        "paper_ledger_modified": False,
    }
    if not all(np.isfinite(payload["standardizer"]["mean"])):
        raise RuntimeError("Non-finite player-strength calibration means")
    if not all(np.isfinite(payload["standardizer"]["scale"])) or any(
        x <= 0 for x in payload["standardizer"]["scale"]
    ):
        raise RuntimeError("Invalid player-strength calibration scales")
    if not all(np.isfinite(payload["ridge"]["coefficients"])) or not np.isfinite(
        payload["ridge"]["intercept"]
    ):
        raise RuntimeError("Non-finite player-strength calibration coefficients")

    output_path = ROOT / "research/live_player_strength_calibration.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("LIVE_PLAYER_STRENGTH_CALIBRATION_JSON_BEGIN")
    print(json.dumps(payload, separators=(",", ":")))
    print("LIVE_PLAYER_STRENGTH_CALIBRATION_JSON_END")
    return {"path": str(output_path), "training_games": payload["training_games"],
            "training_first_date": payload["training_first_date"],
            "training_last_date": payload["training_last_date"],
            "feature_count": len(features), "status": payload["status"]}


def main():
    for f in [PHASE2_DIR/"v4_phase2_game_features.csv", STAGEC_DIR/"stage_c_candidate_profiles.parquet", PROFILE_FILE]:
        if not f.is_file():
            raise FileNotFoundError(f"Required immutable Stage C artifact file missing: {f}")

    from points_engine_v5 import calculate_matchup_points_v5
    from research.v4_phase1_core import connect_and_load
    from research.v4_paired_v2_vs_v4_backtest import (
        build_paired_predictions, load_current_team_names,
    )

    raw, source_metadata = connect_and_load()
    team_names = load_current_team_names()
    history, history_diag = load_score_history()
    v5_cache, cutoffs = {}, {}

    def v5_predictor(*, home_team, away_team, history_df, game_date, **kwargs):
        target = pd.Timestamp(game_date).normalize()
        prior = history.loc[history.game_date < target].copy()
        if prior.empty or not (prior.game_date < target).all():
            raise RuntimeError(f"V5 history is not strictly pregame for {target.date()}")
        cutoffs[target.date().isoformat()] = prior.game_date.max().date().isoformat()
        pred = calculate_matchup_points_v5(
            home_team=home_team, away_team=away_team, history_df=prior,
            game_date=target, _prepared_cache=v5_cache,
        )
        value = float(pred["raw_total"])
        if not np.isfinite(value):
            raise RuntimeError("V5 generated non-finite prediction")
        return {"projected_total": value}

    paired, pair_diag = build_paired_predictions(
        raw=raw, team_names=team_names, predictor=v5_predictor, v2_history=history,
    )
    paired = paired.rename(columns={
        "v2_points_engine_total":"v5_total",
        "v2_max_history_date":"v5_helper_history_date",
    })
    # Normalize season join keys across CSV and Parquet sources. Pandas may
    # infer one source as int64 and another as object despite equal values.
    def normalize_season_key(frame, label):
        if "season" not in frame.columns:
            raise RuntimeError(f"{label} is missing season")
        season = pd.to_numeric(frame["season"], errors="coerce")
        if season.isna().any():
            raise RuntimeError(f"{label} contains non-numeric season keys")
        frame["season"] = season.astype("int64")

    normalize_season_key(paired, "V5 paired predictions")
    paired["game_date"] = pd.to_datetime(paired.game_date).dt.normalize()
    paired["v5_max_history_date"] = paired.game_date.map(lambda d: cutoffs.get(d.date().isoformat()))
    if paired.v5_max_history_date.isna().any() or not (
        pd.to_datetime(paired.v5_max_history_date) < paired.game_date
    ).all():
        raise RuntimeError("V5 pregame history cutoff validation failed")

    phase = pd.read_csv(PHASE2_DIR/"v4_phase2_game_features.csv", dtype={"game_id":"string"})
    normalize_season_key(phase, "Phase2 game features")
    phase["game_id"] = phase.game_id.astype(str)
    phase["game_date"] = pd.to_datetime(phase.game_date).dt.normalize()
    phase["rapm_snapshot_date"] = pd.to_datetime(phase.rapm_snapshot_date, errors="coerce")

    # snapshot_date is an as-of label, not the last observation date.
    # A snapshot labelled on game day is valid only when its fitted history
    # ends strictly before that day. Verify the actual manifest cutoff rather
    # than incorrectly rejecting same-day-labelled pregame snapshots.
    rapm_manifest_path = PHASE2_DIR / "v4_phase2_rapm_snapshots.csv"
    if not rapm_manifest_path.is_file():
        raise FileNotFoundError(f"Missing RAPM snapshot manifest: {rapm_manifest_path}")
    rapm_manifest = pd.read_csv(rapm_manifest_path)
    required_manifest = {"season", "snapshot_date", "max_history_date", "history_rows"}
    missing_manifest = sorted(required_manifest - set(rapm_manifest.columns))
    if missing_manifest:
        raise RuntimeError(f"RAPM manifest missing required fields: {missing_manifest}")
    normalize_season_key(rapm_manifest, "RAPM snapshot manifest")
    rapm_manifest["snapshot_date"] = pd.to_datetime(rapm_manifest.snapshot_date, errors="coerce").dt.normalize()
    rapm_manifest["max_history_date"] = pd.to_datetime(rapm_manifest.max_history_date, errors="coerce").dt.normalize()
    if rapm_manifest.duplicated(["season", "snapshot_date"]).any():
        raise RuntimeError("Duplicate season/snapshot-date rows in RAPM manifest")
    manifest_bad = int((
        rapm_manifest.snapshot_date.isna()
        | rapm_manifest.max_history_date.isna()
        | rapm_manifest.max_history_date.ge(rapm_manifest.snapshot_date)
        | pd.to_numeric(rapm_manifest.history_rows, errors="coerce").fillna(0).le(0)
    ).sum())
    if manifest_bad:
        raise RuntimeError(f"RAPM manifest contains {manifest_bad} invalid snapshot history cutoffs")
    phase = phase.merge(
        rapm_manifest[["season", "snapshot_date", "max_history_date"]].rename(columns={
            "snapshot_date": "rapm_snapshot_date",
            "max_history_date": "rapm_max_history_date",
        }),
        on=["season", "rapm_snapshot_date"], how="left", validate="many_to_one",
    )
    missing_snapshot = int(phase.rapm_max_history_date.isna().sum())
    future_snapshot = int((
        phase.rapm_snapshot_date.notna()
        & phase.rapm_snapshot_date.gt(phase.game_date)
    ).sum())
    non_pregame_history = int((
        phase.rapm_max_history_date.notna()
        & phase.rapm_max_history_date.ge(phase.game_date)
    ).sum())
    rapm_bad = missing_snapshot + future_snapshot + non_pregame_history
    if rapm_bad:
        raise RuntimeError(
            "RAPM point-in-time validation failed: "
            f"missing_manifest_match={missing_snapshot}, "
            f"snapshot_after_game_date={future_snapshot}, "
            f"history_not_strictly_prior={non_pregame_history}"
        )
    phase["game_id"] = phase.game_id.astype(str)
    phase["game_date"] = pd.to_datetime(phase.game_date).dt.normalize()
    phase["rapm_snapshot_date"] = pd.to_datetime(phase.rapm_snapshot_date, errors="coerce")
    profile_wide, profile_diag = build_profile_features(STAGEC_DIR/"stage_c_candidate_profiles.parquet")
    normalize_season_key(profile_wide, "Stage C player profile features")

    rapm_cols = ["season","game_id","game_date",*RAPM_INJURY_FEATURES]
    for c in RAPM_INJURY_FEATURES:
        if c not in phase.columns:
            raise RuntimeError(f"Missing RAPM/injury feature: {c}")
    stage = phase[rapm_cols].merge(
        profile_wide, on=["season","game_id","game_date"], how="inner", validate="one_to_one"
    )
    data = paired.merge(
        stage, on=["season","game_id","game_date"], how="inner", validate="one_to_one"
    )
    if data.empty:
        raise RuntimeError("No common games between V5 predictions and Stage C feature artifact")
    if data.duplicated(["season","game_id"]).any():
        raise RuntimeError("Duplicate game key after Stage C/V5 merge")
    # The merged frame intentionally includes earlier seasons as expanding-window
    # training history. Restrict only emitted predictions to TEST_SEASONS below.

    # Require a common complete target sample across all candidate families.
    # Missing component metrics are retained and imputed from prior training only.
    predictions = []
    for target_date, day in data.groupby("game_date", sort=True):
        season = int(day.season.iloc[0])
        if season not in TEST_SEASONS:
            continue
        train = data.loc[data.game_date.lt(target_date)].copy()
        if len(train) < MIN_TRAIN_GAMES:
            continue
        out = day[["season","game_id","game_date","actual_total","v5_total"]].copy()
        out["training_games"] = int(len(train))
        out["training_latest_date"] = train.game_date.max().date().isoformat()
        if not (train.game_date.max() < target_date):
            raise RuntimeError("Same-date or future residual training data detected")
        for family, cols in FAMILIES.items():
            model = pipeline()
            model.fit(train[cols], train.v5_total-train.actual_total)
            predicted_error = model.predict(day[cols])
            out[f"{family}_total"] = day.v5_total.to_numpy(float)-predicted_error
        predictions.append(out)

    if not predictions:
        raise RuntimeError("No prequential Stage C vs V5 predictions produced")
    result = pd.concat(predictions,ignore_index=True).sort_values(["game_date","game_id"]).reset_index(drop=True)
    if not (pd.to_datetime(result.training_latest_date) < result.game_date).all():
        raise RuntimeError("Residual training cutoff violation")
    if set(result.season.unique()) != set(TEST_SEASONS):
        raise RuntimeError(f"Expected both test seasons {TEST_SEASONS}, got {sorted(result.season.unique())}")

    live_calibration = export_live_player_strength_calibration(data)

    overall = {"V5_frozen_control": metric(result.actual_total,result.v5_total)}
    candidate_results = {}
    for family in FAMILIES:
        col=f"{family}_total"
        candidate_results[family] = {
            "overall":metric(result.actual_total,result[col]),
            "paired_date_cluster_bootstrap":cluster_bootstrap(result,col),
        }
    # A player forecast is enabled only when the corrected RAPM family beats
    # frozen V5 on the common chronological holdout and its date-cluster CI is
    # wholly below zero. This is an accuracy gate, not a betting-probability gate.
    player_oos = candidate_results["rapm_injury"]
    baseline_oos_mae = float(overall["V5_frozen_control"]["mae"])
    player_oos_mae = float(player_oos["overall"]["mae"])
    player_ci = player_oos["paired_date_cluster_bootstrap"]["95pct_date_cluster_ci"]
    validation_games = int(player_oos["overall"]["games"])
    validation_dates = int(player_oos["paired_date_cluster_bootstrap"]["dates"])
    validation_gate_passed = bool(
        validation_games >= 500
        and validation_dates >= 100
        and player_oos_mae < baseline_oos_mae
        and player_ci[1] < 0.0
    )
    calibration_path = ROOT / "research/live_player_strength_calibration.json"
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_payload["validation"] = {
        "method": "expanding-window prequential RAPM/injury challenger vs frozen V5",
        "games": validation_games,
        "dates": validation_dates,
        "v5_baseline_mae": baseline_oos_mae,
        "player_rapm_injury_mae": player_oos_mae,
        "mae_change_player_minus_v5": float(player_oos_mae - baseline_oos_mae),
        "date_cluster_ci_95": [float(player_ci[0]), float(player_ci[1])],
        "gate_minimum_games": 500,
        "gate_minimum_dates": 100,
        "gate_passed": validation_gate_passed,
        "interpretation": "Forecast use requires lower MAE and a date-cluster 95% CI wholly below zero.",
    }
    calibration_payload["forecast_enabled"] = validation_gate_passed
    calibration_payload["status"] = (
        "validated_challenger" if validation_gate_passed else "research_shadow_only"
    )
    calibration_path.write_text(
        json.dumps(calibration_payload, indent=2), encoding="utf-8"
    )
    live_calibration["validation"] = calibration_payload["validation"]
    live_calibration["forecast_enabled"] = validation_gate_passed
    live_calibration["status"] = calibration_payload["status"]

    by_season={}
    for season,part in result.groupby("season",sort=True):
        by_season[str(int(season))]={"V5":metric(part.actual_total,part.v5_total)}
        for family in FAMILIES:
            by_season[str(int(season))][family]=metric(part.actual_total,part[f"{family}_total"])

    OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/"stage_c_v5_prequential_predictions.csv",index=False)
    summary={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "experiment":"Stage C player intelligence residual challenger vs frozen V5",
        "control":"Frozen V5 = 0.55*V3 + 0.45*V4-90; raw total, no market inputs",
        "test_seasons_end_year":TEST_SEASONS,
        "training_policy":"Expanding prequential; strictly earlier game dates only; same-date games cannot train each other",
        "residual_target":"V5 prediction minus actual total; challenger subtracts predicted V5 error from V5",
        "model":f"constant-zero imputation + standardization + Ridge(alpha={RIDGE_ALPHA})",
        "minimum_prior_training_games":MIN_TRAIN_GAMES,
        "feature_families":FAMILIES,
        "common_sample_games":int(len(result)),
        "common_sample_dates":int(result.game_date.nunique()),
        "overall":overall,
        "candidate_results":candidate_results,
        "live_player_strength_calibration":live_calibration,
        "by_season":by_season,
        "point_in_time_checks":{
            "v5_history_cutoff_strictly_prior":True,
            "residual_training_strictly_prior":True,
            "profile_source_date_violations":profile_diag["profile_source_date_violations"],
            "profile_history_date_violations":profile_diag["profile_history_date_violations"],
            "rapm_snapshot_date_violations":rapm_bad,
        },
        "coverage":{
            "stage_c_candidate_games":profile_diag["candidate_games"],
            "stage_c_common_feature_games":int(stage.game_id.nunique()),
            "profile_candidate_rows":profile_diag["candidate_player_rows"],
            "eligible_profile_rows":profile_diag["profiled_candidate_rows"],
        },
        "data_diagnostics":{**history_diag,**pair_diag},
        "limitations":[
            "Historical source tables are reconstructed current-vintage data, not archived as-of-game statistical vintages.",
            "Injury snapshots are only included where the existing Phase 2 process found complete pregame reports.",
            "This is score-total accuracy validation, not a sportsbook market-beating or betting ROI test.",
            "Player box-score profiles are descriptive and do not establish causal player impact.",
        ],
        "V5_modified":False,"production_modified":False,"paper_ledger_modified":False,
        "profitability_claim":False,
    }
    (OUT/"stage_c_v5_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
