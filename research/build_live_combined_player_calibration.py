#!/usr/bin/env python3
"""Build a combined player calibration as a research-only shadow challenger."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import stage_c_v5_residual_experiment as stage_c
from research.prospective_player_profiles import PROFILE_FEATURES as LIVE_PROFILE_FEATURES


def main() -> None:
    frame_path = stage_c.OUT / "combined_player_calibration_training.parquet"
    stage_c.OUT.mkdir(parents=True, exist_ok=True)
    if list(stage_c.PROFILE_FEATURES) != list(LIVE_PROFILE_FEATURES):
        raise RuntimeError("Live player-profile feature contract differs from Stage C training features.")
    original_export = stage_c.export_live_player_strength_calibration

    def capture_frame(data: pd.DataFrame) -> dict:
        data.to_parquet(frame_path, index=False)
        return original_export(data)

    stage_c.export_live_player_strength_calibration = capture_frame
    stage_c.main()

    data = pd.read_parquet(frame_path)
    summary_path = stage_c.OUT / "stage_c_v5_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features = list(stage_c.FAMILIES["combined"])
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise RuntimeError(f"Combined feature columns missing: {missing}")

    cutoff = pd.Timestamp("2026-07-01")
    train = data.loc[pd.to_datetime(data.game_date).dt.normalize() < cutoff].copy()
    train = train.dropna(subset=["v5_total", "actual_total"])
    if len(train) < stage_c.MIN_TRAIN_GAMES:
        raise RuntimeError("Insufficient pre-2026-27 combined calibration rows.")
    if train.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate game keys in combined calibration training data.")

    rapm_columns = list(stage_c.RAPM_INJURY_FEATURES[:4])
    rapm_values = train[rapm_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rapm_nonzero = {
        name: int(rapm_values[name].abs().gt(1e-9).sum())
        for name in rapm_columns
    }
    if any(value <= 0 for value in rapm_nonzero.values()):
        raise RuntimeError(f"RAPM training coverage is incomplete: {rapm_nonzero}")

    model = stage_c.pipeline()
    model.fit(train[features], train.v5_total.astype(float) - train.actual_total.astype(float))
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]

    candidate = summary["candidate_results"]["combined"]
    baseline = summary["overall"]["V5_frozen_control"]
    candidate_mae = float(candidate["overall"]["mae"])
    baseline_mae = float(baseline["mae"])
    ci = [float(x) for x in candidate["paired_date_cluster_bootstrap"]["95pct_date_cluster_ci"]]
    games = int(candidate["overall"]["games"])
    dates = int(candidate["paired_date_cluster_bootstrap"]["dates"])
    historical_gate_passed = bool(
        games >= 500 and dates >= 100 and candidate_mae < baseline_mae and ci[1] < 0.0
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
        "feature_family": "Stage C RAPM + injury + projected-minute player skill profiles",
        "features": features,
        "rapm_feature_nonzero_rows": rapm_nonzero,
        "imputer": {
            "strategy": "constant",
            "fill_value": 0.0,
            "statistics": [float(x) for x in imputer.statistics_],
        },
        "standardizer": {
            "mean": [float(x) for x in scaler.mean_],
            "scale": [float(x) for x in scaler.scale_],
        },
        "ridge": {
            "alpha": float(stage_c.RIDGE_ALPHA),
            "intercept": float(ridge.intercept_),
            "coefficients": [float(x) for x in ridge.coef_],
        },
        "target": "frozen_v5_total_minus_actual_total",
        "prediction_formula": "player_adjusted_total = frozen_v5_total - predicted_v5_error",
        "point_in_time_policy": "training dates strictly before target; player profile history and RAPM history strictly before target",
        "validation": {
            "method": "expanding-window prequential combined RAPM/injury + player-profile challenger vs frozen V5",
            "challenger_family": "combined",
            "games": games,
            "dates": dates,
            "v5_baseline_mae": baseline_mae,
            "player_combined_mae": candidate_mae,
            "mae_change_player_minus_v5": candidate_mae - baseline_mae,
            "date_cluster_ci_95": ci,
            "gate_minimum_games": 500,
            "gate_minimum_dates": 100,
            "gate_passed": historical_gate_passed,
            "interpretation": "Historical accuracy gate only; source-vintage limitations keep this as a research shadow estimate.",
        },
        "forecast_enabled": False,
        "forecast_eligibility_reason": "Historical source vintages are not fully archived as-of-game; prospective evidence is required before activation.",
        "betting_enabled": False,
        "probabilities_calibrated": False,
        "profitability_claim": False,
        "v5_modified": False,
        "production_modified": False,
        "paper_ledger_modified": False,
        "point_in_time_checks": summary.get("point_in_time_checks", {}),
        "limitations": summary.get("limitations", []),
    }
    dimensions = [
        payload["imputer"]["statistics"], payload["standardizer"]["mean"],
        payload["standardizer"]["scale"], payload["ridge"]["coefficients"],
    ]
    if any(len(x) != len(features) or not np.isfinite(x).all() for x in dimensions):
        raise RuntimeError("Combined player calibration has invalid feature dimensions.")
    if any(x <= 0 for x in payload["standardizer"]["scale"]):
        raise RuntimeError("Combined player calibration has invalid standardizer scales.")

    artifact = ROOT / "research" / "live_player_strength_calibration.json"
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary["live_player_strength_calibration"] = {
        "path": str(artifact),
        "training_games": payload["training_games"],
        "training_first_date": payload["training_first_date"],
        "training_last_date": payload["training_last_date"],
        "feature_count": len(features),
        "feature_family": payload["feature_family"],
        "status": payload["status"],
        "forecast_enabled": False,
        "validation": payload["validation"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    frame_path.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
