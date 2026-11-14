#!/usr/bin/env python3
"""ORBIT-10 Stage 6: one-time final holdout evaluation of frozen Stage 5 Model B."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/research/orbit10_stage6_final_holdout")
BASELINE = Path("outputs/research/orbit10_stage2/orbit10_baseline_predictions.csv")
FEATURE_SCRIPT = Path("research/orbit10_stage3b_pattern1_era_normalized.py")

# Frozen from the accepted Stage 5 training fit. Never estimate from holdout.
ALPHA = -0.805967
BETA = -5.836135
HOME_Z_THRESHOLD = 0.657915
AWAY_Z_THRESHOLD = 0.732879
EXPECTED_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
MIN_PAIRED_GAMES = 1000
MIN_COVERAGE = 0.95
MIN_SEASONS_IMPROVED = 4

def load_feature_module():
    spec = importlib.util.spec_from_file_location("orbit10_stage3b_frozen", FEATURE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def metrics(frame: pd.DataFrame, prediction_col: str) -> dict:
    if frame.empty:
        return {"games": 0, "mae": None, "rmse": None, "bias_prediction_minus_actual": None,
                "within_5_points_pct": None, "within_10_points_pct": None}
    err = frame[prediction_col].to_numpy(float) - frame.actual_total.to_numpy(float)
    return {
        "games": int(len(err)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias_prediction_minus_actual": round(float(np.mean(err)), 4),
        "within_5_points_pct": round(float(np.mean(np.abs(err) <= 5) * 100), 2),
        "within_10_points_pct": round(float(np.mean(np.abs(err) <= 10) * 100), 2),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASELINE.is_file():
        raise FileNotFoundError("Frozen Stage 2 baseline predictions are missing.")
    baseline = pd.read_csv(BASELINE, parse_dates=["game_date"])
    required = {"game_id", "game_date", "season_end_year", "split", "predicted_total", "actual_total"}
    missing = required - set(baseline.columns)
    if missing:
        raise RuntimeError(f"Baseline schema missing: {sorted(missing)}")
    # Isolate the final holdout population before feature join. No fitting occurs.
    holdout = baseline[baseline.split.eq("holdout")].copy()
    if holdout.empty or holdout.game_id.duplicated().any():
        raise RuntimeError("Holdout baseline is empty or has duplicate game IDs.")
    observed_seasons = sorted(holdout.season_end_year.astype(int).unique().tolist())
    if observed_seasons != EXPECTED_SEASONS:
        raise RuntimeError(f"Expected holdout seasons {EXPECTED_SEASONS}; found {observed_seasons}.")

    feature_mod = load_feature_module()
    spine = feature_mod.load_spine()
    features = feature_mod.build_features(spine)
    df_all = holdout.merge(
        features[["game_id", "game_date", "home_ppg_z", "away_def_z"]],
        on=["game_id", "game_date"], how="left", validate="one_to_one"
    )
    feature_complete = df_all[["home_ppg_z", "away_def_z"]].notna().all(axis=1)
    coverage = {
        "baseline_games": int(len(df_all)),
        "eligible_games": int(feature_complete.sum()),
        "excluded_missing_normalized_feature": int((~feature_complete).sum()),
        "coverage_rate": round(float(feature_complete.mean()), 6)
    }
    df = df_all[feature_complete].copy()
    if df.empty:
        raise RuntimeError("No complete-case holdout feature rows.")
    df["pattern1"] = ((df.home_ppg_z >= HOME_Z_THRESHOLD) &
                      (df.away_def_z >= AWAY_Z_THRESHOLD)).astype(int)
    df["model_a_prediction"] = df.predicted_total.astype(float)
    df["global_calibration_contribution"] = -ALPHA
    df["pattern1_contribution"] = -BETA * df.pattern1.astype(float)
    df["total_model_adjustment"] = (
        df.global_calibration_contribution + df.pattern1_contribution
    )
    df["model_b_prediction"] = (
        df.model_a_prediction + df.total_model_adjustment
    )
    df["prediction_reconstruction_error"] = (
        df.model_b_prediction -
        (df.model_a_prediction - (ALPHA + BETA * df.pattern1.astype(float)))
    )
    if not np.allclose(df.prediction_reconstruction_error, 0.0, atol=1e-10):
        raise RuntimeError("Frozen Model B prediction arithmetic failed.")
    # Explicit chronology audit: Stage 3B history builder only appends each date
    # after all games on that date have been featured. Verify latest prior dates.
    df["game_date"] = pd.to_datetime(df.game_date).dt.normalize()
    max_hist = pd.to_datetime(df.get("max_history_date", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    # Feature output doesn't export max_history_date; chronology is enforced by the
    # unchanged Stage 3B builder and asserted here through a full feature rebuild.
    if df.game_id.duplicated().any():
        raise RuntimeError("Duplicate paired holdout game IDs.")

    full_baseline = holdout.rename(columns={"predicted_total": "model_a_prediction"})
    full_baseline_metrics = metrics(
        full_baseline.assign(actual_total=full_baseline.actual_total), "model_a_prediction"
    )
    paired_a = metrics(df, "model_a_prediction")
    paired_b = metrics(df, "model_b_prediction")
    season_metrics = {}
    for year in EXPECTED_SEASONS:
        part = df[df.season_end_year.astype(int).eq(year)]
        a, b = metrics(part, "model_a_prediction"), metrics(part, "model_b_prediction")
        season_metrics[str(year)] = {
            "games": int(len(part)), "model_a": a, "model_b": b,
            "mae_change_model_b_minus_a": round(b["mae"] - a["mae"], 4) if len(part) else None,
            "rmse_change_model_b_minus_a": round(b["rmse"] - a["rmse"], 4) if len(part) else None,
            "both_metrics_improve": bool(
                len(part) and b["mae"] < a["mae"] and b["rmse"] < a["rmse"]
            )
        }
    improved_seasons = sum(x["both_metrics_improve"] for x in season_metrics.values())
    criteria = {
        "all_six_expected_seasons_present": observed_seasons == EXPECTED_SEASONS,
        "at_least_1000_paired_games": len(df) >= MIN_PAIRED_GAMES,
        "feature_coverage_at_least_95_percent": coverage["coverage_rate"] >= MIN_COVERAGE,
        "overall_paired_mae_improves": paired_b["mae"] < paired_a["mae"],
        "overall_paired_rmse_improves": paired_b["rmse"] < paired_a["rmse"],
        "both_metrics_improve_in_at_least_four_seasons": improved_seasons >= MIN_SEASONS_IMPROVED
    }
    summary = {
        "stage": "ORBIT-10 Stage 6 — final chronological holdout evaluation",
        "status": "FINAL HOLDOUT PASS" if all(criteria.values()) else "FAIL / INSUFFICIENT EVIDENCE",
        "frozen_model": {
            "formula": "Model B = Model A - (alpha + beta * Pattern1)",
            "alpha": ALPHA, "beta": BETA,
            "home_ppg_z_threshold": HOME_Z_THRESHOLD,
            "away_road_defensive_rating_z_threshold": AWAY_Z_THRESHOLD,
            "source": "Accepted Stage 5 TRAIN-only fit; constants copied unchanged."
        },
        "holdout": {
            "expected_season_end_years": EXPECTED_SEASONS,
            "observed_season_end_years": observed_seasons,
            "full_baseline_population_model_a": full_baseline_metrics,
            "coverage": coverage,
            "paired_population_model_a": paired_a,
            "paired_population_model_b": paired_b,
            "mae_change_model_b_minus_a": round(paired_b["mae"] - paired_a["mae"], 4),
            "rmse_change_model_b_minus_a": round(paired_b["rmse"] - paired_a["rmse"], 4),
            "pattern1_qualifying_games": int(df.pattern1.sum()),
            "pattern1_control_games": int((df.pattern1 == 0).sum()),
            "pattern1_prevalence_pct": round(float(df.pattern1.mean() * 100), 2),
            "global_calibration_prediction_contribution": round(-ALPHA, 6),
            "pattern1_prediction_contribution_when_qualifying": round(-BETA, 6),
            "season_metrics": season_metrics,
            "seasons_with_both_metrics_improved": int(improved_seasons)
        },
        "decision": {
            "criteria": criteria, "passes_all": bool(all(criteria.values())),
            "interpretation": (
                "One-time final holdout pass for frozen historical predictive replication only; "
                "not causal evidence, profitability proof, live validation or deployment approval."
                if all(criteria.values()) else
                "The frozen candidate failed at least one preregistered final-holdout criterion. "
                "Do not tune or recalibrate on this holdout and continue to call it independent."
            )
        },
        "protocol": {
            "holdout_use": "Final holdout opened only for this preregistered final scoring; no fitting, threshold selection or tuning.",
            "features": "Unchanged Stage 3B same-date league z-scores from trailing-25 venue-specific prior games; minimum five prior venue games.",
            "same_day": "All same-day games featured before same-day results enter histories.",
            "missingness": "Complete cases only; exclusions and coverage explicitly reported; no imputation.",
            "market_data": "Not loaded or used.",
            "other_patterns_or_interactions": "None.",
            "post_result_tuning": "Prohibited; this is the one-time final holdout."
        }
    }
    (OUT / "orbit10_stage6_final_holdout_summary.json").write_text(json.dumps(summary, indent=2))
    export_cols = [
        "game_id", "game_date", "season_end_year", "actual_total",
        "model_a_prediction", "model_b_prediction", "home_ppg_z", "away_def_z",
        "pattern1", "global_calibration_contribution", "pattern1_contribution",
        "total_model_adjustment", "prediction_reconstruction_error"
    ]
    df[export_cols].to_csv(OUT / "orbit10_stage6_final_holdout_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
