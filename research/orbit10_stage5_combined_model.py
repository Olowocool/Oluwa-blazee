#!/usr/bin/env python3
"""ORBIT-10 Stage 5: one-pattern combined residual model; final holdout sealed."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/research/orbit10_stage5")
BASELINE = Path("outputs/research/orbit10_stage2/orbit10_baseline_predictions.csv")
FEATURE_SCRIPT = Path("research/orbit10_stage3b_pattern1_era_normalized.py")
HOME_Z_THRESHOLD = 0.657915
AWAY_Z_THRESHOLD = 0.732879
MIN_GROUP = 100
BOOTSTRAPS = 2000
SEED = 20261114

def load_feature_module():
    spec = importlib.util.spec_from_file_location("orbit10_stage3b", FEATURE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def metric_frame(frame, prediction_col):
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

def bootstrap_coefficients(control, qualifying):
    rng = np.random.default_rng(SEED)
    alpha = np.empty(BOOTSTRAPS, dtype=float)
    beta = np.empty(BOOTSTRAPS, dtype=float)
    control = np.asarray(control, dtype=float)
    qualifying = np.asarray(qualifying, dtype=float)
    for i in range(BOOTSTRAPS):
        c = rng.choice(control, size=len(control), replace=True)
        q = rng.choice(qualifying, size=len(qualifying), replace=True)
        alpha[i] = c.mean()
        beta[i] = q.mean() - c.mean()
    return {
        "alpha_95ci": [round(float(np.quantile(alpha, .025)), 4), round(float(np.quantile(alpha, .975)), 4)],
        "beta_95ci": [round(float(np.quantile(beta, .025)), 4), round(float(np.quantile(beta, .975)), 4)]
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASELINE.is_file():
        raise FileNotFoundError("Frozen Stage 2 baseline predictions are missing.")
    baseline = pd.read_csv(BASELINE, parse_dates=["game_date"])
    required = {"game_id", "game_date", "season_end_year", "split", "predicted_total", "actual_total"}
    missing = required - set(baseline.columns)
    if missing:
        raise RuntimeError(f"Baseline is missing required columns: {sorted(missing)}")
    # Critical boundary: no holdout rows enter the feature/prediction analysis join.
    baseline = baseline[baseline.split.isin(["train", "validation"])].copy()
    if set(baseline.split.unique()) != {"train", "validation"}:
        raise RuntimeError("Expected both TRAIN and VALIDATION baseline rows.")
    if baseline.game_id.duplicated().any():
        raise RuntimeError("Duplicate baseline game IDs.")

    feature_mod = load_feature_module()
    spine = feature_mod.load_spine()
    features = feature_mod.build_features(spine)
    df_all = baseline.merge(features[["game_id", "game_date", "home_ppg_z", "away_def_z"]],
                            on=["game_id", "game_date"], how="left", validate="one_to_one")
    complete = df_all[["home_ppg_z", "away_def_z"]].notna().all(axis=1)
    coverage = {}
    for split in ("train", "validation"):
        part = df_all[df_all.split == split]
        n_complete = int((part[["home_ppg_z", "away_def_z"]].notna().all(axis=1)).sum())
        coverage[split] = {
            "baseline_games": int(len(part)),
            "eligible_games": n_complete,
            "excluded_missing_normalized_feature": int(len(part) - n_complete),
            "coverage_rate": round(n_complete / len(part), 6) if len(part) else None
        }
    df = df_all[complete].copy()
    df["pattern1"] = ((df.home_ppg_z >= HOME_Z_THRESHOLD) &
                      (df.away_def_z >= AWAY_Z_THRESHOLD)).astype(int)
    df["baseline_residual"] = df.predicted_total - df.actual_total

    train = df[df.split == "train"].copy()
    valid = df[df.split == "validation"].copy()
    control = train.loc[train.pattern1 == 0, "baseline_residual"].to_numpy(float)
    qualifying = train.loc[train.pattern1 == 1, "baseline_residual"].to_numpy(float)
    if min(len(control), len(qualifying)) < MIN_GROUP:
        raise RuntimeError(f"TRAIN minimum group count failed: control={len(control)}, qualifying={len(qualifying)}")

    # OLS residual model with intercept + one frozen binary indicator:
    # alpha is control-group mean residual; beta is qualifying-control difference.
    alpha = float(control.mean())
    beta = float(qualifying.mean() - control.mean())
    ci = bootstrap_coefficients(control, qualifying)
    df["global_calibration_contribution"] = -alpha
    df["pattern1_contribution"] = -beta * df.pattern1.astype(float)
    df["total_model_adjustment"] = df.global_calibration_contribution + df.pattern1_contribution
    df["model_b_prediction"] = df.predicted_total + df.total_model_adjustment
    df["model_a_prediction"] = df.predicted_total
    df["feature_contribution_explanation"] = np.where(
        df.pattern1.eq(1),
        "Global calibration plus era-normalized Pattern 1",
        "Global calibration only"
    )

    valid_control = valid.loc[valid.pattern1 == 0, "baseline_residual"]
    valid_qualifying = valid.loc[valid.pattern1 == 1, "baseline_residual"]
    if min(len(valid_control), len(valid_qualifying)) < MIN_GROUP:
        raise RuntimeError(f"VALIDATION minimum group count failed: control={len(valid_control)}, qualifying={len(valid_qualifying)}")
    valid_effect = float(valid_qualifying.mean() - valid_control.mean())

    # Score both models on exactly the same eligible validation games.
    valid_scored = df[df.split == "validation"].copy()
    model_a = metric_frame(valid_scored, "model_a_prediction")
    model_b = metric_frame(valid_scored, "model_b_prediction")
    season_metrics = {}
    for year in sorted(valid_scored.season_end_year.astype(int).unique()):
        part = valid_scored[valid_scored.season_end_year.astype(int) == year]
        season_metrics[str(year)] = {
            "model_a": metric_frame(part, "model_a_prediction"),
            "model_b": metric_frame(part, "model_b_prediction"),
            "mae_change_model_b_minus_a": round(metric_frame(part, "model_b_prediction")["mae"] -
                                                metric_frame(part, "model_a_prediction")["mae"], 4),
            "rmse_change_model_b_minus_a": round(metric_frame(part, "model_b_prediction")["rmse"] -
                                                 metric_frame(part, "model_a_prediction")["rmse"], 4)
        }

    criteria = {
        "train_groups_at_least_100": bool(len(control) >= MIN_GROUP and len(qualifying) >= MIN_GROUP),
        "train_beta_bootstrap_interval_excludes_zero": bool(ci["beta_95ci"][0] > 0 or ci["beta_95ci"][1] < 0),
        "validation_groups_at_least_100": bool(len(valid_control) >= MIN_GROUP and len(valid_qualifying) >= MIN_GROUP),
        "validation_effect_same_sign_as_train_beta": bool(np.sign(valid_effect) == np.sign(beta)),
        "validation_mae_improves": bool(model_b["mae"] < model_a["mae"]),
        "validation_rmse_improves": bool(model_b["rmse"] < model_a["rmse"]),
        "identical_validation_game_ids": bool(len(valid_scored) == len(valid_scored.game_id.unique()))
    }
    summary = {
        "stage": "ORBIT-10 Stage 5 — parsimonious combined model",
        "models": {
            "model_a": "Frozen Stage 2 baseline: team strength + projected possessions.",
            "model_b": "Model A plus TRAIN-only residual OLS intercept and one binary era-normalized Pattern 1 feature."
        },
        "frozen_pattern1_thresholds": {
            "home_ppg_z": HOME_Z_THRESHOLD,
            "away_road_defensive_rating_z": AWAY_Z_THRESHOLD
        },
        "coverage": coverage,
        "training_fit": {
            "games": int(len(train)), "control_games": int(len(control)),
            "qualifying_games": int(len(qualifying)),
            "alpha_control_mean_residual": round(alpha, 6),
            "beta_pattern1_incremental_residual_effect": round(beta, 6),
            "global_calibration_prediction_contribution": round(-alpha, 6),
            "pattern1_prediction_contribution_when_qualifying": round(-beta, 6),
            **ci,
            "bootstrap_replicates": BOOTSTRAPS, "seed": SEED
        },
        "validation": {
            "games": int(len(valid_scored)),
            "control_games": int(len(valid_control)),
            "qualifying_games": int(len(valid_qualifying)),
            "observed_pattern1_minus_control_residual_effect": round(valid_effect, 6),
            "model_a_metrics": model_a,
            "model_b_metrics": model_b,
            "mae_change_model_b_minus_a": round(model_b["mae"] - model_a["mae"], 4),
            "rmse_change_model_b_minus_a": round(model_b["rmse"] - model_a["rmse"], 4),
            "season_metrics": season_metrics
        },
        "decision": {
            "criteria": criteria,
            "passes_all": all(criteria.values()),
            "status": "STAGE 5 SCREENING PASS — eligible for Stage 6 chronological replication only" if all(criteria.values())
                      else "DO NOT RETAIN AS COMBINED CANDIDATE — preregistered Stage 5 criteria not met"
        },
        "protocol": {
            "fit": "TRAIN only; residual = baseline prediction minus actual; alpha=mean control residual; beta=mean qualifying residual minus mean control residual.",
            "prediction": "Model B = Model A - (alpha + beta * Pattern1).",
            "missingness": "Complete cases only; report split coverage; no imputation.",
            "holdout": "SEALED — filtered out before feature merge; no holdout predictions, labels, or metrics used.",
            "market_data": "Not loaded or used.",
            "interactions": "None; Stage 4 retained no interactions.",
            "same_day": "Features constructed using only games completed before target date; all same-day games featured before same-day results enter history.",
            "tuning": "No validation tuning, threshold search, hyperparameter search, or model-family selection."
        },
        "interpretation": "This is an observational validation screen, not causal evidence, profitability proof, or deployment approval."
    }
    (OUT / "orbit10_stage5_combined_summary.json").write_text(json.dumps(summary, indent=2))
    export_cols = ["game_id", "game_date", "season_end_year", "split", "actual_total",
                   "model_a_prediction", "model_b_prediction", "home_ppg_z", "away_def_z",
                   "pattern1", "global_calibration_contribution", "pattern1_contribution",
                   "total_model_adjustment", "feature_contribution_explanation"]
    df[export_cols].to_csv(OUT / "orbit10_stage5_combined_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
