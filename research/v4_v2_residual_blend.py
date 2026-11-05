#!/usr/bin/env python3
"""Chronological research evaluation of a V2 + V4 residual blend.

Candidate formula: blend = V2 + beta * (V4 - V2), with beta in [0, 1].
Fit beta on seasons 2020-2022, validate forward on 2023, refit on 2020-2023,
and evaluate once on frozen seasons 2024-2026. Research only; no production
model, application, stake, odds capture, or ledger is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("outputs/research/v4_paired_v2_vs_v4/paired_v2_vs_v4_predictions.csv")
OUT = Path("outputs/research/v4_v2_residual_blend")
TRAIN_SEASONS = (2020, 2021, 2022)
VALIDATION_SEASONS = (2023,)
DEVELOPMENT_SEASONS = (2020, 2021, 2022, 2023)
HOLDOUT_SEASONS = (2024, 2025, 2026)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261031

REQUIRED_COLUMNS = {
    "season", "game_id", "game_date", "actual_total",
    "v2_points_engine_total", "v4_pace_efficiency_total",
    "v2_max_history_date", "v4_max_history_date",
}


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Paired predictions are missing columns: {missing}")
    data = frame.copy()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    numeric = ["actual_total", "v2_points_engine_total", "v4_pace_efficiency_total"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["game_date"] + numeric].isna().any().any():
        raise ValueError("Paired predictions contain missing dates or numeric values.")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Paired predictions contain non-finite numeric values.")
    for column in ("v2_max_history_date", "v4_max_history_date"):
        dates = pd.to_datetime(data[column], errors="coerce")
        if dates.isna().any() or not (dates < data["game_date"]).all():
            raise ValueError(f"{column} is not strictly earlier than game_date.")
    if data.duplicated(["season", "game_id"]).any():
        raise ValueError("Paired predictions contain duplicate season/game_id rows.")
    return data.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fit_blend_weight(frame: pd.DataFrame) -> dict:
    """Fit beta in residual = beta * (V4 - V2), constrained to [0, 1]."""
    if frame.empty:
        raise ValueError("Cannot fit blend weight on an empty development set.")
    gap = (
        frame["v4_pace_efficiency_total"].to_numpy(dtype=float)
        - frame["v2_points_engine_total"].to_numpy(dtype=float)
    )
    residual = (
        frame["actual_total"].to_numpy(dtype=float)
        - frame["v2_points_engine_total"].to_numpy(dtype=float)
    )
    denominator = float(np.dot(gap, gap))
    if denominator <= 1e-12:
        raise ValueError("V4-V2 prediction gap has no variation in the fit set.")
    raw_beta = float(np.dot(gap, residual) / denominator)
    return {
        "games": int(len(frame)),
        "raw_beta": raw_beta,
        "beta": float(np.clip(raw_beta, 0.0, 1.0)),
        "constraint": "convex blend; beta clipped to [0, 1]",
    }


def apply_blend(frame: pd.DataFrame, beta: float) -> pd.DataFrame:
    result = frame.copy()
    result["blend_prediction"] = (
        result["v2_points_engine_total"]
        + float(beta) * (
            result["v4_pace_efficiency_total"]
            - result["v2_points_engine_total"]
        )
    )
    result["blend_error"] = result["blend_prediction"] - result["actual_total"]
    result["blend_absolute_error"] = result["blend_error"].abs()
    return result


def regression_metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    errors = (
        pd.to_numeric(prediction, errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    )
    absolute = np.abs(errors)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Metrics require finite paired actuals and predictions.")
    return {
        "games": int(len(errors)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_prediction_minus_actual": float(errors.mean()),
        "within_5_points_pct": float(np.mean(absolute <= 5.0) * 100.0),
        "within_10_points_pct": float(np.mean(absolute <= 10.0) * 100.0),
    }


def metric_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    predictors = {
        "Points Engine V2": "v2_points_engine_total",
        "V4 pace + opponent-adjusted efficiency": "v4_pace_efficiency_total",
        "V2 + V4 residual blend": "blend_prediction",
    }
    return pd.DataFrame([
        {"split": split, "predictor": name,
         **regression_metrics(frame["actual_total"], frame[column])}
        for name, column in predictors.items()
    ])


def date_cluster_bootstrap(
    frame: pd.DataFrame,
    candidate_error: str,
    baseline_error: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap resampling whole game dates, retaining each day's slate."""
    differences = (
        frame.assign(_difference=frame[candidate_error] - frame[baseline_error])
        .groupby("game_date", sort=True)["_difference"]
        .agg(["sum", "count"])
    )
    if len(differences) < 2:
        raise ValueError("At least two game dates are required for date-cluster bootstrap.")
    sums = differences["sum"].to_numpy(dtype=float)
    counts = differences["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    n_dates = len(differences)
    for index in range(reps):
        selected = rng.integers(0, n_dates, size=n_dates)
        estimates[index] = sums[selected].sum() / counts[selected].sum()
    observed = float(sums.sum() / counts.sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    row_differences = (
        frame[candidate_error].to_numpy(dtype=float)
        - frame[baseline_error].to_numpy(dtype=float)
    )
    return {
        "difference_definition": "candidate absolute error minus baseline absolute error; negative favours candidate",
        "cluster_unit": "game_date; all games on a sampled date travel together",
        "games": int(len(frame)),
        "game_dates": int(n_dates),
        "observed_mae_difference": observed,
        "bootstrap_replicates": int(reps),
        "bootstrap_seed": int(seed),
        "bootstrap_95pct_interval": [float(low), float(high)],
        "candidate_lower_error_games": int((row_differences < 0).sum()),
        "baseline_lower_error_games": int((row_differences > 0).sum()),
        "ties": int((row_differences == 0).sum()),
        "interpretation": (
            "date-cluster interval accounts for same-day slate dependence, "
            "but not all repeated-team dependence across dates"
        ),
    }


def run_evaluation(frame: pd.DataFrame) -> dict:
    data = validate_predictions(frame)
    train = data.loc[data["season"].isin(TRAIN_SEASONS)].copy()
    validation = data.loc[data["season"].isin(VALIDATION_SEASONS)].copy()
    development = data.loc[data["season"].isin(DEVELOPMENT_SEASONS)].copy()
    holdout = data.loc[data["season"].isin(HOLDOUT_SEASONS)].copy()
    if train.empty or validation.empty or development.empty or holdout.empty:
        raise ValueError("Train, validation, development-refit and frozen holdout must all be populated.")

    preliminary_fit = fit_blend_weight(train)
    validation_scored = apply_blend(validation, preliminary_fit["beta"])
    validation_metrics = metric_table(validation_scored, "validation_2023")

    final_fit = fit_blend_weight(development)
    holdout_scored = apply_blend(holdout, final_fit["beta"])
    holdout_scored["v2_absolute_error"] = (
        holdout_scored["v2_points_engine_total"] - holdout_scored["actual_total"]
    ).abs()
    holdout_scored["v4_absolute_error"] = (
        holdout_scored["v4_pace_efficiency_total"] - holdout_scored["actual_total"]
    ).abs()
    holdout_metrics = metric_table(holdout_scored, "frozen_holdout_2024_2026")

    by_season = []
    for season, part in holdout_scored.groupby("season", sort=True):
        by_season.append({
            "season": int(season),
            **{
                predictor: regression_metrics(part["actual_total"], part[column])
                for predictor, column in {
                    "v2": "v2_points_engine_total",
                    "v4": "v4_pace_efficiency_total",
                    "blend": "blend_prediction",
                }.items()
            },
        })
    return {
        "validated_input": data,
        "validation_scored": validation_scored,
        "holdout_scored": holdout_scored,
        "preliminary_fit": preliminary_fit,
        "final_fit": final_fit,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_by_season": pd.DataFrame(by_season),
        "bootstrap_vs_v2": date_cluster_bootstrap(
            holdout_scored, "blend_absolute_error", "v2_absolute_error"
        ),
        "bootstrap_vs_v4": date_cluster_bootstrap(
            holdout_scored, "blend_absolute_error", "v4_absolute_error",
            seed=BOOTSTRAP_SEED + 1,
        ),
    }


def build_report(result: dict) -> str:
    val = result["validation_metrics"].to_markdown(index=False, floatfmt=".4f")
    hold = result["holdout_metrics"].to_markdown(index=False, floatfmt=".4f")
    season = result["holdout_by_season"].to_markdown(index=False)
    fit1, fit2 = result["preliminary_fit"], result["final_fit"]
    b2, b4 = result["bootstrap_vs_v2"], result["bootstrap_vs_v4"]
    lines = [
        "# V2 + V4 residual-blend evaluation", "",
        "## Design fixed before the frozen holdout", "",
        "- Candidate: V2 + beta × (V4 − V2), with beta constrained to [0, 1].",
        "- Preliminary coefficient fit: regular seasons 2020–2022.",
        "- Forward validation: 2023, using the preliminary coefficient without refitting.",
        "- Final coefficient refit: all development seasons 2020–2023.",
        "- Frozen evaluation: regular seasons 2024–2026; no holdout outcomes used to fit beta.",
        "- The blend has one fitted parameter and no intercept; it can only interpolate between V2 and V4.",
        "- Paired forecasts are checked to ensure both historical cutoffs are strictly earlier than each target game.",
        "",
        "## Coefficient", "",
        f"- Preliminary beta (2020–2022, {fit1['games']:,} games): {fit1['beta']:.6f} (raw {fit1['raw_beta']:.6f}).",
        f"- Final beta (2020–2023, {fit2['games']:,} games): {fit2['beta']:.6f} (raw {fit2['raw_beta']:.6f}).",
        "",
        "## 2023 forward validation", "", val, "",
        "## Frozen holdout: 2024–2026", "", hold, "",
        "## Frozen holdout by season", "", season, "",
        "## Paired date-cluster bootstrap", "",
        f"- Blend minus V2 MAE: {b2['observed_mae_difference']:.4f}; 95% interval [{b2['bootstrap_95pct_interval'][0]:.4f}, {b2['bootstrap_95pct_interval'][1]:.4f}].",
        f"- Blend minus V4 MAE: {b4['observed_mae_difference']:.4f}; 95% interval [{b4['bootstrap_95pct_interval'][0]:.4f}, {b4['bootstrap_95pct_interval'][1]:.4f}].",
        f"- Bootstrap resamples whole game dates ({b2['game_dates']:,} clusters); this handles same-day slate dependence but not all repeated-team dependence across dates.",
        "",
        "## Interpretation guardrails", "",
        "- This is score-prediction accuracy research, not betting-profitability evidence.",
        "- A lower holdout point MAE is not enough to promote the blend if paired uncertainty spans zero or season results are unstable.",
        "- No production V2 code, thresholds, stakes, odds capture, live bets, or ledgers are changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Required paired backtest input is missing: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_evaluation(pd.read_csv(INPUT))
    result["validation_metrics"].to_csv(OUT / "validation_metrics.csv", index=False)
    result["holdout_metrics"].to_csv(OUT / "holdout_metrics.csv", index=False)
    result["holdout_by_season"].to_csv(OUT / "holdout_by_season.csv", index=False)
    result["validation_scored"].to_csv(OUT / "validation_predictions.csv", index=False)
    result["holdout_scored"].to_csv(OUT / "holdout_predictions.csv", index=False)
    summary = {
        "train_seasons": list(TRAIN_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS),
        "development_refit_seasons": list(DEVELOPMENT_SEASONS),
        "frozen_holdout_seasons": list(HOLDOUT_SEASONS),
        "input_rows": int(len(result["validated_input"])),
        "input_max_game_date": result["validated_input"]["game_date"].max().date().isoformat(),
        "preliminary_fit": result["preliminary_fit"],
        "final_fit": result["final_fit"],
        "bootstrap_vs_v2": result["bootstrap_vs_v2"],
        "bootstrap_vs_v4": result["bootstrap_vs_v4"],
        "v2_history_strictly_prior": True,
        "v4_history_strictly_prior": True,
    }
    (OUT / "blend_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = build_report(result)
    (OUT / "v4_v2_residual_blend_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nDiagnostics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
