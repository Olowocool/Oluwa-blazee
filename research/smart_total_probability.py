#!/usr/bin/env python3
"""Research-only chronological NBA total model competition and probability calibration.

Builds on the point-in-time NBA Stats team-game features in
research/advanced_totals_experiment.py. It never changes production models.

Protocol:
- Train only before 2024-10-01.
- Select model/hyperparameters on 2024-25 regular-season validation.
- Freeze the selected model and validation residual distribution.
- Evaluate once on the 2025-26 and later regular-season holdout.
- Compare a structural formula, Ridge, ExtraTrees, and HistGradientBoosting.
- Produce empirical over probabilities and conformal prediction intervals.
- Historical market archive comparison is explicitly exploratory because it
  has already been examined in prior work and is not an untouched holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.advanced_totals_experiment import (
    ARCHIVE,
    FEATURES,
    build_features,
    connect,
    load_data,
    make_team_lookup,
    team_key,
)

TRAIN_END = pd.Timestamp("2024-10-01")
VALIDATION_END = pd.Timestamp("2025-10-01")
HOLDOUT_START = VALIDATION_END
RANDOM_STATE = 42
QUANTILES = (0.10, 0.90)


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    values = out[columns].to_numpy(dtype=float)
    return out.loc[np.isfinite(values).all(axis=1)].copy()


def chronological_splits(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"game_date", "game_type", "actual_total_source", *FEATURES}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature frame missing required columns: {sorted(missing)}")
    frame = features.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["game_type"].eq("regular_season")].copy()
    frame = _finite_frame(frame, FEATURES + ["actual_total_source"])
    train = frame.loc[frame.game_date < TRAIN_END].copy()
    validation = frame.loc[
        (frame.game_date >= TRAIN_END) & (frame.game_date < VALIDATION_END)
    ].copy()
    holdout = frame.loc[frame.game_date >= HOLDOUT_START].copy()
    if min(len(train), len(validation), len(holdout)) == 0:
        raise ValueError(
            "Chronological split is incomplete: "
            f"train={len(train)}, validation={len(validation)}, holdout={len(holdout)}. "
            "Do not silently move the dates; inspect source coverage."
        )
    if not train.game_date.max() < validation.game_date.min():
        raise AssertionError("Training and validation dates overlap.")
    if not validation.game_date.max() < holdout.game_date.min():
        raise AssertionError("Validation and holdout dates overlap.")
    return train, validation, holdout


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        raise ValueError("No finite observations available for metrics.")
    error = p - y
    absolute = np.abs(error)
    return {
        "games": int(len(y)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_prediction_minus_actual": float(error.mean()),
        "within_5_points_pct": float(np.mean(absolute <= 5.0) * 100.0),
        "within_10_points_pct": float(np.mean(absolute <= 10.0) * 100.0),
    }


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=float(alpha))),
    ])


def choose_models(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train, y_train = train[FEATURES], train["actual_total_source"].astype(float)
    x_val, y_val = validation[FEATURES], validation["actual_total_source"].astype(float)
    candidates: dict[str, Any] = {}
    validation_scores: dict[str, Any] = {}

    # Ridge regularization is selected only on the validation season.
    ridge_trials = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_ridge(alpha)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        score = regression_metrics(y_val, prediction)["mae"]
        ridge_trials.append({"alpha": alpha, "validation_mae": score})
    best_ridge = min(ridge_trials, key=lambda row: (row["validation_mae"], row["alpha"]))
    ridge = make_ridge(best_ridge["alpha"]).fit(x_train, y_train)
    candidates["ridge"] = ridge
    validation_scores["ridge"] = {
        "selected_alpha": best_ridge["alpha"],
        "trials": ridge_trials,
        "metrics": regression_metrics(y_val, ridge.predict(x_val)),
    }

    # Small, predeclared HGB grid. No random train/validation split or early stopping.
    hgb_trials = []
    for leaves in (7, 15, 31):
        for learning_rate in (0.03, 0.06):
            params = {
                "loss": "squared_error",
                "learning_rate": learning_rate,
                "max_iter": 220,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": 20,
                "l2_regularization": 5.0,
                "early_stopping": False,
                "random_state": RANDOM_STATE,
            }
            model = HistGradientBoostingRegressor(**params)
            model.fit(x_train, y_train)
            prediction = model.predict(x_val)
            hgb_trials.append({
                "params": params,
                "validation_mae": regression_metrics(y_val, prediction)["mae"],
            })
    best_hgb = min(hgb_trials, key=lambda row: row["validation_mae"])
    hgb = HistGradientBoostingRegressor(**best_hgb["params"]).fit(x_train, y_train)
    candidates["hist_gradient_boosting"] = hgb
    validation_scores["hist_gradient_boosting"] = {
        "selected_params": best_hgb["params"],
        "trials": [
            {"params": row["params"], "validation_mae": row["validation_mae"]}
            for row in hgb_trials
        ],
        "metrics": regression_metrics(y_val, hgb.predict(x_val)),
    }

    extra = ExtraTreesRegressor(
        n_estimators=350,
        max_features=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(x_train, y_train)
    candidates["extra_trees"] = extra
    validation_scores["extra_trees"] = {
        "fixed_params": {
            "n_estimators": 350, "max_features": 0.8,
            "min_samples_leaf": 10, "random_state": RANDOM_STATE,
        },
        "metrics": regression_metrics(y_val, extra.predict(x_val)),
    }

    validation_scores["structural_formula"] = {
        "metrics": regression_metrics(y_val, validation["formula_total"])
    }
    return candidates, validation_scores


def empirical_over_probability(
    predicted_median: Any, market_line: Any, calibration_residuals: Any
) -> np.ndarray:
    """P(actual > line) from strictly out-of-sample calibration residuals.

    Plus-one smoothing avoids returning exact zero or one. The calibration
    residuals are actual minus predicted, so larger residuals favour the Over.
    """
    median = np.asarray(predicted_median, dtype=float).reshape(-1)
    line = np.asarray(market_line, dtype=float).reshape(-1)
    residuals = np.asarray(calibration_residuals, dtype=float).reshape(-1)
    if median.shape != line.shape or not len(residuals):
        raise ValueError("Prediction/line shapes must match and calibration cannot be empty.")
    if not np.isfinite(median).all() or not np.isfinite(line).all() or not np.isfinite(residuals).all():
        raise ValueError("Probability inputs must all be finite.")
    thresholds = line - median
    counts = np.asarray([(residuals > value).sum() for value in thresholds], dtype=float)
    return (counts + 1.0) / (len(residuals) + 2.0)


def probability_metrics(
    actual: Any, predicted_median: Any, offsets: tuple[float, ...],
    calibration_residuals: Any,
) -> dict[str, Any]:
    y = np.asarray(actual, dtype=float).reshape(-1)
    median = np.asarray(predicted_median, dtype=float).reshape(-1)
    if y.shape != median.shape:
        raise ValueError("Actual and predicted arrays must have equal shape.")
    rows = []
    for offset in offsets:
        # Synthetic half-point line offsets are used because a full historical
        # market archive does not exist for every feature row.
        line = median + float(offset)
        outcome = (y > line).astype(float)
        probability = empirical_over_probability(median, line, calibration_residuals)
        brier = float(np.mean((probability - outcome) ** 2))
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logloss = float(-np.mean(
            outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)
        ))
        rows.append({
            "line_offset_from_model_points": float(offset),
            "games": int(len(y)),
            "observed_over_rate_pct": float(outcome.mean() * 100.0),
            "mean_predicted_over_probability_pct": float(probability.mean() * 100.0),
            "brier_score": brier,
            "log_loss": logloss,
        })
    return {"synthetic_line_calibration": rows}


def conformal_interval(
    lower: Any, upper: Any, actual: Any, coverage: float = 0.80
) -> tuple[np.ndarray, float]:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(actual, dtype=float)
    if not (lo.shape == hi.shape == y.shape) or not 0 < coverage < 1:
        raise ValueError("Interval arrays must match and coverage must be between zero and one.")
    scores = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])
    correction = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))))
    return correction, correction


def fit_quantile_models(
    train: pd.DataFrame, validation: pd.DataFrame, holdout: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    val_predictions = {}
    for q in QUANTILES:
        model_params = dict(params)
        model_params["loss"] = "quantile"
        model_params["quantile"] = q
        model = HistGradientBoostingRegressor(**model_params)
        model.fit(train[FEATURES], train["actual_total_source"].astype(float))
        models[str(q)] = model
        val_predictions[str(q)] = model.predict(validation[FEATURES])
    correction, _ = conformal_interval(
        val_predictions["0.1"], val_predictions["0.9"],
        validation["actual_total_source"].to_numpy(dtype=float), 0.80,
    )
    holdout_lower = models["0.1"].predict(holdout[FEATURES]) - correction
    holdout_upper = models["0.9"].predict(holdout[FEATURES]) + correction
    holdout_actual = holdout["actual_total_source"].to_numpy(dtype=float)
    diagnostics = {
        "nominal_coverage": 0.80,
        "validation_conformal_correction_points": correction,
        "holdout_empirical_coverage_pct": float(np.mean(
            (holdout_actual >= holdout_lower) & (holdout_actual <= holdout_upper)
        ) * 100.0),
        "holdout_mean_interval_width": float(np.mean(holdout_upper - holdout_lower)),
        "holdout_lower_quantile_crossings": int(np.sum(holdout_lower > holdout_upper)),
    }
    return models, diagnostics


def market_archive_diagnostic(
    features: pd.DataFrame, teams: pd.DataFrame, model: Any,
    model_name: str, calibration_residuals: np.ndarray,
) -> dict[str, Any] | None:
    archive_path = Path(ARCHIVE)
    if not archive_path.exists():
        return None
    market = pd.read_csv(archive_path)
    required = {
        "game_date", "home_team", "away_team", "total_close", "actual_total"
    }
    if required.difference(market.columns):
        return {"status": "unavailable", "reason": "Market archive schema is incomplete."}
    lookup = make_team_lookup(teams)
    market["game_date"] = pd.to_datetime(market["game_date"], errors="coerce").dt.normalize()
    for column in ("total_close", "actual_total"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market["home_id"] = market["home_team"].map(lambda value: lookup.get(team_key(value)))
    market["away_id"] = market["away_team"].map(lambda value: lookup.get(team_key(value)))
    market = market.dropna(subset=["game_date", "home_id", "away_id", "total_close", "actual_total"]).copy()
    market["pair_key"] = market.apply(
        lambda row: (row.game_date, *sorted([str(row.home_id), str(row.away_id)])), axis=1
    )
    source = features.copy()
    source["pair_key"] = source.apply(
        lambda row: (pd.Timestamp(row.game_date).normalize(),
                     *sorted([str(row.team_a_id), str(row.team_b_id)])), axis=1
    )
    if source["pair_key"].duplicated().any():
        raise ValueError("Duplicate date/team-pair feature rows prevent market archive matching.")
    joined = market.merge(source, on="pair_key", how="inner", validate="one_to_one")
    if joined.empty:
        return {"status": "unavailable", "reason": "No market archive games matched features."}
    prediction = (
        joined["formula_total"].to_numpy(dtype=float)
        if model_name == "structural_formula"
        else model.predict(joined[FEATURES])
    )
    line = joined["total_close"].to_numpy(dtype=float)
    actual = joined["actual_total"].to_numpy(dtype=float)
    probability = empirical_over_probability(prediction, line, calibration_residuals)
    outcome = (actual > line).astype(float)
    return {
        "status": "exploratory_previously_examined_archive",
        "games": int(len(joined)),
        "candidate_total_metrics": regression_metrics(actual, prediction),
        "closing_market_metrics": regression_metrics(actual, line),
        "over_probability_brier": float(np.mean((probability - outcome) ** 2)),
        "over_probability_log_loss": float(-np.mean(
            outcome * np.log(np.clip(probability, 1e-6, 1-1e-6))
            + (1-outcome) * np.log(np.clip(1-probability, 1e-6, 1-1e-6))
        )),
        "mean_predicted_over_probability_pct": float(probability.mean() * 100.0),
        "observed_over_rate_pct": float(outcome.mean() * 100.0),
        "warning": "This archive has already been examined; it is not a new confirmatory holdout or proof of ROI.",
    }


def run(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    con, metadata = connect()
    try:
        raw, teams = load_data(con)
    finally:
        con.close()
    features = build_features(raw)
    if features.empty or not (pd.to_datetime(features.max_history_date) < pd.to_datetime(features.game_date)).all():
        raise RuntimeError("Point-in-time feature history check failed.")
    train, validation, holdout = chronological_splits(features)
    candidates, validation_scores = choose_models(train, validation)
    selected_name = min(
        validation_scores,
        key=lambda name: validation_scores[name]["metrics"]["mae"]
    )
    selected_model = candidates.get(selected_name)
    validation_prediction = (
        validation["formula_total"].to_numpy(dtype=float)
        if selected_name == "structural_formula"
        else selected_model.predict(validation[FEATURES])
    )
    calibration_residuals = (
        validation["actual_total_source"].to_numpy(dtype=float)
        - validation_prediction
    )
    if len(calibration_residuals) < 200:
        raise RuntimeError("Too few independent validation residuals for probability calibration.")

    holdout_actual = holdout["actual_total_source"].to_numpy(dtype=float)
    holdout_predictions = {
        "structural_formula": holdout["formula_total"].to_numpy(dtype=float),
        **{
            name: model.predict(holdout[FEATURES])
            for name, model in candidates.items()
        },
    }
    holdout_metrics = {
        "structural_formula": regression_metrics(holdout_actual, holdout["formula_total"]),
        **{
            name: regression_metrics(holdout_actual, prediction)
            for name, prediction in holdout_predictions.items()
        },
    }
    holdout_metrics["selected_model"] = selected_name
    holdout_metrics["selected_model_metrics"] = holdout_metrics[selected_name]
    holdout_probability = probability_metrics(
        holdout_actual, holdout_predictions[selected_name],
        offsets=(-7.5, -5.5, -2.5, 2.5, 5.5, 7.5),
        calibration_residuals=calibration_residuals,
    )

    best_hgb_params = validation_scores["hist_gradient_boosting"]["selected_params"]
    quantile_models, interval_diagnostics = fit_quantile_models(
        train, validation, holdout, best_hgb_params
    )
    quantile_models_holdout = {
        "q10": quantile_models["0.1"].predict(holdout[FEATURES]),
        "q90": quantile_models["0.9"].predict(holdout[FEATURES]),
    }
    correction = interval_diagnostics["validation_conformal_correction_points"]
    interval_diagnostics["holdout_mean_interval_width"] = float(np.mean(
        quantile_models_holdout["q90"] - quantile_models_holdout["q10"] + 2 * correction
    ))

    archive = market_archive_diagnostic(
        features, teams, selected_model, selected_name, calibration_residuals
    )
    result = {
        "status": "completed",
        "method": "point-in-time NBA Stats team-game features; frozen chronological train/validation/holdout",
        "source": "llimllib/nba_data v2 team_game_logs and games",
        "source_metadata": {
            name: {
                "first_season": metadata[name].get("first_season"),
                "last_season": metadata[name].get("last_season"),
                "seasons": metadata[name].get("seasons"),
                "updated": metadata[name].get("updated"),
            } for name in ("team_game_logs", "games")
        },
        "feature_rows": int(len(features)),
        "regular_season_feature_rows": int((features.game_type == "regular_season").sum()),
        "train": {
            "games": int(len(train)),
            "start": str(train.game_date.min().date()),
            "end": str(train.game_date.max().date()),
        },
        "validation": {
            "games": int(len(validation)),
            "start": str(validation.game_date.min().date()),
            "end": str(validation.game_date.max().date()),
            "model_selection_metrics": validation_scores,
            "calibration_residual_count": int(len(calibration_residuals)),
            "calibration_residual_mean": float(np.mean(calibration_residuals)),
            "calibration_residual_std": float(np.std(calibration_residuals, ddof=1)),
        },
        "holdout": {
            "games": int(len(holdout)),
            "start": str(holdout.game_date.min().date()),
            "end": str(holdout.game_date.max().date()),
            "metrics": holdout_metrics,
            "probability_calibration": holdout_probability,
            "prediction_interval": interval_diagnostics,
        },
        "selected_model": selected_name,
        "market_archive_diagnostic": archive,
        "guardrails": {
            "all_feature_history_dates_strictly_prior": True,
            "same_date_games_batch_updated": True,
            "model_and_calibration_frozen_before_holdout": True,
            "holdout_used_for_model_selection": False,
            "probability_lines_are_synthetic_offsets_except_market_archive": True,
            "market_archive_is_previously_examined": True,
            "no_roi_claim": True,
            "production_unchanged": True,
        },
    }
    (output / "smart_total_probability_report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (output / "smart_total_probability_report.md").write_text(
        render_report(result), encoding="utf-8"
    )
    (output / "holdout_predictions.csv").write_text(
        holdout.assign(
            selected_prediction=holdout_predictions[selected_name],
            calibration_residual=holdout_actual - holdout_predictions[selected_name],
            q10=quantile_models_holdout["q10"] - correction,
            q90=quantile_models_holdout["q90"] + correction,
        ).to_csv(index=False),
        encoding="utf-8",
    )
    joblib.dump({
        "model": selected_model,
        "model_name": selected_name,
        "features": FEATURES,
        "calibration_residuals": calibration_residuals,
        "train_end_exclusive": str(TRAIN_END.date()),
        "validation_end_exclusive": str(VALIDATION_END.date()),
        "probability_method": "empirical validation residual distribution with plus-one smoothing",
        "quantile_models": quantile_models,
        "conformal_correction": correction,
    }, output / "smart_total_probability_model.joblib")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Smart NBA total probability — chronological experiment",
        "",
        f"Status: {result['status']}",
        f"Selected model: {result['selected_model']}",
        "",
        "## Frozen chronological data splits",
        "",
        "| Split | Games | Start | End |",
        "|---|---:|---|---|",
    ]
    for key in ("train", "validation", "holdout"):
        row = result[key]
        lines.append(f"| {key} | {row['games']} | {row['start']} | {row['end']} |")
    lines += [
        "",
        "## Holdout point-prediction results",
        "",
        "| Model | Games | MAE | RMSE | Bias | Within 10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = result["holdout"]["metrics"]
    for name in ("structural_formula", "ridge", "extra_trees", "hist_gradient_boosting"):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['games']} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['bias_prediction_minus_actual']:.3f} | {row['within_10_points_pct']:.2f}% |"
        )
    lines += [
        "",
        "## Probability calibration",
        "",
        "| Synthetic line offset | Observed Over | Predicted Over | Brier | Log loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["holdout"]["probability_calibration"]["synthetic_line_calibration"]:
        lines.append(
            f"| {row['line_offset_from_model_points']:+.1f} | "
            f"{row['observed_over_rate_pct']:.2f}% | "
            f"{row['mean_predicted_over_probability_pct']:.2f}% | "
            f"{row['brier_score']:.4f} | {row['log_loss']:.4f} |"
        )
    interval = result["holdout"]["prediction_interval"]
    lines += [
        "",
        "## Prediction interval",
        "",
        f"- Nominal coverage: {interval['nominal_coverage']:.0%}",
        f"- Holdout empirical coverage: {interval['holdout_empirical_coverage_pct']:.2f}%",
        f"- Mean interval width: {interval['holdout_mean_interval_width']:.2f} points",
        f"- Validation conformal correction: {interval['validation_conformal_correction_points']:.2f} points",
        "",
        "## Market archive diagnostic",
        "",
        f"{json.dumps(result['market_archive_diagnostic'], indent=2, default=str)}",
        "",
        "## Interpretation safeguards",
        "",
        "- The holdout is never used to select a model, tune hyperparameters, or calibrate probabilities.",
        "- The probability table uses synthetic half-point offsets around the model prediction because full historical market coverage is unavailable.",
        "- The 181-game market archive has already been examined and is exploratory only.",
        "- The experiment estimates predictive distributions; it does not establish betting ROI.",
        "- No production model, live betting path, or paper-trading ledger is modified.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="outputs/research/smart_total_probability"
    )
    args = parser.parse_args()
    result = run(args.output_dir)
    print(render_report(result))


if __name__ == "__main__":
    main()
