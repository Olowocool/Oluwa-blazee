#!/usr/bin/env python3
"""Objective 5: point-in-time market comparison and calibrated residual probabilities.

Research only. Uses the frozen DraftKings T-120/240-minute protocol from
Objective 3, then compares a market-residual Ridge distribution with V5 and
the entry market's no-vig probability. It does not select bets or claim ROI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.objective3_v5_market_residual import (
    DEFAULT_SNAPSHOTS,
    DEFAULT_SCORES,
    DEFAULT_SCORE_PATCH_DIR,
    BOOKMAKER,
    ENTRY_MINUTES,
    MAX_ENTRY_AGE_MINUTES,
    MIN_TRAIN_GAMES,
    RIDGE_ALPHA,
    FEATURES,
    prepare_snapshots,
    select_entry_and_last,
    discover_score_patch_files,
    build_game_dataset,
    no_vig,
)

MIN_CALIBRATION_GAMES = 100
DEFAULT_OUTPUT = "outputs/research/objective5_market_comparison"


def _finite(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a non-empty finite array.")
    return array


def empirical_total_probabilities(
    predicted_total: float,
    market_line: float,
    calibration_errors: Any,
) -> tuple[float, float, float]:
    """Return P(Over), P(Under), P(Push) from prior OOS total-prediction errors.

    For half-point lines, push probability is exactly zero. For integer lines,
    the one-point interval around the integer outcome is assigned to a push.
    Laplace smoothing is applied to the mutually exclusive outcome buckets.
    """
    errors = _finite(calibration_errors, "Calibration errors")
    predicted_total = float(predicted_total)
    market_line = float(market_line)
    if not np.isfinite([predicted_total, market_line]).all():
        raise ValueError("Predicted total and market line must be finite.")
    threshold = market_line - predicted_total
    if abs(market_line - round(market_line)) < 1e-9:
        over = int(np.sum(errors > threshold + 0.5))
        under = int(np.sum(errors < threshold - 0.5))
        push = int(len(errors) - over - under)
        denom = len(errors) + 3.0
        return (over + 1.0) / denom, (under + 1.0) / denom, (push + 1.0) / denom
    over = int(np.sum(errors > threshold))
    under = int(np.sum(errors < threshold))
    denom = len(errors) + 2.0
    return (over + 1.0) / denom, (under + 1.0) / denom, 0.0


def _price_ev(p_over: float, p_under: float, over_odds: float, under_odds: float) -> dict[str, float]:
    vals = [p_over, p_under, over_odds, under_odds]
    if not np.isfinite(vals).all() or min(p_over, p_under) < 0 or min(over_odds, under_odds) <= 1:
        raise ValueError("Invalid probabilities or decimal odds.")
    return {
        "over_ev_per_unit": float(p_over * (over_odds - 1.0) - p_under),
        "under_ev_per_unit": float(p_under * (under_odds - 1.0) - p_over),
        "over_edge_vs_break_even": float(p_over - 1.0 / over_odds),
        "under_edge_vs_break_even": float(p_under - 1.0 / under_odds),
    }


def _log_loss(probability: np.ndarray, outcome: np.ndarray) -> float:
    p = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(outcome * np.log(p) + (1.0 - outcome) * np.log(1.0 - p)))


def probability_metrics(frame: pd.DataFrame, probability_col: str) -> dict[str, Any]:
    sample = frame.loc[frame[probability_col].notna()].copy()
    if sample.empty:
        return {"games": 0, "brier": None, "log_loss": None, "mean_probability": None,
                "observed_over_rate": None}
    sample = sample.loc[sample["actual_total"] != sample["entry_total"]].copy()
    if sample.empty:
        return {"games": 0, "brier": None, "log_loss": None, "mean_probability": None,
                "observed_over_rate": None}
    outcome = (sample["actual_total"].to_numpy(float) > sample["entry_total"].to_numpy(float)).astype(float)
    p = sample[probability_col].to_numpy(float)
    push_col = probability_col.replace("_over_probability", "_push_probability")
    if probability_col in {"residual_model_over_probability", "v5_over_probability"} and push_col in sample.columns:
        push = sample[push_col].to_numpy(float)
        non_push = 1.0 - push
        p = np.divide(p, non_push, out=np.full_like(p, np.nan), where=non_push > 0)
        keep = np.isfinite(p)
        p, outcome = p[keep], outcome[keep]
    reliability = []
    edges = np.linspace(0.0, 1.0, 6)
    for i in range(len(edges) - 1):
        mask = (p >= edges[i]) & (
            (p < edges[i + 1]) if i < len(edges) - 2 else (p <= edges[i + 1])
        )
        if mask.any():
            reliability.append({
                "probability_bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
                "games": int(mask.sum()),
                "mean_predicted_over": float(p[mask].mean()),
                "observed_over_rate": float(outcome[mask].mean()),
            })
    return {
        "games": int(len(sample)),
        "brier": float(np.mean((p - outcome) ** 2)),
        "log_loss": _log_loss(p, outcome),
        "mean_probability": float(p.mean()),
        "observed_over_rate": float(outcome.mean()),
        "reliability_bins": reliability,
    }


def _date_cluster_bootstrap_brier(
    frame: pd.DataFrame, candidate_col: str, market_col: str,
    split: str = "reserved_holdout", n_resamples: int = 5000, seed: int = 51,
) -> dict[str, Any]:
    d = frame.loc[frame["evaluation_split"].eq(split)].copy()
    d = d.loc[d[candidate_col].notna() & d[market_col].notna()]
    d = d.loc[d["actual_total"] != d["entry_total"]]
    if d.empty:
        return {"games": 0, "game_dates": 0, "difference_candidate_minus_market": None, "ci_95": None}
    d["outcome"] = (d["actual_total"] > d["entry_total"]).astype(float)
    candidate_probability = d[candidate_col].to_numpy(float)
    push_col = candidate_col.replace("_over_probability", "_push_probability")
    if push_col in d.columns:
        candidate_probability = candidate_probability / np.maximum(
            1.0 - d[push_col].to_numpy(float), 1e-12
        )
    d["loss_diff"] = (candidate_probability - d.outcome.to_numpy(float)) ** 2 - (
        d[market_col].to_numpy(float) - d.outcome.to_numpy(float)
    ) ** 2
    clusters = [g.loss_diff.to_numpy(float) for _, g in d.groupby("game_date", sort=True)]
    observed = float(d.loss_diff.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[j] for j in picked])
        boot[i] = values.mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "games": int(len(d)), "game_dates": int(len(clusters)),
        "difference_candidate_minus_market": observed,
        "ci_95": [float(lo), float(hi)], "resamples": n_resamples, "seed": seed,
        "interpretation": "Negative favours candidate; positive favours no-vig market probability.",
    }


def walk_forward_market_probabilities(
    games: pd.DataFrame,
    min_train_games: int = MIN_TRAIN_GAMES,
    min_calibration_games: int = MIN_CALIBRATION_GAMES,
    alpha: float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """Prequential development predictions and one frozen chronological holdout."""
    if min_train_games < 1 or min_calibration_games < 1 or alpha <= 0:
        raise ValueError("Training, calibration and alpha settings must be positive.")
    d = games.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    d = d.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    numeric = FEATURES + ["market_residual_target", "entry_total", "actual_total", "v5_total"]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d[["game_date"] + numeric].isna().any().any():
        raise ValueError("Market evaluation data contain missing required values; no imputation is allowed.")
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise ValueError("Market evaluation data contain non-finite values.")
    dates = np.array(sorted(d.game_date.unique()))
    if len(dates) < 3:
        raise ValueError("At least three distinct game dates are required for chronological evaluation.")
    holdout_start = pd.Timestamp(dates[max(1, int(np.floor(len(dates) * 0.8)))])
    d["evaluation_split"] = np.where(d.game_date >= holdout_start, "reserved_holdout", "development")

    for col in (
        "residual_model_total", "predicted_market_residual", "residual_training_games",
        "residual_model_status", "residual_model_over_probability",
        "residual_model_under_probability", "residual_model_push_probability",
        "v5_over_probability", "v5_under_probability", "v5_push_probability",
        "residual_model_over_ev", "residual_model_under_ev", "v5_over_ev", "v5_under_ev",
        "residual_model_over_edge_vs_market", "v5_over_edge_vs_market",
    ):
        d[col] = np.nan if col != "residual_model_status" else "warmup"

    development = d.loc[d.evaluation_split.eq("development")]
    residual_calibration: list[float] = []
    v5_calibration: list[float] = []
    parts = []
    for day, today in development.groupby("game_date", sort=True):
        train = development.loc[development.game_date < day]
        row = today.copy()
        row["residual_training_games"] = len(train)
        row["residual_model_status"] = "warmup"
        d.loc[today.index, "residual_training_games"] = len(train)
        d.loc[today.index, "residual_model_status"] = "warmup"
        if len(train) >= min_train_games:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[FEATURES], train.market_residual_target)
            pred_residual = model.predict(today[FEATURES])
            pred_total = today.entry_total.to_numpy(float) + pred_residual
            row["predicted_market_residual"] = pred_residual
            row["residual_model_total"] = pred_total
            row["residual_model_status"] = "walk_forward_scored"
            d.loc[today.index, "predicted_market_residual"] = pred_residual
            d.loc[today.index, "residual_model_total"] = pred_total
            d.loc[today.index, "residual_model_status"] = "walk_forward_scored"
            if len(residual_calibration) >= min_calibration_games:
                for idx, prediction in zip(row.index, pred_total):
                    p_over, p_under, p_push = empirical_total_probabilities(
                        prediction, float(d.loc[idx, "entry_total"]), residual_calibration
                    )
                    d.loc[idx, ["residual_model_over_probability", "residual_model_under_probability",
                                "residual_model_push_probability"]] = [p_over, p_under, p_push]
            if len(v5_calibration) >= min_calibration_games:
                for idx, prediction in zip(row.index, today.v5_total.to_numpy(float)):
                    p_over, p_under, p_push = empirical_total_probabilities(
                        prediction, float(d.loc[idx, "entry_total"]), v5_calibration
                    )
                    d.loc[idx, ["v5_over_probability", "v5_under_probability", "v5_push_probability"]] = [
                        p_over, p_under, p_push
                    ]
            # Append today's forecast errors only after all games on this date
            # have been scored, so same-day outcomes cannot calibrate each other.
            residual_calibration.extend((today.actual_total.to_numpy(float) - pred_total).tolist())
        v5_calibration.extend((today.actual_total.to_numpy(float) - today.v5_total.to_numpy(float)).tolist())
        parts.append(row)

    holdout = d.loc[d.evaluation_split.eq("reserved_holdout")].copy()
    holdout["residual_training_games"] = len(development)
    holdout["residual_model_status"] = "warmup"
    if len(development) >= min_train_games:
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        final_model.fit(development[FEATURES], development.market_residual_target)
        pred_residual = final_model.predict(holdout[FEATURES])
        pred_total = holdout.entry_total.to_numpy(float) + pred_residual
        holdout["predicted_market_residual"] = pred_residual
        holdout["residual_model_total"] = pred_total
        holdout["residual_model_status"] = "frozen_holdout_scored"
        if len(residual_calibration) >= min_calibration_games:
            for idx, prediction in zip(holdout.index, pred_total):
                p_over, p_under, p_push = empirical_total_probabilities(
                    prediction, float(holdout.loc[idx, "entry_total"]), residual_calibration
                )
                holdout.loc[idx, ["residual_model_over_probability", "residual_model_under_probability",
                                  "residual_model_push_probability"]] = [p_over, p_under, p_push]
        if len(v5_calibration) >= min_calibration_games:
            for idx, prediction in zip(holdout.index, holdout.v5_total.to_numpy(float)):
                p_over, p_under, p_push = empirical_total_probabilities(
                    prediction, float(holdout.loc[idx, "entry_total"]), v5_calibration
                )
                holdout.loc[idx, ["v5_over_probability", "v5_under_probability", "v5_push_probability"]] = [
                    p_over, p_under, p_push
                ]
    result = pd.concat([d, holdout], ignore_index=True)
    # d already contains original development rows and holdout rows; overwrite
    # holdout rows with the frozen predictions and return one row per game.
    result = pd.concat([d.loc[d.evaluation_split.eq("development")], holdout], ignore_index=True)
    result = result.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)

    for prefix in ("residual_model", "v5"):
        over_col = f"{prefix}_over_probability"
        under_col = f"{prefix}_under_probability"
        for side in ("over", "under"):
            odds_col = f"entry_{side}_odds_decimal"
            ev_col = f"{prefix}_{side}_ev"
            edge_col = f"{prefix}_{side}_edge_vs_market"
            probability_col = over_col if side == "over" else under_col
            if odds_col not in result.columns:
                raise ValueError(f"Missing entry price field: {odds_col}")
            result[ev_col] = result[probability_col] * (result[odds_col] - 1.0) - result[
                under_col if side == "over" else over_col
            ]
            market_probability_col = "over_probability_no_vig" if side == "over" else "under_probability_no_vig"
            conditional_probability = result[probability_col] / (
            result[over_col] + result[under_col]
        ).replace(0, np.nan)
        result[edge_col] = conditional_probability - result[market_probability_col]
    return result


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    holdout = frame.loc[frame.evaluation_split.eq("reserved_holdout")].copy()
    eligible = holdout.loc[holdout.residual_model_status.eq("frozen_holdout_scored")]
    def total_metrics(col: str) -> dict[str, Any]:
        pred = pd.to_numeric(eligible[col], errors="coerce")
        actual = eligible.actual_total.astype(float)
        ok = pred.notna()
        if not ok.any():
            return {"games": 0, "mae": None, "rmse": None, "bias": None}
        error = pred[ok] - actual[ok]
        return {"games": int(ok.sum()), "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}

    market_prob = probability_metrics(eligible, "over_probability_no_vig")
    v5_prob = probability_metrics(eligible, "v5_over_probability")
    residual_prob = probability_metrics(eligible, "residual_model_over_probability")
    # Entry-market point forecast is a benchmark only; the probability benchmark
    # is the no-vig Over probability from the same entry row.
    return {
        "status": (
            "completed" if eligible.residual_model_over_probability.notna().any()
            else "insufficient_calibration_sample"
        ),
        "holdout_start_date": str(holdout.game_date.min().date()) if len(holdout) else None,
        "holdout_market_games": int(len(holdout)),
        "holdout_residual_model_games": int(len(eligible)),
        "residual_model_total": total_metrics("residual_model_total"),
        "v5_total": total_metrics("v5_total"),
        "entry_market_total": total_metrics("entry_total"),
        "probability_comparison": {
            "entry_market_no_vig": market_prob,
            "v5_distribution": v5_prob,
            "market_residual_distribution": residual_prob,
            "residual_vs_market_brier_bootstrap": _date_cluster_bootstrap_brier(
                eligible, "residual_model_over_probability", "over_probability_no_vig"
            ),
            "residual_vs_v5_brier_bootstrap": _date_cluster_bootstrap_brier(
                eligible, "residual_model_over_probability", "v5_over_probability"
            ),
        },
        "market_residual_probability_ev_diagnostics": {
            "games_with_model_probabilities": int(eligible.residual_model_over_probability.notna().sum()),
            "mean_over_ev_per_unit": float(eligible.residual_model_over_ev.mean()) if eligible.residual_model_over_ev.notna().any() else None,
            "mean_under_ev_per_unit": float(eligible.residual_model_under_ev.mean()) if eligible.residual_model_under_ev.notna().any() else None,
            "positive_ev_over_count": int((eligible.residual_model_over_ev > 0).sum()),
            "positive_ev_under_count": int((eligible.residual_model_under_ev > 0).sum()),
            "note": "Expected-value diagnostics only; no bet selection, staking or realized ROI is performed in Objective 5.",
        },
        "calibration_sample": {
            "minimum_prior_oos_errors": MIN_CALIBRATION_GAMES,
            "development_probability_rows": int(frame.loc[frame.evaluation_split.eq("development"), "residual_model_over_probability"].notna().sum()),
            "holdout_probability_rows": int(eligible.residual_model_over_probability.notna().sum()),
        },
        "interpretation": {
            "brier_difference": "Lower is better. Bootstrap difference is candidate minus comparison.",
            "roi_claim": False,
            "market_edge_claim": False,
        },
    }


def market_sample_readiness(
    games: pd.DataFrame,
    min_train_games: int = MIN_TRAIN_GAMES,
    min_calibration_games: int = MIN_CALIBRATION_GAMES,
) -> dict[str, Any]:
    dates = sorted(pd.to_datetime(games.game_date, errors="coerce").dt.normalize().dropna().unique())
    if len(dates) < 3:
        return {
            "ready_for_holdout_probability_test": False,
            "unique_game_dates": int(len(dates)),
            "development_games": 0,
            "reserved_holdout_games": 0,
            "development_walk_forward_predictions": 0,
            "development_probability_predictions": 0,
            "reason": "At least three distinct eligible game dates are required.",
        }
    holdout_start = pd.Timestamp(dates[max(1, int(np.floor(len(dates) * 0.8)))])
    development = games.loc[pd.to_datetime(games.game_date).dt.normalize() < holdout_start]
    holdout = games.loc[pd.to_datetime(games.game_date).dt.normalize() >= holdout_start]
    train_count = 0
    calibration_count = 0
    walk_forward_predictions = 0
    probability_predictions = 0
    for day, today in development.groupby(pd.to_datetime(development.game_date).dt.normalize(), sort=True):
        if train_count >= min_train_games:
            walk_forward_predictions += len(today)
            if calibration_count >= min_calibration_games:
                probability_predictions += len(today)
            calibration_count += len(today)
        train_count += len(today)
    ready = (
        len(development) >= min_train_games
        and walk_forward_predictions >= min_calibration_games
        and len(holdout) > 0
        and calibration_count >= min_calibration_games
    )
    return {
        "ready_for_holdout_probability_test": bool(ready),
        "unique_game_dates": int(len(dates)),
        "development_games": int(len(development)),
        "reserved_holdout_games": int(len(holdout)),
        "development_walk_forward_predictions": int(walk_forward_predictions),
        "development_probability_predictions": int(probability_predictions),
        "minimum_prior_training_games": int(min_train_games),
        "minimum_prior_oos_calibration_games": int(min_calibration_games),
        "reason": None if ready else "Development history has not yet produced the required earlier training and out-of-sample calibration samples.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw = pd.read_csv(args.snapshots_file)
    snapshots, snapshot_diag = prepare_snapshots(raw, args.bookmaker)
    selected, entry_diag = select_entry_and_last(snapshots, args.entry_minutes, args.max_entry_age_minutes)
    if selected.empty:
        raise ValueError("No games have an eligible fixed-cutoff sportsbook entry.")
    patches = discover_score_patch_files(args.score_patch_dir, args.score_patch_files)
    games, score_diag = build_game_dataset(selected, args.scores_file, [str(p) for p in patches])
    readiness = market_sample_readiness(games, args.min_train_games, args.min_calibration_games)
    if readiness["unique_game_dates"] < 3:
        # Preserve the matched game rows and a stable schema so downstream
        # Objective 6/7 runners can fail closed without an empty-CSV crash.
        scored = games.copy()
        scored["evaluation_split"] = "insufficient_sample"
        scored["residual_model_status"] = "insufficient_market_sample"
        for col in (
            "residual_model_total", "predicted_market_residual",
            "residual_model_over_probability", "residual_model_under_probability",
            "residual_model_push_probability", "v5_over_probability",
            "v5_under_probability", "v5_push_probability",
        ):
            scored[col] = np.nan
        summary = {
            "status": "insufficient_market_sample",
            "holdout_market_games": readiness["reserved_holdout_games"],
            "holdout_residual_model_games": 0,
            "probability_comparison": {
                "entry_market_no_vig": {"games": 0, "brier": None, "log_loss": None},
                "v5_distribution": {"games": 0, "brier": None, "log_loss": None},
                "market_residual_distribution": {"games": 0, "brier": None, "log_loss": None},
            },
            "market_residual_probability_ev_diagnostics": {
                "note": "No probability or expected-value result: insufficient distinct game dates."
            },
            "readiness": readiness,
            "interpretation": {"roi_claim": False, "market_edge_claim": False},
        }
    else:
        scored = walk_forward_market_probabilities(
            games, args.min_train_games, args.min_calibration_games, args.ridge_alpha
        )
        summary = build_summary(scored)
        summary["readiness"] = readiness
    diagnostics = {
        "status": summary["status"],
        "bookmaker": args.bookmaker,
        "entry_cutoff_minutes": args.entry_minutes,
        "maximum_entry_age_minutes": args.max_entry_age_minutes,
        "minimum_residual_training_games": args.min_train_games,
        "minimum_prior_oos_calibration_errors": args.min_calibration_games,
        "ridge_alpha": args.ridge_alpha,
        **snapshot_diag, **entry_diag, **score_diag,
        "readiness": readiness,
        "eligible_game_dates": int(games.game_date.nunique()),
        "development_games": readiness["development_games"],
        "reserved_holdout_games": readiness["reserved_holdout_games"],
        "development_residual_model_predictions": int(scored.residual_model_status.eq("walk_forward_scored").sum()) if not scored.empty else 0,
        "holdout_residual_model_predictions": int(scored.residual_model_status.eq("frozen_holdout_scored").sum()) if not scored.empty else 0,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "objective5_market_comparison_game_results.csv", index=False)
    (output / "objective5_market_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (output / "objective5_market_comparison_data_quality.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "data_quality": diagnostics}, indent=2, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-file", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--scores-file", default=DEFAULT_SCORES)
    parser.add_argument("--score-patch-dir", default=str(DEFAULT_SCORE_PATCH_DIR))
    parser.add_argument("--score-patch-files", nargs="*", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--bookmaker", default=BOOKMAKER)
    parser.add_argument("--entry-minutes", type=int, default=ENTRY_MINUTES)
    parser.add_argument("--max-entry-age-minutes", type=int, default=MAX_ENTRY_AGE_MINUTES)
    parser.add_argument("--min-train-games", type=int, default=MIN_TRAIN_GAMES)
    parser.add_argument("--min-calibration-games", type=int, default=MIN_CALIBRATION_GAMES)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
