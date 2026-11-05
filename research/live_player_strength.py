"""Validated inference adapter for the Stage C live player-strength challenger.

The model adjusts the frozen V5 total using point-in-time RAPM, injury and
full-rotation features. It emits a research point estimate only: no probabilities,
bet recommendations, wagers, or paper-ledger writes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = (
    "home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta",
    "raw_injury_total_delta", "home_minutes_changed", "away_minutes_changed",
    "home_reported_out_minutes", "away_reported_out_minutes",
    "home_questionable_minutes", "away_questionable_minutes",
    "home_reported_players_without_prior_team_history",
    "away_reported_players_without_prior_team_history",
    "home_players_excluded_by_opposing_report",
    "away_players_excluded_by_opposing_report", "expected_possessions",
)
CALIBRATION_PATH = Path(__file__).with_name("live_player_strength_calibration.json")


def load_calibration(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else CALIBRATION_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Live player-strength calibration is not installed: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage-c-live-player-strength-calibration-v1":
        raise ValueError("Unsupported live player-strength calibration schema")
    if payload.get("status") != "research_shadow_only":
        raise ValueError("Calibration is not marked research-shadow-only")
    if payload.get("betting_enabled") is not False:
        raise ValueError("Player-strength calibration must not enable betting")
    if payload.get("probabilities_calibrated") is not False:
        raise ValueError("This point-total calibrator must not claim calibrated probabilities")
    if tuple(payload.get("features", ())) != FEATURES:
        raise ValueError("Calibration feature order does not match the live adapter")
    cutoff = str(payload.get("training_cutoff_exclusive", ""))
    last_date = str(payload.get("training_last_date", ""))
    if cutoff != "2026-07-01" or not last_date or last_date >= cutoff:
        raise ValueError("Calibration training window is not strictly pre-2026-27")
    n = len(FEATURES)
    imputer = payload.get("imputer", {})
    standardizer = payload.get("standardizer", {})
    ridge = payload.get("ridge", {})
    means = standardizer.get("mean", [])
    scales = standardizer.get("scale", [])
    coefficients = ridge.get("coefficients", [])
    fill = imputer.get("statistics", [])
    if not all(len(x) == n for x in (means, scales, coefficients, fill)):
        raise ValueError("Calibration parameter dimensions do not match feature count")
    values = [*means, *scales, *coefficients, *fill, ridge.get("intercept")]
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration parameters must be numeric") from exc
    if not np.isfinite(numeric).all() or any(float(x) <= 0 for x in scales):
        raise ValueError("Calibration parameters contain non-finite or invalid values")
    rapm_diagnostics = payload.get("rapm_feature_diagnostics")
    if not isinstance(rapm_diagnostics, dict):
        raise ValueError("Calibration lacks verified nonzero RAPM feature diagnostics")
    for name in FEATURES[:4]:
        item = rapm_diagnostics.get(name, {})
        try:
            nonzero_rows = int(item.get("nonzero_training_rows", 0))
            feature_std = float(item.get("standard_deviation", 0.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid RAPM feature diagnostic: {name}") from exc
        if nonzero_rows < 50 or not math.isfinite(feature_std) or feature_std <= 1e-8:
            raise ValueError(f"Calibration has no usable variation in RAPM feature: {name}")
    if int(payload.get("training_games", 0)) < 300:
        raise ValueError("Calibration has fewer than 300 prior training games")
    return payload


def build_player_strength_features(
    home: dict[str, Any],
    away: dict[str, Any],
    expected_possessions: float,
) -> dict[str, float]:
    """Map the audited Phase 2 team rotation summaries to training feature order."""
    try:
        possessions = float(expected_possessions)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected possessions are missing or invalid") from exc
    if not math.isfinite(possessions) or possessions <= 0:
        raise ValueError("Expected possessions must be finite and positive")

    def value(source: dict[str, Any], key: str) -> float:
        try:
            number = float(source[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Missing or invalid rotation feature: {key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Non-finite rotation feature: {key}")
        return number

    h_off, h_def = value(home, "off_delta"), value(home, "def_delta")
    a_off, a_def = value(away, "off_delta"), value(away, "def_delta")
    result = {
        "home_off_delta": h_off,
        "home_def_delta": h_def,
        "away_off_delta": a_off,
        "away_def_delta": a_def,
        "raw_injury_total_delta": possessions / 100.0 * (h_off + h_def + a_off + a_def),
        "home_minutes_changed": value(home, "minutes_changed"),
        "away_minutes_changed": value(away, "minutes_changed"),
        "home_reported_out_minutes": value(home, "reported_out_minutes"),
        "away_reported_out_minutes": value(away, "reported_out_minutes"),
        "home_questionable_minutes": value(home, "questionable_minutes"),
        "away_questionable_minutes": value(away, "questionable_minutes"),
        "home_reported_players_without_prior_team_history": value(
            home, "reported_players_without_prior_team_history"
        ),
        "away_reported_players_without_prior_team_history": value(
            away, "reported_players_without_prior_team_history"
        ),
        "home_players_excluded_by_opposing_report": value(
            home, "players_excluded_by_opposing_report"
        ),
        "away_players_excluded_by_opposing_report": value(
            away, "players_excluded_by_opposing_report"
        ),
        "expected_possessions": possessions,
    }
    if tuple(result) != FEATURES:
        raise RuntimeError("Live player-strength feature ordering drifted")
    return result


def predict_player_adjusted_total(
    frozen_v5_total: float,
    features: dict[str, float],
    calibration: dict[str, Any],
) -> dict[str, float]:
    """Apply the exported sklearn imputer/scaler/Ridge pipeline exactly."""
    try:
        baseline = float(frozen_v5_total)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Frozen V5 baseline must be numeric") from exc
    if not math.isfinite(baseline):
        raise ValueError("Frozen V5 baseline must be finite")
    if tuple(features) != FEATURES:
        raise ValueError("Live feature keys/order do not match calibration")

    x = np.asarray([features[name] for name in FEATURES], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Live player-strength features contain non-finite values")
    fill = np.asarray(calibration["imputer"]["statistics"], dtype=float)
    x = np.where(np.isnan(x), fill, x)
    mean = np.asarray(calibration["standardizer"]["mean"], dtype=float)
    scale = np.asarray(calibration["standardizer"]["scale"], dtype=float)
    coefficients = np.asarray(calibration["ridge"]["coefficients"], dtype=float)
    predicted_v5_error = float(
        calibration["ridge"]["intercept"] + np.dot((x - mean) / scale, coefficients)
    )
    adjusted = baseline - predicted_v5_error
    if not math.isfinite(adjusted):
        raise ValueError("Player-adjusted total is non-finite")
    return {
        "frozen_v5_total": round(baseline, 2),
        "predicted_v5_error": round(predicted_v5_error, 4),
        "player_adjusted_total": round(adjusted, 2),
        "player_adjustment": round(adjusted - baseline, 2),
    }
