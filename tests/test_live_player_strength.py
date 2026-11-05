import json

import numpy as np
import pytest

from research.live_player_strength import (
    FEATURES,
    build_player_strength_features,
    load_calibration,
    predict_player_adjusted_total,
)


def sample_calibration():
    return {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "research_shadow_only",
        "training_cutoff_exclusive": "2026-07-01",
        "training_last_date": "2026-06-30",
        "training_games": 900,
        "features": list(FEATURES),
        "imputer": {"statistics": [0.0] * len(FEATURES)},
        "standardizer": {"mean": [0.0] * len(FEATURES),
                         "scale": [1.0] * len(FEATURES)},
        "ridge": {"intercept": 1.0, "coefficients": [0.1] * len(FEATURES)},
        "betting_enabled": False,
        "probabilities_calibrated": False,
    }


def test_live_feature_mapping_matches_training_order():
    home = {
        "off_delta": 0.2, "def_delta": -0.1, "minutes_changed": 18,
        "reported_out_minutes": 24, "questionable_minutes": 12,
        "reported_players_without_prior_team_history": 0,
        "players_excluded_by_opposing_report": 1,
    }
    away = {
        "off_delta": -0.1, "def_delta": 0.05, "minutes_changed": 10,
        "reported_out_minutes": 12, "questionable_minutes": 0,
        "reported_players_without_prior_team_history": 1,
        "players_excluded_by_opposing_report": 0,
    }
    features = build_player_strength_features(home, away, 99.5)
    assert tuple(features) == FEATURES
    assert features["raw_injury_total_delta"] == pytest.approx(0.04975)
    assert features["expected_possessions"] == pytest.approx(99.5)


def test_prediction_reproduces_exported_standardized_ridge_formula():
    calibration = sample_calibration()
    features = {name: 1.0 for name in FEATURES}
    result = predict_player_adjusted_total(228.0, features, calibration)
    assert result["predicted_v5_error"] == pytest.approx(2.6)
    assert result["player_adjusted_total"] == pytest.approx(225.4)


def test_loader_rejects_future_training_cutoff(tmp_path):
    payload = sample_calibration()
    payload["training_last_date"] = "2026-07-01"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly pre-2026-27"):
        load_calibration(path)


def test_prediction_rejects_missing_features():
    with pytest.raises(ValueError, match="keys/order"):
        predict_player_adjusted_total(228.0, {"home_off_delta": 1.0}, sample_calibration())
