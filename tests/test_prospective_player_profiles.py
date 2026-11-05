from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.prospective_player_profiles import (
    HISTORY_COLUMNS,
    PROFILE_FEATURES,
    aggregate_live_player_profile_features,
)


def _rotation_result():
    rows = []
    for team_id, prefix in (("HOME", "h"), ("AWAY", "a")):
        for rank in range(1, 6):
            rows.append({
                "game_id": "fixture-1",
                "team_id": team_id,
                "player_id": f"{prefix}{rank}",
                "rotation_rank": rank,
                "baseline_minutes": 48.0,
                "expected_minutes": 48.0,
                "status": "AVAILABLE",
            })
    return {
        "projections": pd.DataFrame(rows),
        "target_games": pd.DataFrame([{
            "game_id": "fixture-1",
            "home_team_id": "HOME",
            "away_team_id": "AWAY",
        }]),
    }


def _history(player_id: str, days: int = 5, points: float = 20.0):
    target = pd.Timestamp("2026-11-05")
    rows = []
    for offset in range(days, 0, -1):
        rows.append({
            "player_id": player_id,
            "game_date": target - pd.Timedelta(days=offset),
            "min": 30.0,
            "pts": points,
            "fga": 15.0,
            "fg3a": 5.0,
            "fg3m": 2.0,
            "fta": 4.0,
            "oreb": 1.0,
            "reb": 7.0,
            "ast": 5.0,
            "tov": 2.0,
            "stl": 1.0,
            "blk": 1.0,
            "usg_pct": 0.25,
            "ts_pct": 0.60,
        })
    return rows


def test_profile_features_match_stage_c_contract_and_use_expected_minutes():
    history = pd.DataFrame(
        _history("h1", days=5, points=20.0)
        + _history("h2", days=4, points=50.0)
        + _history("a1", days=5, points=18.0)
        + _history("a2", days=4, points=50.0)
    )
    features, players, diagnostics = aggregate_live_player_profile_features(
        _rotation_result(), "2026-11-05", history
    )

    assert len(PROFILE_FEATURES) == 26
    assert len(set(PROFILE_FEATURES)) == 26
    assert set(features["fixture-1"]) == set(PROFILE_FEATURES)
    assert features["fixture-1"]["home_profile_expected_pts"] == pytest.approx(32.0)
    assert features["fixture-1"]["away_profile_expected_pts"] == pytest.approx(28.8)
    assert features["fixture-1"]["home_profile_profiled_minutes_share"] == pytest.approx(0.2)
    assert features["fixture-1"]["home_profile_minute_weighted_ts"] == pytest.approx(0.60)

    eligible = players[players.profile_eligible]
    assert set(eligible.player_id) == {"h1", "a1"}
    assert (pd.to_datetime(eligible.profile_through_date) < pd.Timestamp("2026-11-05")).all()
    assert diagnostics["same_day_or_future_profile_rows"] == 0
    assert diagnostics["eligible_profile_players"] == 2


def test_profile_builder_rejects_same_day_or_future_history():
    history = pd.DataFrame(_history("h1", days=5))
    future = pd.DataFrame([{
        **_history("h1", days=1)[0],
        "game_date": pd.Timestamp("2026-11-05"),
    }])
    history = pd.concat([history, future], ignore_index=True)
    with pytest.raises(RuntimeError, match="same-day or future"):
        aggregate_live_player_profile_features(_rotation_result(), "2026-11-05", history)


def test_profile_builder_handles_rookies_without_prior_profile_rows():
    empty = pd.DataFrame(columns=HISTORY_COLUMNS)
    features, players, diagnostics = aggregate_live_player_profile_features(
        _rotation_result(), "2026-11-05", empty
    )
    assert set(features["fixture-1"]) == set(PROFILE_FEATURES)
    assert features["fixture-1"]["home_profile_expected_pts"] == 0.0
    assert features["fixture-1"]["home_profile_profiled_minutes_share"] == 0.0
    assert players.profile_eligible.sum() == 0
    assert diagnostics["players_with_prior_profile"] == 0


def test_profile_builder_requires_each_team_rotation_to_reconcile_to_240():
    rotation = _rotation_result()
    rotation["projections"].loc[0, "expected_minutes"] = 47.0
    history = pd.DataFrame(_history("h1", days=5))
    with pytest.raises(ValueError, match="240 minutes"):
        aggregate_live_player_profile_features(rotation, "2026-11-05", history)
