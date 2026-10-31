from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v4_paired_v2_vs_v4_backtest import (
    build_paired_predictions,
    paired_bootstrap_mae_difference,
    regression_metrics,
)


def raw_games():
    rows = []
    games = [
        ("1", "2026-01-01", "A", "B", 100, 95, False),
        ("2", "2026-01-02", "C", "D", 101, 99, False),
        ("3", "2026-01-03", "A", "B", 110, 115, False),
        ("4", "2026-01-03", "C", "D", 105, 100, True),
    ]
    for gid, date, home, away, hp, ap, neutral in games:
        rows.extend([
            {
                "season": 2026, "game_id": gid, "game_date": date,
                "team_id": home, "opp_team_id": away, "pts": hp,
                "home": not neutral, "neutral_site": neutral,
                "game_type": "regular_season",
            },
            {
                "season": 2026, "game_id": gid, "game_date": date,
                "team_id": away, "opp_team_id": home, "pts": ap,
                "home": False, "neutral_site": neutral,
                "game_type": "regular_season",
            },
        ])
    return pd.DataFrame(rows)


def test_paired_backtest_uses_same_games_and_strictly_prior_history():
    features = pd.DataFrame([{
        "season": 2026,
        "game_id": "3",
        "game_date": pd.Timestamp("2026-01-03"),
        "home_team_id": "A",
        "away_team_id": "B",
        "actual_total": 225.0,
        "projected_total": 220.0,
        "league_environment_total": 218.0,
        "efficiency_only_total": 219.0,
        "pace_only_total": 217.0,
        "max_history_date": pd.Timestamp("2026-01-02"),
    }])
    seen = []

    def fake_v2(**kwargs):
        history = kwargs["history_df"]
        seen.append(history.copy())
        assert (history["game_date"] < pd.Timestamp(kwargs["game_date"])).all()
        return {"projected_total": 222.0}

    results, diagnostics = build_paired_predictions(
        raw_games(),
        {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"},
        v4_features=features,
        predictor=fake_v2,
    )

    assert len(results) == 1
    assert results.loc[0, "actual_total"] == 225.0
    assert results.loc[0, "v2_points_engine_total"] == 222.0
    assert results.loc[0, "v4_pace_efficiency_total"] == 220.0
    assert len(seen) == 1
    assert seen[0]["game_date"].max() == pd.Timestamp("2026-01-02")
    assert diagnostics["neutral_site_games_excluded"] == 0
    assert diagnostics["v2_history_strictly_prior"] is True


def test_neutral_site_targets_are_excluded_from_paired_comparison():
    features = pd.DataFrame([
        {
            "season": 2026,
            "game_id": "3",
            "game_date": pd.Timestamp("2026-01-03"),
            "home_team_id": "A",
            "away_team_id": "B",
            "actual_total": 225.0,
            "projected_total": 220.0,
            "league_environment_total": 218.0,
            "efficiency_only_total": 219.0,
            "pace_only_total": 217.0,
            "max_history_date": pd.Timestamp("2026-01-02"),
        },
        {
            "season": 2026,
            "game_id": "4",
            "game_date": pd.Timestamp("2026-01-03"),
            "home_team_id": "C",
            "away_team_id": "D",
            "actual_total": 205.0,
            "projected_total": 210.0,
            "league_environment_total": 210.0,
            "efficiency_only_total": 210.0,
            "pace_only_total": 210.0,
            "max_history_date": pd.Timestamp("2026-01-02"),
        },
    ])
    results, diagnostics = build_paired_predictions(
        raw_games(),
        {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"},
        v4_features=features,
        predictor=lambda **kwargs: {"projected_total": 210.0},
    )
    assert len(results) == 1
    assert results.loc[0, "game_id"] == "3"
    assert diagnostics["neutral_site_games_excluded"] == 1
    assert diagnostics["v2_predictions_successful"] == 1


def test_metrics_and_paired_bootstrap_are_deterministic():
    actual = pd.Series([200.0, 210.0, 220.0])
    predicted = pd.Series([202.0, 207.0, 221.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["games"] == 3
    assert metrics["mae"] == pytest.approx(2.0)
    frame = pd.DataFrame({
        "v2_absolute_error": [3.0, 2.0, 4.0, 1.0],
        "v4_absolute_error": [2.0, 3.0, 2.0, 1.0],
    })
    first = paired_bootstrap_mae_difference(frame, reps=200, seed=123)
    second = paired_bootstrap_mae_difference(frame, reps=200, seed=123)
    assert first == second
    assert first["observed_mae_difference"] == pytest.approx(-0.5)
    assert first["v4_lower_absolute_error_games"] == 2
    assert first["v2_lower_absolute_error_games"] == 1
    assert first["ties"] == 1
