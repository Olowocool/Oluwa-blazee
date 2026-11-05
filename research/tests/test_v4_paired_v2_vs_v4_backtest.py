from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from points_engine import calculate_matchup_points
from research.v4_paired_v2_vs_v4_backtest import (
    build_paired_predictions,
    build_score_history,
    prepare_v2_score_history,
    paired_bootstrap_mae_difference,
    regression_metrics,
    select_v2_relevant_history,
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
        "expected_possessions": 100.0,
        "league_off_eff_prior": 110.0,
        "league_pace_prior": 100.0,
        "expected_home_efficiency": 115.0,
        "expected_away_efficiency": 110.0,
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

    names = {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"}
    results, diagnostics = build_paired_predictions(
        raw_games(),
        names,
        v4_features=features,
        predictor=fake_v2,
        v2_history=build_score_history(raw_games(), names),
    )

    assert len(results) == 1
    assert results.loc[0, "actual_total"] == 225.0
    assert results.loc[0, "v2_points_engine_total"] == 222.0
    assert results.loc[0, "v4_pace_efficiency_total"] == 220.0
    assert len(seen) == 1
    assert seen[0]["game_date"].max() == pd.Timestamp("2026-01-01")
    assert len(seen[0]) == 1
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
            "expected_possessions": 100.0,
            "league_off_eff_prior": 110.0,
            "league_pace_prior": 100.0,
            "expected_home_efficiency": 115.0,
            "expected_away_efficiency": 110.0,
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
            "expected_possessions": 100.0,
            "league_off_eff_prior": 110.0,
            "league_pace_prior": 100.0,
            "expected_home_efficiency": 115.0,
            "expected_away_efficiency": 110.0,
            "actual_total": 205.0,
            "projected_total": 210.0,
            "league_environment_total": 210.0,
            "efficiency_only_total": 210.0,
            "pace_only_total": 210.0,
            "max_history_date": pd.Timestamp("2026-01-02"),
        },
    ])
    names = {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"}
    results, diagnostics = build_paired_predictions(
        raw_games(),
        names,
        v4_features=features,
        predictor=lambda **kwargs: {"projected_total": 210.0},
        v2_history=build_score_history(raw_games(), names),
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



def test_relevant_team_history_is_equivalent_to_full_prior_score_history():
    history = build_score_history(
        raw_games(),
        {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"},
    )
    prior = history.loc[history["game_date"] < pd.Timestamp("2026-01-03")].copy()
    relevant = prior.loc[
        prior["home_team"].isin(["Team A", "Team B"])
        | prior["away_team"].isin(["Team A", "Team B"])
    ].copy()

    for frame in (prior, relevant):
        frame.attrs["_strict_history_cutoff_date"] = "2026-01-03"
        frame.attrs["_team_names_normalized"] = True

    full_prediction = calculate_matchup_points(
        "Team A", "Team B", history_df=prior, game_date="2026-01-03"
    )
    relevant_prediction = calculate_matchup_points(
        "Team A", "Team B", history_df=relevant, game_date="2026-01-03"
    )
    assert relevant_prediction["projected_total"] == pytest.approx(
        full_prediction["projected_total"]
    )
    assert relevant_prediction["h2h_adjustment"] == pytest.approx(
        full_prediction["h2h_adjustment"]
    )
    assert relevant_prediction["rest_adjustment"] == pytest.approx(
        full_prediction["rest_adjustment"]
    )



def test_v2_score_history_rebuilds_totals_and_fails_closed_on_duplicate_games():
    source = build_score_history(
        raw_games(),
        {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"},
    )
    prepared, diagnostics = prepare_v2_score_history(source)
    assert len(prepared) == 4
    assert prepared["total_score"].tolist() == [195.0, 200.0, 225.0, 205.0]
    assert diagnostics["v2_score_history_rows"] == 4
    assert diagnostics["v2_score_history_duplicate_rows"] == 0

    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate game rows"):
        prepare_v2_score_history(duplicate)



def test_sufficient_v2_history_matches_full_history_forecast():
    names = {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"}
    history, _diagnostics = prepare_v2_score_history(build_score_history(raw_games(), names))
    team_rows = {}
    for record in history.to_dict("records"):
        team_rows.setdefault(record["home_team"], []).append(record)
        team_rows.setdefault(record["away_team"], []).append(record)
    team_frames = {}
    team_dates = {}
    for team, records in team_rows.items():
        frame = pd.DataFrame(records).sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
        team_frames[team] = frame
        team_dates[team] = pd.to_datetime(frame["game_date"]).to_numpy(dtype="datetime64[ns]")

    pair_rows = {}
    for record in history.to_dict("records"):
        pair = tuple(sorted((record["home_team"], record["away_team"])))
        pair_rows.setdefault(pair, []).append(record)
    pair_frames = {}
    pair_dates = {}
    for pair, records in pair_rows.items():
        frame = pd.DataFrame(records).sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
        pair_frames[pair] = frame
        pair_dates[pair] = pd.to_datetime(frame["game_date"]).to_numpy(dtype="datetime64[ns]")

    target = pd.Timestamp("2026-01-03")
    full = history.loc[history["game_date"] < target].copy()
    full.attrs["_strict_history_cutoff_date"] = target.date().isoformat()
    full.attrs["_team_names_normalized"] = True
    selected = select_v2_relevant_history(
        team_frames, team_dates, pair_frames, pair_dates,
        "Team A", "Team B", target,
    )

    full_prediction = calculate_matchup_points(
        "Team A", "Team B", history_df=full, game_date=target
    )
    selected_prediction = calculate_matchup_points(
        "Team A", "Team B", history_df=selected, game_date=target
    )
    assert selected_prediction["projected_total"] == pytest.approx(
        full_prediction["projected_total"]
    )
    assert selected_prediction["h2h_adjustment"] == pytest.approx(
        full_prediction["h2h_adjustment"]
    )
    assert selected_prediction["rest_adjustment"] == pytest.approx(
        full_prediction["rest_adjustment"]
    )
    assert len(selected) <= 30

def test_sufficient_history_selector_normalizes_clippers_alias():
    dates = pd.date_range("2024-10-01", periods=12, freq="2D")
    rows = []
    for i, date in enumerate(dates):
        opponent = "Dallas Mavericks" if i in {0, 4, 8} else f"Opponent {i}"
        rows.append({
            "game_date": date,
            "home_team": "Los Angeles Clippers",
            "away_team": opponent,
            "home_score": 108 + i,
            "away_score": 99 + (i % 5),
            "is_neutral_site": False,
        })
    history, _diagnostics = prepare_v2_score_history(pd.DataFrame(rows))

    team_rows = {}
    for record in history.to_dict("records"):
        team_rows.setdefault(record["home_team"], []).append(record)
        team_rows.setdefault(record["away_team"], []).append(record)
    team_frames, team_dates = {}, {}
    for team, records in team_rows.items():
        frame = pd.DataFrame(records).sort_values(
            ["game_date", "home_team", "away_team"]
        ).reset_index(drop=True)
        team_frames[team] = frame
        team_dates[team] = pd.to_datetime(frame["game_date"]).to_numpy(
            dtype="datetime64[ns]"
        )

    pair_rows = {}
    for record in history.to_dict("records"):
        pair = tuple(sorted((record["home_team"], record["away_team"])))
        pair_rows.setdefault(pair, []).append(record)
    pair_frames, pair_dates = {}, {}
    for pair, records in pair_rows.items():
        frame = pd.DataFrame(records).sort_values(
            ["game_date", "home_team", "away_team"]
        ).reset_index(drop=True)
        pair_frames[pair] = frame
        pair_dates[pair] = pd.to_datetime(frame["game_date"]).to_numpy(
            dtype="datetime64[ns]"
        )

    target = pd.Timestamp("2024-11-01")
    full = history.loc[history["game_date"] < target].copy()
    full.attrs["_strict_history_cutoff_date"] = target.date().isoformat()
    full.attrs["_team_names_normalized"] = True
    selected = select_v2_relevant_history(
        team_frames, team_dates, pair_frames, pair_dates,
        "LA Clippers", "Dallas Mavericks", target,
    )

    assert "Los Angeles Clippers" in set(selected["home_team"]) | set(
        selected["away_team"]
    )
    full_prediction = calculate_matchup_points(
        "LA Clippers", "Dallas Mavericks",
        history_df=full, game_date=target,
    )
    selected_prediction = calculate_matchup_points(
        "LA Clippers", "Dallas Mavericks",
        history_df=selected, game_date=target,
    )
    assert selected_prediction["projected_total"] == pytest.approx(
        full_prediction["projected_total"]
    )
    assert selected_prediction["h2h_adjustment"] == pytest.approx(
        full_prediction["h2h_adjustment"]
    )
    assert selected_prediction["rest_adjustment"] == pytest.approx(
        full_prediction["rest_adjustment"]
    )
