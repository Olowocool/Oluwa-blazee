import pandas as pd
import pytest

from injury_impact_engine import (
    PlayerHistoryIndex,
    calculate_team_offensive_injury_impact,
    fit_injury_adjustment_beta,
    fit_status_absence_probabilities,
)


def test_status_absence_probabilities_are_learned_from_supplied_development_rows():
    snapshot = pd.DataFrame([
        {"game_id": "G1", "person_id": 1, "current_status": "Out", "match_method": "exact"},
        {"game_id": "G2", "person_id": 2, "current_status": "Out", "match_method": "exact"},
        {"game_id": "G3", "person_id": 3, "current_status": "Questionable", "match_method": "exact"},
        {"game_id": "G4", "person_id": 4, "current_status": "Questionable", "match_method": "league"},
    ])
    box = pd.DataFrame([
        {"game_id": "G3", "person_id": 3, "minutes": "22:00"},
    ])

    model = fit_status_absence_probabilities(snapshot, box, prior_strength=0)

    assert model["rows_used"] == 3
    assert model["probabilities"]["Out"] == 1.0
    assert model["probabilities"]["Questionable"] == 0.0
    assert model["by_status"]["Out"]["rows"] == 2


def _build_player_history():
    dates = pd.date_range("2025-10-01", periods=11, freq="D")
    schedule_rows = []
    box_rows = []
    minutes = [35, 32, 30, 28, 25, 12, 10, 8]
    ppm = [1.0, 0.9, 0.8, 0.75, 0.7, 0.5, 0.5, 0.5]
    for game_no, game_date in enumerate(dates):
        game_id = f"G{game_no}"
        schedule_rows.append({"game_id": game_id, "game_date": game_date})
        for player_index, player_minutes in enumerate(minutes):
            player_id = 100 + player_index
            rate = ppm[player_index]
            box_rows.append({
                "game_id": game_id,
                "team_id": 1,
                "person_id": player_id,
                "minutes": f"{player_minutes}:00",
                "points": round(player_minutes * rate),
            })
    return pd.DataFrame(box_rows), pd.DataFrame(schedule_rows), dates


def test_player_history_uses_only_games_strictly_before_target_date():
    box, schedule, dates = _build_player_history()
    index = PlayerHistoryIndex(box, schedule, window=10, minimum_games=5, max_stale_days=60)

    # G10 is played on the target date and contains deliberately extreme stats.
    target_date = dates[-1]
    form = index.get_player_form(1, 100, target_date)

    assert form is not None
    assert form.last_game_date == dates[-2]
    assert form.expected_minutes == pytest.approx(35.0)
    assert form.points_per_minute == pytest.approx(1.0)
    assert form.games_in_window == 10


def test_offensive_injury_impact_nets_missing_scoring_against_bench_rate():
    box, schedule, dates = _build_player_history()
    index = PlayerHistoryIndex(box, schedule, window=10, minimum_games=5, max_stale_days=60)
    target_date = dates[-1]
    snapshot = pd.DataFrame([{
        "team_id": 1,
        "person_id": 100,
        "current_status": "Out",
        "match_method": "exact",
    }, {
        "team_id": 1,
        "person_id": 105,
        "current_status": "Available",
        "match_method": "exact",
    }])
    status_model = {
        "probabilities": {
            "Out": 1.0, "Doubtful": 0.9, "Questionable": 0.5,
            "Probable": 0.1, "Available": 0.0,
        },
        "global_absence_probability": 0.25,
    }

    result = calculate_team_offensive_injury_impact(
        1, target_date, snapshot, index, status_model
    )

    assert result["reported_players"] == 2
    assert result["injury_candidate_players"] == 1
    assert result["players_used"] == 1
    assert result["replacement_pool_size"] == 3
    assert result["replacement_ppm"] == pytest.approx(0.5)
    assert result["expected_missing_points"] == pytest.approx(35.0)
    assert result["expected_replacement_points"] == pytest.approx(17.5)
    assert result["net_offensive_loss"] == pytest.approx(17.5)


def test_injury_beta_is_fitted_nonnegative_and_shrunk():
    model = fit_injury_adjustment_beta(
        actual_minus_baseline=[-5, -10, -15],
        net_offensive_loss=[5, 10, 15],
        ridge=0,
    )
    assert model["beta"] == pytest.approx(1.0)

    no_gain = fit_injury_adjustment_beta(
        actual_minus_baseline=[5, 10],
        net_offensive_loss=[5, 10],
        ridge=0,
    )
    assert no_gain["beta"] == 0.0
