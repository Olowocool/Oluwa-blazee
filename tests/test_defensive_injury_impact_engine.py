import pandas as pd

from defensive_injury_impact_engine import (
    calculate_team_defensive_injury_impact,
    fit_matchup_defensive_rapm,
)


def test_matchup_rapm_returns_centered_defender_scores():
    rows = []
    # Two offensive players, two defenders. Defender 10 consistently allows
    # fewer points per possession than defender 20.
    for off in (1, 2, 3):
        rows.extend([
            {"off_player_id": off, "def_player_id": 10, "partial_poss": 100, "player_pts": 90},
            {"off_player_id": off, "def_player_id": 20, "partial_poss": 100, "player_pts": 110},
        ])
    result = fit_matchup_defensive_rapm(pd.DataFrame(rows), min_possessions=25, ridge_alpha=1)
    assert result["status"] == "ok"
    assert result["defenders"] == 2
    assert result["scores"][10] > result["scores"][20]
    assert abs(sum(result["scores"].values())) < 1e-9


def test_defensive_absence_uses_replacement_and_status_probability():
    class History:
        def get_team_rotation(self, team_id, target_date, availability_by_player=None):
            return [
                {"person_id": 1, "expected_minutes": 30, "available_probability": 0.0},
                {"person_id": 2, "expected_minutes": 24, "available_probability": 1.0},
                {"person_id": 3, "expected_minutes": 18, "available_probability": 1.0},
                {"person_id": 4, "expected_minutes": 12, "available_probability": 1.0},
                {"person_id": 5, "expected_minutes": 8, "available_probability": 1.0},
                {"person_id": 6, "expected_minutes": 6, "available_probability": 1.0},
            ]

        def get_player_form(self, team_id, person_id, target_date):
            class Form:
                expected_minutes = 30
            return Form()

    snapshot = pd.DataFrame([{
        "team_id": 99, "person_id": 1, "current_status": "Out", "match_method": "exact"
    }])
    result = calculate_team_defensive_injury_impact(
        99, "2025-01-01", snapshot, History(),
        {"probabilities": {"Out": 1.0}, "global_absence_probability": 0.25},
        {1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    )
    assert result["players_used"] == 1
    assert result["net_defensive_loss"] > 0
    assert result["replacement_pool_size"] == 1


def test_empty_matchups_are_flagged_not_fabricated():
    result = fit_matchup_defensive_rapm(pd.DataFrame(columns=[
        "off_player_id", "def_player_id", "partial_poss", "player_pts"
    ]))
    assert result["status"] == "insufficient_matchups"
    assert result["scores"] == {}
