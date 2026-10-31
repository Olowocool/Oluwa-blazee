from pathlib import Path
import sys

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.v4_team_strength_season_audit import (
    build_league_season_environment,
    build_team_season_profiles,
    prepare_team_game_data,
)


def synthetic_games():
    rows = []
    for game_no in range(11):
        game_id = f"g{game_no:02d}"
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=game_no)
        a_points = 130 if game_no == 10 else 110
        b_points = 110 if game_no == 10 else 100
        common = {
            "season": 2026,
            "game_id": game_id,
            "game_date": date,
            "source_poss": 100.8,
            "source_pace": 100.0,
            "fga": 85,
            "fgm": 41,
            "fg3m": 10,
            "fg3a": 30,
            "fta": 20,
            "oreb": 8,
            "tov": 15,
            "neutral_site": False,
            "game_type": "regular_season",
        }
        rows.append({
            **common,
            "team_id": "A",
            "team_name": "Team A",
            "opp_team_id": "B",
            "opponent_name": "Team B",
            "pts": a_points,
            "home": True,
        })
        rows.append({
            **common,
            "team_id": "B",
            "team_name": "Team B",
            "opp_team_id": "A",
            "opponent_name": "Team A",
            "pts": b_points,
            "home": False,
        })
    return pd.DataFrame(rows)


def test_true_possessions_and_opponent_pairing_are_calculated():
    paired = prepare_team_game_data(synthetic_games())
    assert len(paired) == 22
    assert paired["formula_poss"].iloc[0] == pytest.approx(100.8)
    assert paired["opp_pts"].iloc[0] == 100
    assert paired["game_total"].iloc[0] == 210
    assert paired["margin"].iloc[0] == 10


def test_team_season_profile_contains_full_season_and_last_ten_form():
    paired = prepare_team_game_data(synthetic_games())
    profiles = build_team_season_profiles(paired)
    team_a = profiles.loc[profiles["team_id"] == "A"].iloc[0]

    assert team_a["games"] == 11
    assert team_a["wins"] == 11
    assert team_a["losses"] == 0
    assert team_a["points_for_per_game"] == pytest.approx((10 * 110 + 130) / 11)
    assert team_a["last10_points_for_per_game"] == pytest.approx(112.0)
    assert team_a["last10_games"] == 10
    assert team_a["last10_true_possessions_per_game"] == pytest.approx(100.8)
    assert "schedule_adj_net_strength_effect_per100" in profiles.columns


def test_league_season_environment_reports_game_totals_and_possessions():
    paired = prepare_team_game_data(synthetic_games())
    league = build_league_season_environment(paired)
    season = league.loc[league["season"] == 2026].iloc[0]

    expected_total = (10 * 210 + 240) / 11
    assert season["games"] == 11
    assert season["average_game_total"] == pytest.approx(expected_total)
    assert season["average_true_possessions"] == pytest.approx(100.8)


def test_missing_opponent_record_fails_closed():
    raw = synthetic_games().iloc[:-1].copy()
    with pytest.raises(ValueError, match="do not have exactly one opponent row"):
        prepare_team_game_data(raw)
