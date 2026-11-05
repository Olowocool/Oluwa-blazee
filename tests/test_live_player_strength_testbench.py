import pandas as pd
import pytest

from research.live_player_strength_testbench import _expected_possessions_from_v4_history


def _team_rows(season, team_id, start, count, possessions):
    rows = []
    for i in range(count):
        rows.append({
            "season": season,
            "game_id": f"{season}-{team_id}-{i}",
            "game_date": pd.Timestamp(start) + pd.Timedelta(days=i),
            "team_id": str(team_id),
            "fga": possessions - 10,
            "fta": 0,
            "oreb": 0,
            "tov": 10,
        })
    return rows


def test_v4_expected_possessions_uses_strictly_prior_team_history():
    rows = []
    rows += _team_rows(2026, "H", "2026-01-01", 5, 90)
    rows += _team_rows(2026, "A", "2026-01-01", 5, 110)
    rows += _team_rows(2027, "H", "2026-10-20", 5, 100)
    rows += _team_rows(2027, "A", "2026-10-20", 5, 100)
    raw = pd.DataFrame(rows)
    result = _expected_possessions_from_v4_history(
        raw, "2026-11-01", "H", "A"
    )
    assert result == pytest.approx(100.0)


def test_v4_expected_possessions_fails_before_five_prior_games():
    rows = []
    rows += _team_rows(2026, "H", "2026-01-01", 5, 90)
    rows += _team_rows(2026, "A", "2026-01-01", 5, 110)
    rows += _team_rows(2027, "H", "2026-10-20", 4, 100)
    rows += _team_rows(2027, "A", "2026-10-20", 5, 100)
    with pytest.raises(ValueError, match="warm-up incomplete"):
        _expected_possessions_from_v4_history(
            pd.DataFrame(rows), "2026-11-01", "H", "A"
        )


def test_v4_expected_possessions_excludes_target_date_rows():
    rows = []
    rows += _team_rows(2026, "H", "2026-01-01", 5, 90)
    rows += _team_rows(2026, "A", "2026-01-01", 5, 110)
    rows += _team_rows(2027, "H", "2026-10-20", 5, 100)
    rows += _team_rows(2027, "A", "2026-10-20", 5, 100)
    rows += _team_rows(2027, "H", "2026-11-01", 1, 999)
    rows += _team_rows(2027, "A", "2026-11-01", 1, 999)
    result = _expected_possessions_from_v4_history(
        pd.DataFrame(rows), "2026-11-01", "H", "A"
    )
    assert result == pytest.approx(100.0)
