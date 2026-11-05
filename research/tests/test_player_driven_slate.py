from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from research.player_driven_slate import (
    build_player_scoring_projection,
    fetch_scheduled_nba_games,
)


class PlayerDrivenSlateTests(unittest.TestCase):
    def test_schedule_fetch_returns_only_pregame_fixtures(self):
        payload = {
            "events": [
                {
                    "id": "game-1",
                    "date": "2026-11-06T00:00:00Z",
                    "status": {"type": {"state": "pre", "name": "Scheduled", "completed": False}},
                    "competitions": [{"competitors": [
                        {"homeAway": "home", "team": {"displayName": "Boston Celtics"}},
                        {"homeAway": "away", "team": {"displayName": "New York Knicks"}},
                    ]}],
                },
                {
                    "id": "game-2",
                    "date": "2026-11-06T02:00:00Z",
                    "status": {"type": {"state": "post", "name": "Final", "completed": True}},
                    "competitions": [{"competitors": [
                        {"homeAway": "home", "team": {"displayName": "Denver Nuggets"}},
                        {"homeAway": "away", "team": {"displayName": "Phoenix Suns"}},
                    ]}],
                },
            ]
        }
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        with patch("research.player_driven_slate.requests.get", return_value=response) as get:
            games = fetch_scheduled_nba_games("2026-11-06")
        self.assertEqual(len(games), 1)
        self.assertEqual(games.iloc[0].game_key, "game-1")
        self.assertEqual(games.iloc[0].home_team, "Boston Celtics")
        self.assertEqual(games.iloc[0].away_team, "New York Knicks")
        self.assertEqual(get.call_args.kwargs["params"]["dates"], "20261106")

    def _players(self, eligible_count: int) -> pd.DataFrame:
        rows = []
        for game_id, team_id in [("g1", "home"), ("g1", "away")]:
            for i in range(10):
                eligible = i < eligible_count
                rows.append({
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": f"{team_id}-{i}",
                    "expected_minutes": 24.0,
                    "baseline_minutes": 24.0,
                    "status": "Available",
                    "availability_probability": 1.0,
                    "profile_eligible": eligible,
                    "pts_per36": 18.0 if eligible else np.nan,
                    "o_rapm": 1.0,
                    "d_rapm": -0.5,
                })
        return pd.DataFrame(rows)

    def test_player_scoring_formula_reconciles_and_applies_pace_and_coverage(self):
        target_games = pd.DataFrame([{
            "game_id": "g1", "home_team_id": "home", "away_team_id": "away"
        }])
        pace = {"g1": {
            "available": True,
            "expected_matchup_pace_l10": 105.0,
            "league_pace_prior": 100.0,
        }}
        summary, detail = build_player_scoring_projection(
            self._players(9), target_games, pace
        )
        row = summary.iloc[0]
        # 9 profiled players x (24 min x 18 PTS/36) x (240/216) x 1.05
        self.assertAlmostEqual(row.home_player_score, 126.0, places=6)
        self.assertAlmostEqual(row.away_player_score, 126.0, places=6)
        self.assertAlmostEqual(row.player_driven_total, 252.0, places=6)
        self.assertAlmostEqual(row.home_profiled_minutes_share, 0.9, places=6)
        self.assertAlmostEqual(row.home_profile_coverage_scale, 240.0 / 216.0, places=6)
        self.assertEqual(len(detail), 20)
        self.assertAlmostEqual(
            detail.loc[detail.team_id.eq("home"), "player_expected_points"].sum(),
            row.home_player_score,
            places=6,
        )

    def test_inadequate_profile_coverage_withholds_candidate_total(self):
        target_games = pd.DataFrame([{
            "game_id": "g1", "home_team_id": "home", "away_team_id": "away"
        }])
        pace = {"g1": {
            "available": True,
            "expected_matchup_pace_l10": 100.0,
            "league_pace_prior": 100.0,
        }}
        summary, _ = build_player_scoring_projection(
            self._players(8), target_games, pace
        )
        self.assertTrue(pd.isna(summary.iloc[0].player_driven_total))
        self.assertIn("withheld", summary.iloc[0].forecast_status)


if __name__ == "__main__":
    unittest.main()
