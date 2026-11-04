import unittest

import pandas as pd

from research.prospective_player_data import normalize_source_frames


class ProspectivePlayerDataTests(unittest.TestCase):
    def fixtures(self):
        games = pd.DataFrame([
            {"season": 2026, "game_id": "old", "game_date": "2026-04-01",
             "game_type": "regular_season", "home_team_id": "H", "away_team_id": "A",
             "neutral_site": False},
            {"season": 2027, "game_id": "today", "game_date": "2026-11-05",
             "game_type": "regular_season", "home_team_id": "H", "away_team_id": "A",
             "neutral_site": False},
        ])
        team = pd.DataFrame([
            {"game_id": "old", "game_date": "2026-04-01", "team_id": "H", "opp_team_id": "A"},
            {"game_id": "old", "game_date": "2026-04-01", "team_id": "A", "opp_team_id": "H"},
        ])
        players = pd.DataFrame([
            {"game_id": "old", "game_date": "2026-04-01", "team_id": "H", "player_id": "p1"},
        ])
        names = pd.DataFrame([{"season": 2027, "player_id": "p1", "name": "Player One"}])
        teams = pd.DataFrame([{"season": 2027, "team_id": "H", "full_name": "Home"}])
        rotations = pd.DataFrame([{
            "season": 2026, "game_id": "old", "team_id": "H", "player_id": "p1",
            "period": 1, "in_time": 0, "out_time": 100, "plus_minus": 0,
            "matches_box_score": True,
        }])
        events = pd.DataFrame([{
            "game_id": "old", "period": 1, "time": 50, "team_id": "H", "pts": 2,
        }])
        return games, team, players, names, teams, rotations, events

    def test_future_and_target_date_game_rows_are_excluded(self):
        frames = normalize_source_frames(*self.fixtures(), "2026-11-05")
        games, team, players, _, _, rotations, events = frames
        self.assertEqual(games.game_id.tolist(), ["old"])
        self.assertEqual(set(team.game_id), {"old"})
        self.assertEqual(set(players.game_id), {"old"})
        self.assertEqual(set(rotations.game_id), {"old"})
        self.assertEqual(set(events.game_id), {"old"})

    def test_same_day_team_log_is_rejected(self):
        games, team, players, names, teams, rotations, events = self.fixtures()
        team.loc[0, "game_date"] = "2026-11-05"
        with self.assertRaisesRegex(ValueError, "same-day/future"):
            normalize_source_frames(games, team, players, names, teams,
                                    rotations, events, "2026-11-05")

    def test_unknown_player_game_id_is_rejected(self):
        games, team, players, names, teams, rotations, events = self.fixtures()
        players.loc[0, "game_id"] = "not-a-game"
        with self.assertRaisesRegex(ValueError, "unknown game IDs"):
            normalize_source_frames(games, team, players, names, teams,
                                    rotations, events, "2026-11-05")

    def test_playoff_rows_are_excluded_from_regular_season_history(self):
        games, team, players, names, teams, rotations, events = self.fixtures()
        playoff = games.iloc[[0]].copy()
        playoff["game_id"] = "playoff"
        playoff["game_type"] = "playoffs"
        games = pd.concat([games, playoff], ignore_index=True)
        playoff_team = team.iloc[[0]].copy()
        playoff_team["game_id"] = "playoff"
        team = pd.concat([team, playoff_team], ignore_index=True)
        frames = normalize_source_frames(games, team, players, names, teams,
                                          rotations, events, "2026-11-05")
        self.assertEqual(set(frames[1].game_id), {"old"})

    def test_duplicate_completed_game_id_is_rejected(self):
        games, team, players, names, teams, rotations, events = self.fixtures()
        games = pd.concat([games, games.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate game IDs"):
            normalize_source_frames(games, team, players, names, teams,
                                    rotations, events, "2026-11-05")


if __name__ == "__main__":
    unittest.main()
