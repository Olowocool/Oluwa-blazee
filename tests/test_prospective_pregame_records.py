import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.build_prospective_pregame_records import (
    build_pregame_records, build_v5_score_history,
)


class ProspectivePregameRecordTests(unittest.TestCase):
    def setUp(self):
        self.target = "2026-11-05"
        self.quotes = pd.DataFrame([{
            "game_key": "fixture-1",
            "game_date": self.target,
            "commence_time_utc": "2026-11-06T00:00:00Z",
            "away_team": "LA Clippers",
            "home_team": "Los Angeles Lakers",
            "bookmaker": "DraftKings",
            "total_line": 221.5,
            "over_odds_american": -110,
            "under_odds_american": -110,
            "published_as_of_utc": "2026-11-05T20:00:00Z",
            "observed_at_utc": "2026-11-05T20:02:00Z",
            "observation_age_minutes": 3.0,
        }])
        teams = pd.DataFrame([
            {"season": 2027, "team_id": "LAC", "full_name": "Los Angeles Clippers"},
            {"season": 2027, "team_id": "LAL", "full_name": "Los Angeles Lakers"},
        ])
        self.state = {
            "team_names": teams,
            "team_names_all": teams,
            "games": pd.DataFrame([
                {"season": 2027, "game_id": "prior-game", "game_date": "2026-11-04",
                 "home_team_id": "LAL", "away_team_id": "LAC"},
                {"season": 2027, "game_id": "target-game", "game_date": self.target,
                 "home_team_id": "LAC", "away_team_id": "LAL"},
            ]),
            "team_logs": pd.DataFrame([
                {"season": 2027, "game_id": "prior-game", "team_id": "LAL", "pts": 112},
                {"season": 2027, "game_id": "prior-game", "team_id": "LAC", "pts": 108},
                {"season": 2027, "game_id": "target-game", "team_id": "LAC", "pts": 120},
                {"season": 2027, "game_id": "target-game", "team_id": "LAL", "pts": 115},
            ]),
            "v5_score_history": pd.DataFrame([{
                "game_date": "2026-11-04", "home_team": "Los Angeles Lakers",
                "away_team": "Los Angeles Clippers", "home_score": 112, "away_score": 108,
            }]),
            "diagnostics": {
                "source_seasons": [2026, 2027],
                "rapm_max_history_date": "2026-11-04",
                "rapm_history_segments": 12000,
            },
        }
        self.rotation = {
            "manifest": pd.DataFrame([{
                "game_date": self.target, "status": "complete",
                "snapshot_time_utc": "2026-11-05T19:00:00Z", "sha256": "report-hash",
            }]),
            "report_metadata": {
                "snapshot_time_utc": "2026-11-05T19:00:00Z",
                "sha256": "report-hash",
            },
            "projections": pd.DataFrame([
                {"game_id":"fixture-1","team_id":team,"player_id":f"{team}-{i}","rotation_rank":i,
                 "baseline_minutes":30.0,"expected_minutes":30.0,"availability_probability":1.0,
                 "status":"NOT_REPORTED","o_rapm":1.2,"d_rapm":-0.2,
                 "o_delta_contribution":0.0,"d_delta_contribution":0.0}
                for team in ("LAC", "LAL") for i in range(1, 9)
            ]),
            "summaries": [
                {"game_id":"fixture-1","team_id":"LAC","features":{"off_delta":0.0}},
                {"game_id":"fixture-1","team_id":"LAL","features":{"off_delta":0.0}},
            ],
        }

    def test_v5_history_uses_base_patches_and_only_prior_current_season_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "historical.csv"
            base.write_text(
                "game_date,home_team,away_team,home_score,away_score\n"
                "2026-10-01,Boston Celtics,Chicago Bulls,110,100\n",
                encoding="utf-8",
            )
            patches = root / "patches"
            patches.mkdir()
            (patches / "score_history_patch_test.csv").write_text(
                "game_date,home_team,away_team,home_score,away_score\n"
                "2026-11-03,Miami Heat,Orlando Magic,105,101\n",
                encoding="utf-8",
            )
            history = build_v5_score_history(
                self.state, self.target, base_history_path=base, patch_dir=patches
            )
            self.assertEqual(len(history), 3)
            self.assertEqual(history.game_date.max(), pd.Timestamp("2026-11-04"))
            self.assertNotIn(pd.Timestamp(self.target), set(history.game_date))
            current = history[history.game_date.eq(pd.Timestamp("2026-11-04"))].iloc[0]
            self.assertEqual(current.home_score, 112)
            self.assertEqual(current.away_score, 108)

    def test_verified_score_patch_overrides_conflicting_public_feed(self):
        state = {
            "games": pd.DataFrame([{
                "season": 2027, "game_id": "okc-dal-20261028", "game_date": "2026-10-28",
                "home_team_id": "DAL", "away_team_id": "OKC",
            }]),
            "team_logs": pd.DataFrame([
                {"season": 2027, "game_id": "okc-dal-20261028", "team_id": "DAL", "pts": 118},
                {"season": 2027, "game_id": "okc-dal-20261028", "team_id": "OKC", "pts": 96},
            ]),
            "team_names": pd.DataFrame([
                {"season": 2027, "team_id": "DAL", "full_name": "Dallas Mavericks"},
                {"season": 2027, "team_id": "OKC", "full_name": "Oklahoma City Thunder"},
            ]),
            "team_names_all": pd.DataFrame([
                {"season": 2027, "team_id": "DAL", "full_name": "Dallas Mavericks"},
                {"season": 2027, "team_id": "OKC", "full_name": "Oklahoma City Thunder"},
            ]),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "historical.csv"
            base.write_text(
                "game_date,home_team,away_team,home_score,away_score\n",
                encoding="utf-8",
            )
            patches = root / "research" / "data"
            patches.mkdir(parents=True)
            (patches / "score_history_patch_verified.csv").write_text(
                "game_date,home_team,away_team,home_score,away_score\n"
                "2026-10-28,Dallas Mavericks,Oklahoma City Thunder,120,96\n",
                encoding="utf-8",
            )
            history = build_v5_score_history(
                state, "2026-11-05", base_history_path=base, patch_dir=patches
            )
            game = history.iloc[0]
            self.assertEqual(game.home_score, 120)
            self.assertEqual(game.away_score, 96)

    def test_joins_market_report_and_both_team_rotations(self):
        records = build_pregame_records(
            self.quotes, self.state, self.rotation, self.target, "2026-11-05T20:05:00Z"
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["market"]["total_line"], 221.5)
        self.assertEqual(record["official_injury_report"]["sha256"], "report-hash")
        self.assertEqual(record["player_data"]["team_rotation"]["away"]["team_id"], "LAC")
        self.assertEqual(record["player_data"]["team_rotation"]["home"]["players"][0]["player_id"], "LAL-1")
        self.assertGreater(record["forecast"]["v5_baseline_total"], 0)
        self.assertEqual(record["forecast"]["v5_history_cutoff"], self.target)
        self.assertEqual(record["forecast"]["v5_history_games"], 1)
        self.assertFalse(record["forecast"]["player_adjusted_forecast_emitted"])
        self.assertIsNone(record["forecast"]["player_adjusted_total"])

    def test_rejects_injury_report_after_tip(self):
        self.rotation["report_metadata"]["snapshot_time_utc"] = "2026-11-06T00:01:00Z"
        with self.assertRaisesRegex(ValueError, "not pregame"):
            build_pregame_records(
                self.quotes, self.state, self.rotation, self.target, "2026-11-05T20:05:00Z"
            )

    def test_rejects_missing_team_rotation(self):
        self.rotation["projections"] = self.rotation["projections"][
            self.rotation["projections"].team_id.eq("LAC")
        ]
        with self.assertRaisesRegex(ValueError, "Missing home rotation"):
            build_pregame_records(
                self.quotes, self.state, self.rotation, self.target, "2026-11-05T20:05:00Z"
            )

    def test_rejects_duplicate_market_fixture(self):
        self.quotes = pd.concat([self.quotes, self.quotes], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate market"):
            build_pregame_records(
                self.quotes, self.state, self.rotation, self.target, "2026-11-05T20:05:00Z"
            )


if __name__ == "__main__":
    unittest.main()
