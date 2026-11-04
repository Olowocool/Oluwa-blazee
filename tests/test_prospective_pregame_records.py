import unittest

import pandas as pd

from research.build_prospective_pregame_records import build_pregame_records


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
        self.state = {
            "team_names": pd.DataFrame([
                {"season": 2027, "team_id": "LAC", "full_name": "Los Angeles Clippers"},
                {"season": 2027, "team_id": "LAL", "full_name": "Los Angeles Lakers"},
            ]),
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

    def test_joins_market_report_and_both_team_rotations(self):
        records = build_pregame_records(
            self.quotes, self.state, self.rotation, self.target, "2026-11-05T20:05:00Z"
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["market"]["total_line"], 221.5)
        self.assertEqual(record["official_injury_report"]["sha256"], "report-hash")
        self.assertEqual(record["player_data"]["team_rotation"]["away"]["team_id"], "LAC")
        self.assertEqual(record["player_data"]["team_rotation"]["home"]["players"][0]["player_id"], "p2")
        self.assertFalse(record["forecast"]["forecast_emitted"])
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
