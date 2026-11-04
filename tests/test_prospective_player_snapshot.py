import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.prospective_player_snapshot import (
    immutable_snapshot, latest_draftkings_quotes, map_odds_teams,
    require_complete_injury_report, validate_pregame_history,
    validate_player_sources,
)


class ProspectiveSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.now = "2026-11-04T22:00:00Z"

    def odds(self):
        return pd.DataFrame([{
            "game_key":"abc","game_date":"2026-11-05",
            "commence_time_utc":"2026-11-06T00:00:00Z",
            "away_team":"LA Clippers","home_team":"Los Angeles Lakers",
            "bookmaker":"DraftKings","total_line":"221.5",
            "over_odds_american":"-110","under_odds_american":"-110",
            "published_as_of_utc":"2026-11-04T21:55:00Z",
            "observed_at_utc":"2026-11-04T21:58:00Z",
        }])

    def test_latest_dk_quote_keeps_paired_prices(self):
        q=latest_draftkings_quotes(self.odds(), self.now)
        self.assertEqual(len(q),1)
        self.assertEqual(float(q.iloc[0].total_line),221.5)
        self.assertEqual(float(q.iloc[0].over_odds_american),-110)

    def test_stale_market_fails_closed(self):
        d=self.odds(); d.loc[0,"observed_at_utc"]="2026-11-04T20:00:00Z"
        with self.assertRaises(ValueError): latest_draftkings_quotes(d,self.now)

    def test_team_alias_mapping(self):
        teams=pd.DataFrame([{"team_id":"1","full_name":"Los Angeles Clippers"},{"team_id":"2","full_name":"Los Angeles Lakers"}])
        self.assertEqual(map_odds_teams("LA Clippers","Los Angeles Lakers",teams),("1","2"))

    def test_same_day_score_rows_excluded(self):
        h=pd.DataFrame({"game_date":["2026-11-03","2026-11-04","2026-11-05"]})
        p=validate_pregame_history(h,"2026-11-04")
        self.assertEqual(p.game_date.max(),pd.Timestamp("2026-11-03"))

    def test_injury_report_must_be_complete_and_hashed(self):
        m=pd.DataFrame([{"game_date":"2026-11-05","status":"complete","snapshot_time_utc":"2026-11-05T14:00:00Z","sha256":"abc"}])
        self.assertEqual(require_complete_injury_report(m,"2026-11-05")["sha256"],"abc")
        m.loc[0,"status"]="missing_or_incomplete"
        with self.assertRaises(ValueError): require_complete_injury_report(m,"2026-11-05")

    def test_source_same_day_rows_rejected(self):
        g=pd.DataFrame({"game_id":["g1"],"game_date":["2026-11-03"]})
        x=pd.DataFrame({"game_id":["g1"],"game_date":["2026-11-04"]})
        with self.assertRaises(ValueError): validate_player_sources(g,x,x,x,x,"2026-11-04")

    def test_content_addressed_snapshot_is_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            p=immutable_snapshot({"a":1},Path(td))
            self.assertTrue(p.exists())
            self.assertEqual(immutable_snapshot({"a":1},Path(td)),p)
            self.assertEqual(len(list(Path(td).glob("*.json"))),1)


if __name__ == "__main__":
    unittest.main()
