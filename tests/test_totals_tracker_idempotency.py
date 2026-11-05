import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from totals_tracker import save_totals_pick


class TotalsTrackerIdempotencyTests(unittest.TestCase):
    def _save(self, event_id, **overrides):
        values = {
            "game_date": "10/30/2026",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "projected_total": 229.5,
            "sportsbook_total": 224.5,
            "recommendation": "Strong Over",
            "prediction_time_utc": "2026-10-30T17:00:00Z",
            "commence_time_utc": "2026-10-30T19:00:00Z",
            "market_snapshot_time_utc": "2026-10-30T16:59:00Z",
            "market_last_update_utc": "2026-10-30T16:58:30Z",
            "market_snapshot_total_line": 224.5,
            "over_odds": 1.91,
            "under_odds": 1.93,
            "totals_bookmaker": "DraftKings",
            "snapshot_source": "the-odds-api-v4-current-endpoint",
            "source_event_id": event_id,
            "experimental_adjustments_applied": False,
        }
        values.update(overrides)
        return save_totals_pick(**values)

    def test_repeated_streamlit_rerun_does_not_duplicate_same_event(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "totals_history.csv"
            with patch("totals_tracker.TOTALS_HISTORY_FILE", str(path)):
                with patch("totals_tracker.sync_csv_mirror"):
                    first = self._save("event-123")
                    second = self._save(
                        "event-123",
                        projected_total=231.0,
                        sportsbook_total=225.5,
                        market_snapshot_total_line=225.5,
                        prediction_time_utc="2026-10-30T17:04:00Z",
                        market_snapshot_time_utc="2026-10-30T17:03:00Z",
                    )

            self.assertTrue(first)
            self.assertFalse(second)
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["source_event_id"], "event-123")
            self.assertEqual(float(saved.iloc[0]["projected_total"]), 229.5)

    def test_distinct_events_are_saved_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "totals_history.csv"
            with patch("totals_tracker.TOTALS_HISTORY_FILE", str(path)):
                with patch("totals_tracker.sync_csv_mirror"):
                    first = self._save("event-123")
                    second = self._save(
                        "event-456",
                        home_team="Denver Nuggets",
                        away_team="Phoenix Suns",
                    )

            self.assertTrue(first)
            self.assertTrue(second)
            self.assertEqual(len(pd.read_csv(path)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
