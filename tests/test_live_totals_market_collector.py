import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import live_totals_market_collector as collector


def event(
    *,
    event_id="event-1",
    home="Boston Celtics",
    away="New York Knicks",
    commence="2026-10-30T23:00:00Z",
    line=221.5,
    over=1.91,
    under=1.93,
    bookmaker="draftkings",
):
    return {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": line, "price": over},
                            {"name": "Under", "point": line, "price": under},
                        ],
                    }
                ],
            },
            {
                "key": bookmaker,
                "title": "DraftKings",
                "last_update": "2026-10-30T21:59:00Z",
                "markets": [
                    {
                        "key": "totals",
                        "last_update": "2026-10-30T21:59:30Z",
                        "outcomes": [
                            {"name": "Over", "point": line, "price": over},
                            {"name": "Under", "point": line, "price": under},
                        ],
                    }
                ],
            },
        ],
    }


class LiveTotalsMarketCollectorTests(unittest.TestCase):
    def test_keeps_only_draftkings_complete_pregame_total(self):
        captured = "2026-10-30T22:00:00Z"
        mapped = collector.build_market_snapshot_map([event()], captured)

        self.assertEqual(len(mapped), 1)
        row = mapped[("boston celtics", "new york knicks")]
        self.assertEqual(row["totals_bookmaker"], "DraftKings")
        self.assertEqual(row["snapshot_timestamp_utc"], captured)
        self.assertEqual(row["market_last_update_utc"], "2026-10-30T21:59:30Z")
        self.assertEqual(row["source_event_id"], "event-1")
        self.assertEqual(row["total_line"], 221.5)

    def test_skips_events_without_draftkings_market(self):
        mapped = collector.build_market_snapshot_map(
            [event(bookmaker="fanduel")],
            "2026-10-30T22:00:00Z",
        )
        self.assertEqual(mapped, {})

    def test_skips_post_tipoff_and_invalid_line_or_price_pairs(self):
        post_tip = event(commence="2026-10-30T21:59:59Z")
        mismatched = event(event_id="event-2")
        mismatched["bookmakers"][1]["markets"][0]["outcomes"][1]["point"] = 222.5
        bad_odds = event(event_id="event-3", over=1.0)
        mapped = collector.build_market_snapshot_map(
            [post_tip, mismatched, bad_odds],
            "2026-10-30T22:00:00Z",
        )
        self.assertEqual(mapped, {})

    def test_requires_timezone_aware_capture_time(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            collector.build_market_snapshot_map([event()], "2026-10-30T22:00:00")

    def test_collect_once_fails_closed_without_api_key_or_durable_storage(self):
        with patch.dict("os.environ", {"ODDS_API_KEY": ""}), patch.object(
            collector, "durable_storage_required", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "ODDS_API_KEY"):
                collector.collect_once()

        with patch.dict("os.environ", {"ODDS_API_KEY": "test-key"}), patch.object(
            collector, "durable_storage_required", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "CSV-only"):
                collector.collect_once()

    def test_collect_once_fetches_and_persists_with_database_check(self):
        fake_response = Mock()
        fake_response.json.return_value = [event()]
        fake_response.headers = {
            "x-requests-remaining": "499",
            "x-requests-used": "1",
            "x-requests-last": "1",
        }
        fake_response.raise_for_status.return_value = None
        fake_session = Mock()
        fake_session.get.return_value = fake_response
        fixed_now = datetime(2026, 10, 30, 22, 0, tzinfo=timezone.utc)
        saved_result = {
            "status": "success",
            "rows_added": 1,
            "invalid_rows_skipped": 0,
            "post_tipoff_rows_skipped": 0,
        }

        with patch.dict(
            "os.environ",
            {
                "ODDS_API_KEY": "test-key",
                "NBA_TOTALS_REQUIRE_DURABLE_STORAGE": "true",
            },
        ), patch.object(collector, "durable_storage_required", return_value=True), patch.object(
            collector,
            "test_database_connection",
            return_value={"ok": True, "message": "ok"},
        ), patch.object(
            collector,
            "append_live_totals_market_snapshots",
            return_value=saved_result,
        ) as append:
            result = collector.collect_once(session=fake_session, now=fixed_now)

        self.assertEqual(result["rows_added"], 1)
        self.assertEqual(result["requests_remaining"], "499")
        fake_session.get.assert_called_once()
        call = fake_session.get.call_args
        self.assertEqual(call.args[0], collector.ODDS_API_URL)
        self.assertEqual(call.kwargs["params"]["markets"], "totals")
        self.assertEqual(call.kwargs["params"]["regions"], "us")
        append.assert_called_once()
        self.assertEqual(append.call_args.args[1][("boston celtics", "new york knicks")]["source_event_id"], "event-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
