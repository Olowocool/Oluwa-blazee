import argparse
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from collect_historical_totals_odds import (
    build_requested_snapshots,
    collect,
    estimate_api_credits,
    merge_archive,
    parse_snapshot_response,
    parse_snapshot_times,
)


def payload(snapshot="2025-01-01T20:55:00Z", commence="2025-01-02T00:30:00Z"):
    return {
        "timestamp": snapshot,
        "data": [
            {
                "id": "event-1",
                "commence_time": commence,
                "home_team": "Los Angeles Clippers",
                "away_team": "Los Angeles Lakers",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": "2025-01-01T20:54:00Z",
                        "markets": [
                            {
                                "key": "totals",
                                "last_update": "2025-01-01T20:54:00Z",
                                "outcomes": [
                                    {"name": "Over", "price": 1.91, "point": 220.5},
                                    {"name": "Under", "price": 1.95, "point": 220.5},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "event-live",
                "commence_time": "2025-01-01T20:30:00Z",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [{
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 225.5},
                                {"name": "Under", "price": 1.95, "point": 225.5},
                            ],
                        }],
                    }
                ],
            },
        ],
    }


class HistoricalTotalsOddsCollectorTests(unittest.TestCase):
    def test_times_are_unique_and_sorted(self):
        self.assertEqual(
            [t.strftime("%H:%M") for t in parse_snapshot_times("20:00,16:00,18:30")],
            ["16:00", "18:30", "20:00"],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_snapshot_times("16:00,16:00")
        with self.assertRaisesRegex(ValueError, "HH:MM"):
            parse_snapshot_times("4pm")

    def test_requested_times_use_eastern_calendar_and_dst(self):
        requested = build_requested_snapshots(
            "2025-01-01",
            "2025-01-01",
            parse_snapshot_times("16:00"),
        )
        self.assertEqual(len(requested), 1)
        # 16:00 Eastern is 21:00 UTC in January.
        self.assertEqual(requested[0], datetime(2025, 1, 1, 21, 0, tzinfo=timezone.utc))

        summer = build_requested_snapshots(
            "2025-07-01", "2025-07-01", parse_snapshot_times("16:00")
        )
        # 16:00 Eastern is 20:00 UTC during daylight saving time.
        self.assertEqual(summer[0], datetime(2025, 7, 1, 20, 0, tzinfo=timezone.utc))

    def test_credit_estimate_is_explicit_and_deterministic(self):
        self.assertEqual(estimate_api_credits(0), 0)
        self.assertEqual(estimate_api_credits(12), 120)

    def test_parser_keeps_real_prices_and_eastern_game_date(self):
        rows, diagnostics = parse_snapshot_response(
            payload(),
            datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # 00:30 UTC Jan 2 is 19:30 ET Jan 1.
        self.assertEqual(row["game_date"], "2025-01-01")
        self.assertEqual(row["snapshot_timestamp_utc"], "2025-01-01T20:55:00Z")
        self.assertEqual(row["commence_time_utc"], "2025-01-02T00:30:00Z")
        self.assertEqual(row["bookmaker"], "DraftKings")
        self.assertEqual(row["bookmaker_key"], "draftkings")
        self.assertEqual(row["total_line"], 220.5)
        self.assertEqual(row["over_odds"], 1.91)
        self.assertEqual(row["under_odds"], 1.95)
        self.assertEqual(row["snapshot_source"], "the-odds-api-v4-historical")
        self.assertEqual(diagnostics["post_tipoff_events_excluded"], 1)

    def test_parser_can_filter_to_one_bookmaker(self):
        rows, _ = parse_snapshot_response(
            payload(),
            datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc),
            bookmaker_key="DRAFTKINGS",
        )
        self.assertEqual([row["bookmaker_key"] for row in rows], ["draftkings"])

        no_match, _ = parse_snapshot_response(
            payload(),
            datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc),
            bookmaker_key="unknown_book",
        )
        self.assertEqual(no_match, [])

    def test_parser_skips_ambiguous_or_invalid_price_pairs(self):
        broken = payload()
        broken["data"][0]["bookmakers"][0]["markets"][0]["outcomes"][1]["point"] = 221.5
        rows, diag = parse_snapshot_response(
            broken, datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(rows, [])
        self.assertGreater(diag["invalid_market_pairs_skipped"], 0)

    def test_merge_is_idempotent_and_rejects_conflicting_price(self):
        rows, _ = parse_snapshot_response(
            payload(), datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc)
        )
        merged = merge_archive(rows, rows)
        self.assertEqual(len(merged), 1)

        conflict = dict(rows[0])
        conflict["over_odds"] = 1.80
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            merge_archive(rows, [conflict])

    def test_plan_only_never_calls_network(self):
        args = argparse.Namespace(
            start_date="2025-01-01",
            end_date="2025-01-02",
            snapshot_times="16:00,18:00",
            bookmaker_key=None,
            output_file="unused.csv",
            plan_only=True,
            confirm_api_cost=False,
            max_api_credits=None,
        )
        with patch("collect_historical_totals_odds.fetch_snapshot", side_effect=AssertionError("network called")):
            result = collect(args)
        self.assertEqual(result["status"], "plan_only")
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["estimated_api_credits"], 40)

    def test_collection_requires_cost_confirmation_and_budget_before_network(self):
        base = dict(
            start_date="2025-01-01",
            end_date="2025-01-01",
            snapshot_times="16:00",
            bookmaker_key=None,
            output_file="unused.csv",
            plan_only=False,
            confirm_api_cost=False,
            max_api_credits=10,
        )
        with patch("collect_historical_totals_odds.fetch_snapshot", side_effect=AssertionError("network called")):
            with self.assertRaisesRegex(ValueError, "No API calls were made"):
                collect(argparse.Namespace(**base))

        base["confirm_api_cost"] = True
        base["max_api_credits"] = 9
        with patch("collect_historical_totals_odds.fetch_snapshot", side_effect=AssertionError("network called")):
            with self.assertRaisesRegex(ValueError, "exceeds --max-api-credits"):
                collect(argparse.Namespace(**base))

    def test_collection_persists_provider_snapshot_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "market.csv")
            args = argparse.Namespace(
                start_date="2025-01-01",
                end_date="2025-01-01",
                snapshot_times="16:00",
                bookmaker_key="draftkings",
                output_file=output,
                plan_only=False,
                confirm_api_cost=True,
                max_api_credits=10,
            )
            requested = datetime(2025, 1, 1, 21, 0, tzinfo=timezone.utc)
            with patch("collect_historical_totals_odds.load_api_key", return_value="secret-do-not-log"), \
                 patch("collect_historical_totals_odds.fetch_snapshot", return_value=(payload(), {"requests_remaining": "999", "requests_used": "10"})) as fetch:
                result = collect(args)

            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["archive_rows"], 1)
            with open(output, encoding="utf-8", newline="") as handle:
                rows = list(__import__("csv").DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bookmaker_key"], "draftkings")
            self.assertEqual(rows[0]["snapshot_timestamp_utc"], "2025-01-01T20:55:00Z")
            self.assertNotIn("secret-do-not-log", Path(output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
