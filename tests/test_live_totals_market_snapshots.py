import csv
import tempfile
import unittest
from pathlib import Path

from live_totals_market_snapshots import append_live_totals_market_snapshots


def market_row(
    *,
    snapshot="2026-10-20T19:59:00Z",
    commence="2026-10-20T23:00:00Z",
    update="2026-10-20T19:58:30Z",
    line=220.5,
    over=1.91,
    under=1.93,
    over_line=None,
    under_line=None,
):
    return {
        "total_line": line,
        "over_total_line": line if over_line is None else over_line,
        "under_total_line": line if under_line is None else under_line,
        "over_odds": over,
        "under_odds": under,
        "totals_bookmaker": "DraftKings",
        "snapshot_timestamp_utc": snapshot,
        "market_last_update_utc": update,
        "commence_time_utc": commence,
        "snapshot_source": "the-odds-api-v4-current-endpoint",
        "source_event_id": "event-1",
    }


class LiveTotalsMarketSnapshotTests(unittest.TestCase):
    def test_writes_real_pregame_snapshot_with_price_and_time_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data" / "snapshots.csv"
            result = append_live_totals_market_snapshots(
                "10/21/2026",
                {("Home", "Away"): market_row()},
                output_file=path,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["rows_added"], 1)
            self.assertEqual(result["total_rows"], 1)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["game_date"], "2026-10-20")
            self.assertEqual(rows[0]["display_game_date"], "10/21/2026")
            self.assertEqual(rows[0]["snapshot_timestamp_utc"], "2026-10-20T19:59:00Z")
            self.assertEqual(rows[0]["commence_time_utc"], "2026-10-20T23:00:00Z")
            self.assertEqual(float(rows[0]["total_line"]), 220.5)
            self.assertEqual(float(rows[0]["over_odds"]), 1.91)
            self.assertEqual(float(rows[0]["under_odds"]), 1.93)
            self.assertEqual(rows[0]["bookmaker"], "DraftKings")

    def test_each_distinct_snapshot_is_retained_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            first = market_row(snapshot="2026-10-20T19:58:00Z", line=220.5)
            second = market_row(
                snapshot="2026-10-20T19:59:00Z",
                update="2026-10-20T19:58:45Z",
                line=221.5,
                over=1.90,
                under=1.94,
            )
            result1 = append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): first}, path)
            result2 = append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): second}, path)
            self.assertEqual(result1["total_rows"], 1)
            self.assertEqual(result2["rows_added"], 1)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["total_line"] for row in rows}, {"220.5", "221.5"})

    def test_repeated_cached_snapshot_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            row = market_row()
            first = append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): row}, path)
            second = append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): row}, path)
            self.assertEqual(first["rows_added"], 1)
            self.assertEqual(second["rows_added"], 0)
            self.assertEqual(second["total_rows"], 1)

    def test_conflicting_prices_at_same_snapshot_fail_without_rewriting_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            original = market_row()
            append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): original}, path)
            before = path.read_bytes()
            conflict = market_row(over=1.80)
            with self.assertRaisesRegex(ValueError, "Conflicting market data"):
                append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): conflict}, path)
            self.assertEqual(path.read_bytes(), before)

    def test_post_tipoff_and_invalid_price_or_line_pairs_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            post_tipoff = market_row(snapshot="2026-10-20T23:01:00Z", update="2026-10-20T23:00:00Z")
            bad_price = market_row(over=1.0)
            mismatched_pair = market_row(under_line=221.5)
            result = append_live_totals_market_snapshots(
                "10/20/2026",
                {
                    ("Home", "Away"): post_tipoff,
                    ("Home 2", "Away 2"): bad_price,
                    ("Home 3", "Away 3"): mismatched_pair,
                },
                output_file=path,
            )
            self.assertEqual(result["rows_added"], 0)
            self.assertEqual(result["post_tipoff_rows_skipped"], 1)
            self.assertEqual(result["invalid_rows_skipped"], 2)
            self.assertFalse(path.exists())

    def test_naive_snapshot_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            bad = market_row(snapshot="2026-10-20T19:59:00")
            result = append_live_totals_market_snapshots("10/20/2026", {("Home", "Away"): bad}, path)
            self.assertEqual(result["rows_added"], 0)
            self.assertEqual(result["invalid_rows_skipped"], 1)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
