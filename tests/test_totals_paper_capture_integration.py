"""End-to-end guard for capturing and evaluating timestamped paper totals picks."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from live_totals_market_snapshots import append_live_totals_market_snapshots
from totals_forward_paper_trading import evaluate_forward_paper_trading
from totals_tracker import save_totals_pick


def market_observation(snapshot_time, update_time, line, over, under):
    return {
        "total_line": line,
        "over_total_line": line,
        "under_total_line": line,
        "over_odds": over,
        "under_odds": under,
        "totals_bookmaker": "DraftKings",
        "snapshot_timestamp_utc": snapshot_time,
        "market_last_update_utc": update_time,
        "commence_time_utc": "2026-10-29T18:30:00Z",
        "snapshot_source": "test-current-market-observation",
        "source_event_id": "nba-event-123",
    }


def settle_pick(path, *, actual_total, result, profit_loss):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["actual_total"] = str(actual_total)
    rows[0]["result"] = result
    rows[0]["profit_loss"] = str(profit_loss)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TotalsPaperCaptureIntegrationTests(unittest.TestCase):
    def test_saved_pick_matches_immutable_entry_snapshot_and_evaluates_clv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks_path = root / "totals_history.csv"
            snapshots_path = root / "live_totals_market_snapshots.csv"

            entry = market_observation(
                "2026-10-29T17:55:00Z",
                "2026-10-29T17:54:30Z",
                220.5,
                1.91,
                1.93,
            )
            later = market_observation(
                "2026-10-29T18:10:00Z",
                "2026-10-29T18:09:30Z",
                222.5,
                1.90,
                1.94,
            )

            # Avoid using any configured developer/CI database. All writes stay
            # inside this temporary directory and exercise the CSV mirrors.
            with patch("totals_tracker.TOTALS_HISTORY_FILE", str(picks_path)), \
                 patch("totals_tracker.sync_csv_mirror", return_value=False), \
                 patch("totals_tracker.persist_records"), \
                 patch("live_totals_market_snapshots.sync_csv_mirror", return_value=False), \
                 patch("totals_forward_paper_trading.sync_csv_mirror", return_value=False):
                first_capture = append_live_totals_market_snapshots(
                    "10/29/2026",
                    {("Boston Celtics", "New York Knicks"): entry},
                    output_file=snapshots_path,
                )
                self.assertEqual(first_capture["rows_added"], 1)

                save_totals_pick(
                    game_date="10/29/2026",
                    home_team="Boston Celtics",
                    away_team="New York Knicks",
                    projected_total=224.0,
                    sportsbook_total=220.5,
                    recommendation="Strong Over",
                    prediction_time_utc="2026-10-29T17:57:00Z",
                    commence_time_utc="2026-10-29T18:30:00Z",
                    market_snapshot_time_utc=entry["snapshot_timestamp_utc"],
                    market_last_update_utc=entry["market_last_update_utc"],
                    market_snapshot_total_line=entry["total_line"],
                    over_odds=entry["over_odds"],
                    under_odds=entry["under_odds"],
                    totals_bookmaker=entry["totals_bookmaker"],
                    snapshot_source=entry["snapshot_source"],
                    source_event_id=entry["source_event_id"],
                    experimental_adjustments_applied=False,
                )

                later_capture = append_live_totals_market_snapshots(
                    "10/29/2026",
                    {("Boston Celtics", "New York Knicks"): later},
                    output_file=snapshots_path,
                )
                self.assertEqual(later_capture["rows_added"], 1)
                self.assertEqual(later_capture["total_rows"], 2)

                # Settle the paper record as the results updater would after the
                # game. The evaluator independently checks this label against score.
                settle_pick(
                    picks_path,
                    actual_total=225,
                    result="win",
                    profit_loss=91.0,
                )

                report = evaluate_forward_paper_trading(
                    totals_history_path=picks_path,
                    market_snapshot_path=snapshots_path,
                    picks_output_path=root / "evaluated_picks.csv",
                    summary_output_path=root / "summary.json",
                )

            self.assertEqual(report["status"], "success")
            self.assertEqual(report["picks_seen"], 1)
            self.assertEqual(report["eligible_picks"], 1)
            self.assertEqual(report["wins"], 1)
            self.assertEqual(report["net_profit"], 91.0)
            self.assertEqual(report["roi_pct"], 91.0)
            self.assertEqual(report["picks_with_later_valid_pre_tip_snapshot"], 1)
            self.assertEqual(report["mean_line_clv_points"], 2.0)
            self.assertFalse(report["live_betting_enabled"])
            self.assertFalse(report["probabilities_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
