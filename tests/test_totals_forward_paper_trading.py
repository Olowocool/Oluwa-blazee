import csv
import tempfile
import unittest
from pathlib import Path

from totals_forward_paper_trading import evaluate_forward_paper_trading


PICK_FIELDS = [
    "game_date", "home_team", "away_team", "pick_type",
    "prediction_time_utc", "commence_time_utc", "market_snapshot_time_utc",
    "market_last_update_utc", "market_snapshot_total_line", "over_odds",
    "under_odds", "totals_bookmaker", "market_snapshot_source",
    "source_event_id", "sportsbook_total", "actual_total", "result", "stake",
    "experimental_adjustments_applied", "profit_loss",
]
SNAPSHOT_FIELDS = [
    "game_date", "home_team", "away_team", "bookmaker", "source_event_id",
    "snapshot_timestamp_utc", "market_last_update_utc", "commence_time_utc",
    "total_line", "over_odds", "under_odds", "snapshot_source",
]


def pick_row(**overrides):
    row = {
        "game_date": "2026-10-29",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "pick_type": "Over",
        "prediction_time_utc": "2026-10-29T17:57:00Z",
        "commence_time_utc": "2026-10-29T18:30:00Z",
        "market_snapshot_time_utc": "2026-10-29T17:55:00Z",
        "market_last_update_utc": "2026-10-29T17:54:30Z",
        "market_snapshot_total_line": 220.5,
        "over_odds": 1.91,
        "under_odds": 1.93,
        "totals_bookmaker": "DraftKings",
        "market_snapshot_source": "the-odds-api-v4-current-endpoint",
        "source_event_id": "event-123",
        "sportsbook_total": 220.5,
        "actual_total": 225,
        "result": "win",
        "stake": 100,
        "experimental_adjustments_applied": "false",
        "profit_loss": 91,
    }
    row.update(overrides)
    return row


def snapshot_row(**overrides):
    row = {
        "game_date": "2026-10-29",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmaker": "DraftKings",
        "source_event_id": "event-123",
        "snapshot_timestamp_utc": "2026-10-29T17:55:00Z",
        "market_last_update_utc": "2026-10-29T17:54:30Z",
        "commence_time_utc": "2026-10-29T18:30:00Z",
        "total_line": 220.5,
        "over_odds": 1.91,
        "under_odds": 1.93,
        "snapshot_source": "the-odds-api-v4-current-endpoint",
    }
    row.update(overrides)
    return row


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TotalsForwardPaperTradingTests(unittest.TestCase):
    def run_evaluation(self, picks, snapshots):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        history_path = root / "totals_history.csv"
        snapshots_path = root / "live_totals_market_snapshots.csv"
        write_csv(history_path, PICK_FIELDS, picks)
        write_csv(snapshots_path, SNAPSHOT_FIELDS, snapshots)
        report = evaluate_forward_paper_trading(
            totals_history_path=history_path,
            market_snapshot_path=snapshots_path,
        )
        return report

    def test_over_pick_uses_entry_price_and_latest_valid_pretip_line(self):
        entry = snapshot_row()
        later = snapshot_row(
            snapshot_timestamp_utc="2026-10-29T18:10:00Z",
            market_last_update_utc="2026-10-29T18:09:30Z",
            total_line=222.5,
            over_odds=1.90,
            under_odds=1.94,
        )
        report = self.run_evaluation([pick_row()], [entry, later])

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["eligible_picks"], 1)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["net_profit"], 91.0)
        self.assertEqual(report["roi_pct"], 91.0)
        self.assertEqual(report["mean_line_clv_points"], 2.0)
        self.assertFalse(report["probabilities_used"])
        self.assertFalse(report["live_betting_enabled"])

    def test_under_line_movement_uses_correct_clv_sign(self):
        pick = pick_row(
            pick_type="Under",
            market_snapshot_total_line=224.5,
            sportsbook_total=224.5,
            over_odds=1.92,
            under_odds=1.93,
            actual_total=220,
            result="win",
            profit_loss=93,
        )
        entry = snapshot_row(
            total_line=224.5,
            over_odds=1.92,
            under_odds=1.93,
        )
        later = snapshot_row(
            snapshot_timestamp_utc="2026-10-29T18:10:00Z",
            market_last_update_utc="2026-10-29T18:09:30Z",
            total_line=222.5,
            over_odds=1.94,
            under_odds=1.90,
        )
        report = self.run_evaluation([pick], [entry, later])
        self.assertEqual(report["eligible_picks"], 1)
        self.assertEqual(report["mean_line_clv_points"], 2.0)
        self.assertEqual(report["net_profit"], 93.0)

    def test_push_is_counted_and_has_zero_profit(self):
        pick = pick_row(
            market_snapshot_total_line=220,
            sportsbook_total=220,
            actual_total=220,
            result="push",
            profit_loss=0,
        )
        entry = snapshot_row(total_line=220)
        later = snapshot_row(
            snapshot_timestamp_utc="2026-10-29T18:10:00Z",
            market_last_update_utc="2026-10-29T18:09:30Z",
            total_line=220,
        )
        report = self.run_evaluation([pick], [entry, later])
        self.assertEqual(report["eligible_picks"], 1)
        self.assertEqual(report["pushes"], 1)
        self.assertEqual(report["wins"], 0)
        self.assertEqual(report["net_profit"], 0.0)
        self.assertEqual(report["roi_pct"], 0.0)
        self.assertIsNone(report["win_rate_pct_excluding_pushes"])
        self.assertEqual(report["mean_line_clv_points"], 0.0)

    def test_stale_entry_snapshot_is_excluded(self):
        pick = pick_row(prediction_time_utc="2026-10-29T18:01:00Z")
        report = self.run_evaluation([pick], [snapshot_row()])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["stale_entry_snapshot"], 1)

    def test_result_label_must_match_actual_total_and_pick_side(self):
        pick = pick_row(actual_total=210, result="win")
        report = self.run_evaluation([pick], [snapshot_row()])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["settlement_label_mismatch"], 1)

    def test_entry_snapshot_must_exist_in_append_only_market_ledger(self):
        only_later = snapshot_row(
            snapshot_timestamp_utc="2026-10-29T18:10:00Z",
            market_last_update_utc="2026-10-29T18:09:30Z",
            total_line=222.5,
        )
        report = self.run_evaluation([pick_row()], [only_later])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["entry_snapshot_not_found_or_mismatched"], 1)

    def test_missing_experimental_flag_is_excluded(self):
        pick = pick_row(experimental_adjustments_applied="")
        report = self.run_evaluation([pick], [snapshot_row()])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["missing_or_invalid_experimental_flag"], 1)

    def test_provider_update_timestamp_must_match_entry_snapshot(self):
        pick = pick_row(market_last_update_utc="2026-10-29T17:54:00Z")
        report = self.run_evaluation([pick], [snapshot_row()])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["entry_snapshot_not_found_or_mismatched"], 1)

    def test_experimental_adjustment_pick_is_not_in_official_results(self):
        pick = pick_row(experimental_adjustments_applied="true")
        report = self.run_evaluation([pick], [snapshot_row()])
        self.assertEqual(report["eligible_picks"], 0)
        self.assertEqual(report["excluded_by_reason"]["experimental_adjustment_pick"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
