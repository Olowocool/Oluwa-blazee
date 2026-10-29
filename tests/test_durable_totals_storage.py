import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import durable_totals_storage as storage
from live_totals_market_snapshots import append_live_totals_market_snapshots
import prediction_history_logger
import totals_tracker
from totals_forward_paper_trading import evaluate_forward_paper_trading


TEST_DATABASE_URL = os.getenv("NBA_TOTALS_TEST_DATABASE_URL", "").strip()


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "Set NBA_TOTALS_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)
class DurableTotalsStorageTests(unittest.TestCase):
    def setUp(self):
        self.environment_patch = patch.dict(
            os.environ,
            {
                "NBA_TOTALS_DATABASE_URL": TEST_DATABASE_URL,
                "NBA_TOTALS_REQUIRE_DURABLE_STORAGE": "true",
            },
        )
        self.environment_patch.start()
        connection = storage._connect()
        try:
            storage._ensure_schema(connection)
            connection.execute(f"DELETE FROM {storage.TABLE_NAME}")
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.environment_patch.stop()

    @staticmethod
    def snapshot(**overrides):
        row = {
            "game_date": "2026-10-29",
            "display_game_date": "10/29/2026",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "bookmaker": "DraftKings",
            "snapshot_timestamp_utc": "2026-10-29T16:00:00Z",
            "market_last_update_utc": "2026-10-29T15:59:30Z",
            "commence_time_utc": "2026-10-29T20:00:00Z",
            "total_line": 220.5,
            "over_odds": 1.91,
            "under_odds": 1.93,
            "snapshot_source": "the-odds-api-v4-current-endpoint",
            "source_event_id": "nba-event-123",
        }
        row.update(overrides)
        return row

    def test_market_snapshot_insert_is_idempotent_and_conflicts_fail_closed(self):
        snapshot = self.snapshot()
        first = storage.persist_records(
            "market_snapshot",
            [snapshot],
            immutable=True,
            conflict_fields=storage.MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
        )
        repeated = storage.persist_records(
            "market_snapshot",
            [snapshot],
            immutable=True,
            conflict_fields=storage.MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
        )

        self.assertEqual(first["inserted"], 1)
        self.assertEqual(repeated["unchanged"], 1)
        self.assertEqual(len(storage.fetch_records("market_snapshot")), 1)

        changed = dict(snapshot, total_line=221.5)
        with self.assertRaisesRegex(ValueError, "Conflicting immutable market observation"):
            storage.persist_records(
                "market_snapshot",
                [changed],
                immutable=True,
                conflict_fields=storage.MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
            )

        saved = storage.fetch_records("market_snapshot")
        self.assertEqual(len(saved), 1)
        self.assertEqual(float(saved[0]["total_line"]), 220.5)

    def test_same_timestamp_price_conflict_from_snapshot_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.csv"
            first = self.snapshot()
            odds = {
                ("Boston Celtics", "New York Knicks"): {
                    "snapshot_timestamp_utc": first["snapshot_timestamp_utc"],
                    "market_last_update_utc": first["market_last_update_utc"],
                    "commence_time_utc": first["commence_time_utc"],
                    "total_line": first["total_line"],
                    "over_total_line": first["total_line"],
                    "under_total_line": first["total_line"],
                    "over_odds": first["over_odds"],
                    "under_odds": first["under_odds"],
                    "totals_bookmaker": first["bookmaker"],
                    "snapshot_source": first["snapshot_source"],
                    "source_event_id": first["source_event_id"],
                    "market_home_team": first["home_team"],
                    "market_away_team": first["away_team"],
                }
            }
            saved = append_live_totals_market_snapshots("10/29/2026", odds, path)
            self.assertEqual(saved["rows_added"], 1)
            before = path.read_bytes()

            odds[("Boston Celtics", "New York Knicks")]["over_odds"] = 1.80
            with self.assertRaisesRegex(ValueError, "Conflicting market data"):
                append_live_totals_market_snapshots("10/29/2026", odds, path)

            self.assertEqual(path.read_bytes(), before)
            database_rows = storage.fetch_records("market_snapshot")
            self.assertEqual(len(database_rows), 1)
            self.assertEqual(float(database_rows[0]["over_odds"]), 1.91)

    def test_pick_result_updates_existing_record_by_stable_pick_id(self):
        initial = {
            "pick_id": "test-pick-stable-id",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "result": "Pending",
            "actual_total": None,
            "profit_loss": 0,
        }
        updated = dict(initial, result="Win", actual_total=225, profit_loss=91)

        created = storage.persist_records("totals_pick", [initial])
        changed = storage.persist_records("totals_pick", [updated])

        self.assertEqual(created["inserted"], 1)
        self.assertEqual(changed["updated"], 1)
        rows = storage.fetch_records("totals_pick")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "Win")
        self.assertEqual(rows[0]["actual_total"], 225)

    def test_database_hydrates_a_stale_csv_mirror_without_overwriting_database(self):
        stored = {
            "pick_id": "test-sync-stable-id",
            "game_date": "2026-10-29",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "result": "Win",
            "actual_total": 225,
            "profit_loss": 91,
        }
        storage.persist_records("totals_pick", [stored])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "totals_history.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pick_id", "game_date", "result", "actual_total", "profit_loss"],
                )
                writer.writeheader()
                writer.writerow({
                    "pick_id": "test-sync-stable-id",
                    "game_date": "2026-10-29",
                    "result": "Pending",
                    "actual_total": "",
                    "profit_loss": 0,
                })

            active = storage.sync_csv_mirror("totals_pick", path)
            self.assertTrue(active)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["result"], "Win")
            self.assertEqual(rows[0]["actual_total"], "225")

    def test_real_pick_tracker_saves_and_updates_the_same_durable_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "totals_history.csv"
            with patch.object(totals_tracker, "TOTALS_HISTORY_FILE", str(history_path)):
                totals_tracker.save_totals_pick(
                    game_date="10/29/2026",
                    home_team="Boston Celtics",
                    away_team="New York Knicks",
                    projected_total=230.0,
                    sportsbook_total=220.5,
                    recommendation="Strong Over",
                    prediction_time_utc="2026-10-29T15:57:00Z",
                    commence_time_utc="2026-10-29T20:00:00Z",
                    market_snapshot_time_utc="2026-10-29T15:55:00Z",
                    market_last_update_utc="2026-10-29T15:54:30Z",
                    market_snapshot_total_line=220.5,
                    over_odds=1.91,
                    under_odds=1.93,
                    totals_bookmaker="DraftKings",
                    snapshot_source="the-odds-api-v4-current-endpoint",
                    source_event_id="nba-event-tracker-test",
                )
                saved = totals_tracker.load_totals_history()
                pick_id = str(saved.loc[0, "pick_id"])
                saved.loc[0, "actual_total"] = 230
                saved.loc[0, "result"] = "Win"
                saved.loc[0, "profit_loss"] = 91
                totals_tracker.save_totals_history(saved)

            durable_rows = storage.fetch_records("totals_pick")
            self.assertEqual(len(durable_rows), 1)
            self.assertEqual(durable_rows[0]["pick_id"], pick_id)
            self.assertEqual(durable_rows[0]["result"], "Win")
            self.assertEqual(float(durable_rows[0]["actual_total"]), 230.0)

    def test_real_prediction_logger_writes_and_deduplicates_durable_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            history_path = output_dir / "prediction_history.csv"
            prediction = {
                "prediction_time_utc": "2026-10-29T15:57:00Z",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "bookmaker_total": 220.5,
                "projected_total": 230.0,
                "projected_home_points": 116.0,
                "projected_away_points": 114.0,
                "edge": 9.5,
                "recommendation": "Strong Over",
            }
            with patch.object(prediction_history_logger, "OUTPUT_DIR", output_dir), patch.object(
                prediction_history_logger, "HISTORY_FILE", history_path
            ), patch.object(prediction_history_logger, "latest_injury_snapshot", return_value=""):
                first = prediction_history_logger.log_prediction(
                    prediction,
                    game_date="10/29/2026",
                    commence_time_utc="2026-10-29T20:00:00Z",
                    market_snapshot_time_utc="2026-10-29T15:55:00Z",
                    market_last_update_utc="2026-10-29T15:54:30Z",
                    market_snapshot_total_line=220.5,
                    over_odds=1.91,
                    under_odds=1.93,
                    totals_bookmaker="DraftKings",
                    snapshot_source="the-odds-api-v4-current-endpoint",
                    source_event_id="nba-event-audit-test",
                )
                prediction["prediction_time_utc"] = "2026-10-29T15:57:30Z"
                duplicate = prediction_history_logger.log_prediction(
                    prediction,
                    game_date="10/29/2026",
                    commence_time_utc="2026-10-29T20:00:00Z",
                    market_snapshot_time_utc="2026-10-29T15:55:00Z",
                    market_last_update_utc="2026-10-29T15:54:30Z",
                    market_snapshot_total_line=220.5,
                    over_odds=1.91,
                    under_odds=1.93,
                    totals_bookmaker="DraftKings",
                    snapshot_source="the-odds-api-v4-current-endpoint",
                    source_event_id="nba-event-audit-test",
                )

            durable_rows = storage.fetch_records("prediction_audit")
            self.assertTrue(first["prediction_id"])
            self.assertTrue(duplicate["prediction_id"])
            self.assertEqual(len(durable_rows), 1)
            self.assertEqual(durable_rows[0]["recommendation"], "Strong Over")
            self.assertTrue(history_path.exists())

    def test_forward_evaluator_hydrates_both_ledgers_from_postgres(self):
        snapshot = self.snapshot()
        pick = {
            "pick_id": "durable-forward-eval-pick",
            "game_date": "2026-10-29",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "pick_type": "Over",
            "prediction_time_utc": "2026-10-29T16:02:00Z",
            "commence_time_utc": "2026-10-29T20:00:00Z",
            "market_snapshot_time_utc": "2026-10-29T16:00:00Z",
            "market_last_update_utc": "2026-10-29T15:59:30Z",
            "market_snapshot_total_line": 220.5,
            "over_odds": 1.91,
            "under_odds": 1.93,
            "totals_bookmaker": "DraftKings",
            "market_snapshot_source": "the-odds-api-v4-current-endpoint",
            "source_event_id": "nba-event-123",
            "sportsbook_total": 220.5,
            "actual_total": 225,
            "result": "Win",
            "stake": 100,
            "experimental_adjustments_applied": False,
            "profit_loss": 91,
        }
        storage.persist_records(
            "market_snapshot",
            [snapshot],
            immutable=True,
            conflict_fields=storage.MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
        )
        storage.persist_records("totals_pick", [pick])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = evaluate_forward_paper_trading(
                totals_history_path=root / "totals_history.csv",
                market_snapshot_path=root / "live_totals_market_snapshots.csv",
            )

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["eligible_picks"], 1)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["net_profit"], 91.0)
        self.assertEqual(report["roi_pct"], 91.0)
        self.assertFalse(report["probabilities_used"])
        self.assertFalse(report["live_betting_enabled"])

    def test_required_durable_mode_refuses_missing_database_url(self):
        with patch("durable_totals_storage.get_database_url", return_value=""), patch(
            "durable_totals_storage.durable_storage_required", return_value=True
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError, "NBA_TOTALS_DATABASE_URL"):
                    storage.sync_csv_mirror("totals_pick", Path(tmp) / "picks.csv")


class DurableTotalsStorageNoDatabaseTests(unittest.TestCase):
    def test_local_mode_is_explicitly_reported_as_non_durable(self):
        with patch("durable_totals_storage.get_database_url", return_value=""), patch(
            "durable_totals_storage.durable_storage_required", return_value=False
        ):
            status = storage.storage_status()
        self.assertEqual(status["mode"], "local-csv")
        self.assertFalse(status["durable"])
        self.assertFalse(status["database_configured"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
