import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import best_bet_selector_v2
import totals_ai_predictor
import totals_auto_learning
import totals_model
import train_totals_model
import generate_totals_test_data
import totals_tracker


class TotalsLeakageGuardTests(unittest.TestCase):
    def test_historical_cutoff_excludes_target_day_and_future(self):
        history = pd.DataFrame({
            "game_date": ["2026-10-20", "2026-10-26", "2026-10-27", "2026-10-28"],
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
            "home_score": [100, 101, 102, 103],
            "away_score": [99, 98, 97, 96],
        })
        actual = totals_model._history_before_date(history, "10/27/2026")
        self.assertEqual(pd.to_datetime(actual["game_date"]).dt.strftime("%Y-%m-%d").tolist(), ["2026-10-20", "2026-10-26"])


    def test_cutoff_excludes_entire_target_calendar_day_when_time_is_supplied(self):
        history = pd.DataFrame({
            "game_date": [
                "2026-10-26 23:30:00",
                "2026-10-27 08:00:00",
                "2026-10-27 19:45:00",
                "2026-10-28 01:00:00",
            ],
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
            "home_score": [100, 101, 102, 103],
            "away_score": [99, 98, 97, 96],
            "total_score": [199, 199, 199, 199],
        })
        actual = totals_model._history_before_date(
            history, "2026-10-27 20:00:00"
        )
        self.assertEqual(
            pd.to_datetime(actual["game_date"]).dt.strftime("%Y-%m-%d").tolist(),
            ["2026-10-26"],
        )
        self.assertEqual(actual["total_score"].tolist(), [199])

    def test_missing_target_date_defaults_to_lagos_today_cutoff(self):
        history = pd.DataFrame({
            "game_date": ["1900-01-01", "2999-01-01"],
            "home_team": ["A", "B"],
            "away_team": ["E", "F"],
            "home_score": [100, 101],
            "away_score": [99, 98],
        })
        actual = totals_model._history_before_date(history, None)
        dates = pd.to_datetime(actual["game_date"]).dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(dates, ["1900-01-01"])

    def test_historical_cutoff_fails_closed_for_missing_date_column(self):
        history = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        with self.assertRaisesRegex(ValueError, "no game_date/date column"):
            totals_model._history_before_date(history, "2026-10-27")

    def test_predict_totals_forwards_history_and_cutoff(self):
        expected = {"projected_total": 221.0}
        history = pd.DataFrame({
            "game_date": ["2026-10-20"],
            "home_team": ["A"], "away_team": ["B"],
            "home_score": [110], "away_score": [100],
        })
        with patch("totals_model.predict_game_total", return_value=expected) as mocked:
            result = totals_model.predict_totals("A", "B", 220.5, history_df=history, game_date="2026-10-27")
        self.assertEqual(result, expected)
        mocked.assert_called_once_with(
            home_team="A",
            away_team="B",
            bookmaker_total=220.5,
            game_date="2026-10-27",
            history_df=history,
        )

    def test_totals_ai_is_disabled_and_returns_no_numeric_confidence(self):
        result = totals_ai_predictor.totals_ai_prediction(220.0, 218.5, 1.5)
        self.assertEqual(result["status"], "disabled")
        self.assertIsNone(result["over_probability"])
        self.assertIsNone(result["under_probability"])
        self.assertIsNone(result["confidence"])

    def test_disabled_totals_confidence_cannot_win_best_bet_selector(self):
        result = best_bet_selector_v2.select_best_bet_v2(
            moneyline_confidence=63,
            moneyline_pick="Home Team",
            totals_confidence=None,
            totals_pick="Strong Over",
        )
        self.assertEqual(result["market"], "Moneyline")
        self.assertEqual(result["pick"], "Home Team")

    def test_post_game_fields_are_not_training_predictors(self):
        self.assertNotIn("actual_total", train_totals_model.FEATURES)
        self.assertNotIn("profit_loss", train_totals_model.FEATURES)
        self.assertNotIn("result", train_totals_model.FEATURES)
        self.assertNotIn("target", train_totals_model.FEATURES)

    def test_totals_learning_rejects_missing_pregame_timestamp_columns(self):
        row = {
            "game_date": "2026-10-20",
            "home_team": "Home",
            "away_team": "Away",
            "pick_type": "Over",
            "projected_total": 225.0,
            "sportsbook_total": 220.0,
            "edge": 5.0,
            "actual_total": 230.0,
            "profit_loss": 91.0,
            "result": "Win",
        }
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "totals_history.csv")
            learning_path = os.path.join(directory, "totals_learning_dataset.csv")
            pd.DataFrame([row] * 20).to_csv(history_path, index=False)
            with patch.object(totals_auto_learning, "TOTALS_HISTORY_FILE", history_path):
                with patch.object(totals_auto_learning, "TOTALS_LEARNING_FILE", learning_path):
                    result = totals_auto_learning.build_totals_learning_dataset()
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["training_allowed"])
            self.assertIn("before tip-off", result["message"])
            self.assertFalse(os.path.exists(learning_path))

    def test_totals_learning_excludes_post_tipoff_and_invalid_timestamp_rows(self):
        base = {
            "game_date": "2026-10-20",
            "home_team": "Home",
            "away_team": "Away",
            "pick_type": "Over",
            "projected_total": 225.0,
            "sportsbook_total": 220.0,
            "edge": 5.0,
            "actual_total": 230.0,
            "profit_loss": 91.0,
            "result": "Win",
            "source": "live_prediction_ui",
        }
        eligible = dict(base, prediction_time_utc="2026-10-20T20:00:00Z",
                        commence_time_utc="2026-10-20T23:00:00Z")
        post_tipoff = dict(base, prediction_time_utc="2026-10-20T23:01:00Z",
                           commence_time_utc="2026-10-20T23:00:00Z")
        naive_timestamp = dict(base, prediction_time_utc="2026-10-20T20:00:00",
                               commence_time_utc="2026-10-20T23:00:00Z")
        wrong_label = dict(base, prediction_time_utc="2026-10-20T20:00:00Z",
                           commence_time_utc="2026-10-20T23:00:00Z",
                           actual_total=215.0, result="Win")
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "totals_history.csv")
            learning_path = os.path.join(directory, "totals_learning_dataset.csv")
            pd.DataFrame([eligible, post_tipoff, naive_timestamp, wrong_label]).to_csv(history_path, index=False)
            with patch.object(totals_auto_learning, "TOTALS_HISTORY_FILE", history_path):
                with patch.object(totals_auto_learning, "TOTALS_LEARNING_FILE", learning_path):
                    result = totals_auto_learning.build_totals_learning_dataset()
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["timestamp_invalid_rows_excluded"], 1)
            self.assertEqual(result["post_tipoff_rows_excluded"], 1)
            self.assertEqual(result["inconsistent_rows_excluded"], 1)
            saved = pd.read_csv(learning_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["prediction_time_utc"], "2026-10-20T20:00:00Z")

    def test_save_totals_pick_records_prediction_and_tipoff_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "totals_history.csv")
            with patch.object(totals_tracker, "TOTALS_HISTORY_FILE", history_path):
                totals_tracker.save_totals_pick(
                    game_date="2026-10-20",
                    home_team="Home",
                    away_team="Away",
                    projected_total=225.0,
                    sportsbook_total=220.0,
                    recommendation="Lean Over",
                    prediction_time_utc="2026-10-20T20:00:00Z",
                    commence_time_utc="2026-10-20T23:00:00Z",
                )
            saved = pd.read_csv(history_path)
            self.assertEqual(saved.iloc[0]["prediction_time_utc"], "2026-10-20T20:00:00Z")
            self.assertEqual(saved.iloc[0]["commence_time_utc"], "2026-10-20T23:00:00Z")
            self.assertTrue(saved.iloc[0]["saved_at_utc"])

    def test_legacy_synthetic_test_rows_are_excluded_from_learning_dataset(self):
        synthetic = {
            "game_date": "06/14/2026",
            "prediction_time_utc": "2026-06-14T17:00:00Z",
            "commence_time_utc": "2026-06-14T20:00:00Z",
            "home_team": "Boston Celtics",
            "away_team": "Denver Nuggets",
            "market": "Totals",
            "pick_type": "Under",
            "projected_total": 220,
            "sportsbook_total": 230,
            "edge": -10,
            "recommendation": "Strong Under",
            "stake": 100,
            "actual_total": 215,
            "result": "Win",
            "profit_loss": 91,
        }
        genuine = dict(synthetic)
        genuine.update({
            "game_date": "2026-10-20",
            "prediction_time_utc": "2026-10-20T20:00:00Z",
            "commence_time_utc": "2026-10-20T23:00:00Z",
            "home_team": "New York Knicks",
            "away_team": "Philadelphia 76ers",
        })

        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "totals_history.csv")
            learning_path = os.path.join(directory, "totals_learning_dataset.csv")
            pd.DataFrame([synthetic, genuine]).to_csv(history_path, index=False)

            with patch.object(totals_auto_learning, "TOTALS_HISTORY_FILE", history_path), \
                 patch.object(totals_auto_learning, "TOTALS_LEARNING_FILE", learning_path):
                result = totals_auto_learning.build_totals_learning_dataset()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["synthetic_rows_excluded"], 1)

            saved = pd.read_csv(learning_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["home_team"], "New York Knicks")

    def test_training_uses_chronological_holdout_and_validates_pregame_timestamps(self):
        rows = []
        for day in range(10):
            date_value = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            date_text = date_value.strftime("%Y-%m-%d")
            for within_day in range(4):
                index = day * 4 + within_day
                line = 218 + (index % 7)
                pick = "Over" if index % 2 == 0 else "Under"
                target = 0 if index % 3 == 0 else 1
                if pick == "Over":
                    actual_total = line + 2 if target else line - 2
                    edge = 1.0
                else:
                    actual_total = line - 2 if target else line + 2
                    edge = -1.0
                rows.append({
                    "game_date": date_value,
                    "prediction_time_utc": date_text + "T12:00:00Z",
                    "commence_time_utc": date_text + "T18:00:00Z",
                    "pick_type": pick,
                    "projected_total": line + edge,
                    "sportsbook_total": line,
                    "edge": edge,
                    "is_under": int(pick == "Under"),
                    "is_over": int(pick == "Over"),
                    "actual_total": actual_total,
                    "profit_loss": 91 if target else -100,
                    "target": target,
                })

        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "totals_learning_dataset.csv")
            model_path = os.path.join(directory, "model.joblib")
            pd.DataFrame(rows).to_csv(data_path, index=False)
            with patch.object(train_totals_model, "DATA_FILE", data_path):
                with patch.object(train_totals_model, "MODEL_FILE", model_path):
                    result = train_totals_model.train_totals_model()

            self.assertEqual(result["status"], "success")
            self.assertLess(result["training_through"], result["holdout_from"])
            self.assertEqual(
                result["features_used"],
                ["projected_total", "sportsbook_total", "edge", "is_under", "is_over"],
            )
            self.assertNotIn("actual_total", result["features_used"])
            self.assertNotIn("profit_loss", result["features_used"])
            self.assertEqual(result["timestamp_invalid_rows_excluded"], 0)
            self.assertEqual(result["post_tipoff_rows_excluded"], 0)

    def test_training_refuses_dataset_without_timestamp_provenance(self):
        rows = []
        for day in range(10):
            date_value = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            for within_day in range(4):
                index = day * 4 + within_day
                line = 220.0
                pick = "Over" if index % 2 == 0 else "Under"
                target = index % 2
                actual_total = line + 2 if ((pick == "Over") == bool(target)) else line - 2
                edge = 5.0 if pick == "Over" else -5.0
                rows.append({
                    "game_date": date_value,
                    "pick_type": pick,
                    "projected_total": line + edge,
                    "sportsbook_total": line,
                    "edge": edge,
                    "is_under": int(pick == "Under"),
                    "is_over": int(pick == "Over"),
                    "actual_total": actual_total,
                    "target": target,
                })
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "totals_learning_dataset.csv")
            model_path = os.path.join(directory, "model.joblib")
            pd.DataFrame(rows).to_csv(data_path, index=False)
            with patch.object(train_totals_model, "DATA_FILE", data_path):
                with patch.object(train_totals_model, "MODEL_FILE", model_path):
                    result = train_totals_model.train_totals_model()
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["training_allowed"])
            self.assertFalse(os.path.exists(model_path))

    def test_synthetic_generator_does_not_touch_real_totals_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = os.path.join(directory, "totals_test_data.csv")
            history_path = os.path.join(directory, "totals_history.csv")
            with open(history_path, "w", encoding="utf-8") as file:
                file.write("sentinel\nreal ledger must stay unchanged\n")
            with open(history_path, "rb") as file:
                before = file.read()

            with patch.object(generate_totals_test_data, "TEST_DATA_FILE", test_path):
                with patch.object(generate_totals_test_data, "TOTALS_HISTORY_FILE", history_path, create=True):
                    result = generate_totals_test_data.add_totals_test_data()

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["rows_added"], 20)
            self.assertTrue(os.path.exists(test_path))
            with open(history_path, "rb") as file:
                self.assertEqual(file.read(), before)
            generated = pd.read_csv(test_path)
            self.assertTrue(generated["is_synthetic"].all())
            self.assertTrue((generated["source"] == "synthetic_test").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
