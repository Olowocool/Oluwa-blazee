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
        self.assertEqual(actual["game_date"].tolist(), ["2026-10-20", "2026-10-26"])

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

    def test_legacy_synthetic_test_rows_are_excluded_from_learning_dataset(self):
        synthetic = {
            "game_date": "06/14/2026",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
