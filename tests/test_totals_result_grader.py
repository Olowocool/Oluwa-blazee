import unittest
from unittest.mock import patch

import pandas as pd

from totals_result_grader import fetch_final_score, grade_totals_results


def make_pick(**overrides):
    row = {
        "game_date": "2026-10-29",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "recommendation": "Strong Over",
        "sportsbook_total": 220.5,
        "saved_total": 220.5,
        "closing_total": None,
        "over_odds": 1.91,
        "under_odds": 1.93,
        "stake": 50.0,
        "result": "Pending",
        "actual_total": None,
        "profit_loss": None,
        "clv": None,
    }
    row.update(overrides)
    return row


def score(total, home=110):
    away = total - home
    return {
        "home_score": float(home),
        "away_score": float(away),
        "matched_home_team": "Boston Celtics",
        "matched_away_team": "New York Knicks",
    }


class TotalsResultGraderTests(unittest.TestCase):
    def grade(self, row, actual_total):
        frame = pd.DataFrame([row])
        with patch("totals_result_grader.load_totals_history", return_value=frame), \
             patch("totals_result_grader.fetch_final_score", return_value=score(actual_total)), \
             patch("totals_result_grader.save_totals_history") as save:
            result = grade_totals_results()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["updated_rows"], 1)
        saved_frame = save.call_args.args[0]
        return saved_frame.iloc[0].to_dict()

    def test_over_win_uses_captured_over_price_and_actual_stake(self):
        graded = self.grade(make_pick(), 230.0)
        self.assertEqual(graded["result"], "Win")
        self.assertAlmostEqual(float(graded["profit_loss"]), 45.5)

    def test_under_win_uses_captured_under_price_and_actual_stake(self):
        graded = self.grade(
            make_pick(recommendation="Strong Under"),
            210.0,
        )
        self.assertEqual(graded["result"], "Win")
        self.assertAlmostEqual(float(graded["profit_loss"]), 46.5)

    def test_push_returns_zero_and_is_not_a_loss(self):
        graded = self.grade(make_pick(), 220.5)
        self.assertEqual(graded["result"], "Push")
        self.assertEqual(float(graded["profit_loss"]), 0.0)

    def test_missing_captured_price_grades_result_without_fabricating_profit(self):
        graded = self.grade(make_pick(over_odds=None), 230.0)
        self.assertEqual(graded["result"], "Win")
        self.assertTrue(pd.isna(graded["profit_loss"]))

    def test_missing_closing_line_does_not_create_zero_clv(self):
        graded = self.grade(make_pick(closing_total=None), 230.0)
        self.assertTrue(pd.isna(graded["closing_total"]))
        self.assertTrue(pd.isna(graded["clv"]))

    def test_push_is_already_settled_and_is_not_fetched_again(self):
        frame = pd.DataFrame([make_pick(result="Push", actual_total=220.5, profit_loss=0)])
        with patch("totals_result_grader.load_totals_history", return_value=frame), \
             patch("totals_result_grader.fetch_final_score") as fetch, \
             patch("totals_result_grader.save_totals_history") as save:
            result = grade_totals_results()

        self.assertEqual(result["updated_rows"], 0)
        fetch.assert_not_called()
        save.assert_not_called()

    def test_loss_uses_full_recorded_stake_not_fixed_constant(self):
        graded = self.grade(make_pick(stake=25.0), 210.0)
        self.assertEqual(graded["result"], "Loss")
        self.assertEqual(float(graded["profit_loss"]), -25.0)


    def test_http_429_uses_local_research_score_fallback(self):
        class RateLimitedResponse:
            status_code = 429

        completed = {
            "status": "completed",
            "home_score": 112,
            "away_score": 108,
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
        }
        diagnostics = {}
        with patch("totals_result_grader.requests.get", return_value=RateLimitedResponse()):
            with patch(
                "totals_result_grader._local_score_result",
                return_value=completed,
            ) as fallback:
                result = fetch_final_score(
                    "10/30/2026",
                    "Boston Celtics",
                    "New York Knicks",
                    diagnostics=diagnostics,
                )

        fallback.assert_called_once()
        self.assertEqual(result["home_score"], 112.0)
        self.assertEqual(result["away_score"], 108.0)
        self.assertEqual(diagnostics["shared_api_unavailable"], 1)

    def test_pending_score_is_reported_separately_from_lookup_errors(self):
        class PendingResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"status": "pending", "message": "Game not final yet."}

        diagnostics = {}
        with patch("totals_result_grader.requests.get", return_value=PendingResponse()):
            result = fetch_final_score(
                "10/30/2026",
                "Boston Celtics",
                "New York Knicks",
                diagnostics=diagnostics,
            )

        self.assertIsNone(result)
        self.assertEqual(diagnostics["pending_rows"], 1)
        self.assertNotIn("score_lookup_errors", diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
