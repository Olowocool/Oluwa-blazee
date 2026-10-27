import unittest

import pandas as pd

from points_engine import calculate_matchup_points, calculate_team_rest_days
from totals_model import classify_total_edge, predict_game_total


def sample_history():
    rows = [
        ("2026-10-20", "Lakers", "Bulls", 110, 105),
        ("2026-10-21", "Bulls", "Kings", 108, 102),
        ("2026-10-22", "Kings", "Lakers", 111, 115),
        ("2026-10-24", "Bulls", "Lakers", 103, 107),
        ("2026-10-25", "Lakers", "Kings", 116, 108),
        # These target/future rows must never be used to predict October 27.
        ("2026-10-27", "Lakers", "Bulls", 150, 150),
        ("2026-10-28", "Bulls", "Lakers", 145, 145),
    ]
    df = pd.DataFrame(
        rows,
        columns=["game_date", "home_team", "away_team", "home_score", "away_score"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["total_score"] = df["home_score"] + df["away_score"]
    return df


class TotalsBaselineFallbackTests(unittest.TestCase):
    def test_rest_days_are_measured_to_target_date(self):
        history = sample_history()
        prior = history.loc[history["game_date"] < pd.Timestamp("2026-10-27")]

        self.assertEqual(
            calculate_team_rest_days("Lakers", prior, game_date="2026-10-27"),
            1,
        )
        self.assertEqual(
            calculate_team_rest_days("Bulls", prior, game_date="2026-10-27"),
            2,
        )

    def test_active_projection_is_points_engine_baseline_only(self):
        history = sample_history()
        target_date = "2026-10-27"
        line = 230.5
        prior = history.loc[history["game_date"] < pd.Timestamp(target_date)]

        expected = calculate_matchup_points(
            home_team="Lakers",
            away_team="Bulls",
            history_df=prior,
            bookmaker_total=None,
            game_date=target_date,
        )
        result = predict_game_total(
            home_team="Lakers",
            away_team="Bulls",
            bookmaker_total=line,
            game_date=target_date,
            history_df=history,
        )

        expected_total = round(float(expected["projected_total"]), 2)
        self.assertEqual(result["history_rows"], len(prior))
        self.assertEqual(result["projected_total"], expected_total)
        self.assertEqual(result["points_engine_total"], expected_total)
        self.assertEqual(result["total_adjustment_after_baseline"], 0.0)
        self.assertEqual(result["projection_mode"], "points_engine_baseline_only")
        self.assertFalse(result["adjustments_applied"])
        self.assertEqual(result["home_rest_days"], 1)
        self.assertEqual(result["away_rest_days"], 2)
        self.assertEqual(
            round(result["projected_home_points"] + result["projected_away_points"], 2),
            result["projected_total"],
        )
        self.assertEqual(result["edge"], round(expected_total - line, 2))
        self.assertEqual(
            result["recommendation"],
            classify_total_edge(result["edge"])[0],
        )

    def test_target_day_and_future_scores_do_not_change_prediction(self):
        history = sample_history()
        target_date = "2026-10-27"

        with_future_rows = predict_game_total(
            "Lakers", "Bulls", 230.5, game_date=target_date, history_df=history
        )
        only_prior_rows = history.loc[
            history["game_date"] < pd.Timestamp(target_date)
        ]
        prior_only = predict_game_total(
            "Lakers", "Bulls", 230.5, game_date=target_date, history_df=only_prior_rows
        )

        self.assertEqual(with_future_rows["history_rows"], len(only_prior_rows))
        self.assertEqual(
            with_future_rows["projected_total"],
            prior_only["projected_total"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
