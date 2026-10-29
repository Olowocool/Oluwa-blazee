import unittest

import pandas as pd

from points_engine import calculate_matchup_points


class PointsEngineLeakageGuardTests(unittest.TestCase):
    def setUp(self):
        # Deliberately omit total_score so the forecast API must rebuild it.
        # The extreme target-day and future outcomes must not affect features.
        self.history = pd.DataFrame([
            {
                "game_date": "2026-10-01",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_date": "2026-10-02",
                "home_team": "Miami Heat",
                "away_team": "Boston Celtics",
                "home_score": 105,
                "away_score": 112,
            },
            {
                "game_date": "2026-10-03",
                "home_team": "Philadelphia 76ers",
                "away_team": "New York Knicks",
                "home_score": 101,
                "away_score": 99,
            },
            {
                "game_date": "2026-10-10",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "home_score": 155,
                "away_score": 150,
            },
            {
                "game_date": "2026-10-11",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "home_score": 160,
                "away_score": 158,
            },
        ])

    def test_direct_forecast_excludes_target_day_and_future_rows(self):
        target_date = "2026-10-10"
        prior_only = self.history.loc[
            pd.to_datetime(self.history["game_date"]) < pd.Timestamp(target_date)
        ].copy()

        from_full_table = calculate_matchup_points(
            "Boston Celtics",
            "New York Knicks",
            history_df=self.history,
            game_date=target_date,
        )
        from_prior_only = calculate_matchup_points(
            "Boston Celtics",
            "New York Knicks",
            history_df=prior_only,
            game_date=target_date,
        )

        self.assertEqual(from_full_table["history_rows"], 3)
        self.assertEqual(from_full_table["h2h_games_used"], 1)
        self.assertEqual(from_full_table["h2h_average_total"], 210.0)
        self.assertEqual(
            from_full_table["projected_total"],
            from_prior_only["projected_total"],
        )
        self.assertEqual(
            from_full_table["projected_home_points"],
            from_prior_only["projected_home_points"],
        )
        self.assertEqual(
            from_full_table["projected_away_points"],
            from_prior_only["projected_away_points"],
        )

    def test_cutoff_marker_cannot_admit_rows_appended_after_cutoff(self):
        target_date = "2026-10-10"
        prior_only = self.history.loc[
            pd.to_datetime(self.history["game_date"]) < pd.Timestamp(target_date)
        ].copy()
        prior_only["game_date"] = pd.to_datetime(prior_only["game_date"])
        prior_only["total_score"] = prior_only["home_score"] + prior_only["away_score"]
        prior_only.attrs["_strict_history_cutoff_date"] = target_date

        # A caller that appends a same-day result must not be able to reuse the
        # marker to bypass the check.
        target_row = self.history.loc[
            pd.to_datetime(self.history["game_date"]) == pd.Timestamp(target_date)
        ].copy()
        target_row["game_date"] = pd.to_datetime(target_row["game_date"])
        target_row["total_score"] = target_row["home_score"] + target_row["away_score"]
        appended = pd.concat([prior_only, target_row], ignore_index=True)
        appended.attrs["_strict_history_cutoff_date"] = target_date

        result = calculate_matchup_points(
            "Boston Celtics",
            "New York Knicks",
            history_df=appended,
            game_date=target_date,
        )
        self.assertEqual(result["history_rows"], 3)
        self.assertEqual(result["h2h_games_used"], 1)

    def test_stale_cached_totals_are_rebuilt_from_scores(self):
        history = self.history.copy()
        history["total_score"] = 1
        result = calculate_matchup_points(
            "Boston Celtics",
            "New York Knicks",
            history_df=history,
            game_date="2026-10-10",
        )
        self.assertEqual(result["h2h_average_total"], 210.0)

    def test_missing_date_column_fails_closed(self):
        history = self.history.drop(columns=["game_date"])
        with self.assertRaisesRegex(ValueError, "historical cutoff.*game_date/date column"):
            calculate_matchup_points(
                "Boston Celtics",
                "New York Knicks",
                history_df=history,
                game_date="2026-10-10",
            )

    def test_invalid_target_date_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Invalid target game date"):
            calculate_matchup_points(
                "Boston Celtics",
                "New York Knicks",
                history_df=self.history,
                game_date="not-a-date",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
