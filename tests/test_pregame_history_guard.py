import unittest

import pandas as pd

from pregame_history_guard import (
    history_strictly_before_target_date,
    parse_target_calendar_date,
)


class PregameHistoryGuardTests(unittest.TestCase):
    def test_excludes_target_lagos_calendar_day_and_all_future_rows(self):
        history = pd.DataFrame([
            {"date": "2026-10-26T18:00:00Z", "home_team_name": "A", "home_score": 100},
            # UTC previous day, but already the target day in Lagos: exclude.
            {"date": "2026-10-26T23:30:00Z", "home_team_name": "B", "home_score": 999},
            {"date": "2026-10-27T22:30:00Z", "home_team_name": "C", "home_score": 998},
            # After midnight in Lagos: also exclude.
            {"date": "2026-10-27T23:30:00Z", "home_team_name": "D", "home_score": 997},
            {"date": "2026-10-28", "home_team_name": "E", "home_score": 996},
        ])

        filtered = history_strictly_before_target_date(history, "10/27/2026")

        self.assertEqual(filtered["home_team_name"].tolist(), ["A"])
        self.assertEqual(filtered["home_score"].tolist(), [100])
        self.assertTrue((filtered["date"] < pd.Timestamp("2026-10-27")).all())

    def test_equivalent_history_already_cut_off_produces_same_result(self):
        history = pd.DataFrame([
            {"date": "2026-10-25", "value": 1},
            {"date": "2026-10-26", "value": 2},
            {"date": "2026-10-27", "value": 10000},
            {"date": "2026-10-28", "value": 20000},
        ])
        already_filtered = history.iloc[:2].copy()

        from_full = history_strictly_before_target_date(history, "2026-10-27")
        from_prior = history_strictly_before_target_date(already_filtered, "2026-10-27")

        self.assertEqual(from_full["value"].tolist(), from_prior["value"].tolist())
        self.assertEqual(from_full["date"].tolist(), from_prior["date"].tolist())

    def test_invalid_history_dates_are_dropped_not_passed_through(self):
        history = pd.DataFrame([
            {"date": "not-a-date", "value": 1},
            {"date": "2026-10-26", "value": 2},
        ])

        filtered = history_strictly_before_target_date(history, "2026-10-27")

        self.assertEqual(filtered["value"].tolist(), [2])

    def test_missing_history_date_column_fails_closed(self):
        history = pd.DataFrame([{"home_team_name": "A", "home_score": 100}])
        with self.assertRaisesRegex(ValueError, "no date/game_date column"):
            history_strictly_before_target_date(history, "2026-10-27")

    def test_invalid_target_date_fails_closed(self):
        history = pd.DataFrame([{"date": "2026-10-26", "value": 1}])
        with self.assertRaisesRegex(ValueError, "Invalid game date"):
            history_strictly_before_target_date(history, "definitely-not-a-date")

    def test_naive_datetime_is_treated_as_a_calendar_date(self):
        self.assertEqual(
            parse_target_calendar_date("2026-10-27"),
            pd.Timestamp("2026-10-27").date(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
