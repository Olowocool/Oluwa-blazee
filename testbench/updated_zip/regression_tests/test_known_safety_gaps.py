"""Regression tests for safety gaps found in the uploaded source snapshot."""

import os
import sys
import unittest
from unittest import mock

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import betting_guards
import points_engine
import totals_model


class MoneylineInputValidation(unittest.TestCase):
    def test_rejects_decimal_odds_at_or_below_one(self):
        for bad_odds in (1.0, 0.0, -1.5):
            with self.subTest(odds=bad_odds):
                with self.assertRaises(ValueError):
                    betting_guards.assess_moneyline(0.55, bad_odds, 2.0)

    def test_rejects_non_finite_decimal_odds(self):
        for bad_odds in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(odds=bad_odds):
                with self.assertRaises(ValueError):
                    betting_guards.assess_moneyline(0.55, bad_odds, 2.0)

    def test_rejects_probabilities_outside_zero_to_one(self):
        for bad_probability in (-0.01, 1.01):
            with self.subTest(probability=bad_probability):
                with self.assertRaises(ValueError):
                    betting_guards.assess_moneyline(bad_probability, 1.9, 2.0)

    def test_rejects_non_finite_probabilities(self):
        for bad_probability in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(probability=bad_probability):
                with self.assertRaises(ValueError):
                    betting_guards.assess_moneyline(bad_probability, 1.9, 2.0)

    def test_accepts_valid_boundary_probabilities(self):
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                result = betting_guards.assess_moneyline(probability, 1.9, 2.0)
                self.assertIn("blocked", result)


class TargetDateFiltering(unittest.TestCase):
    def history(self):
        return pd.DataFrame(
            [
                # Completed before the target date: these are eligible.
                ("2026-01-01", "A", "B", 110, 100),
                ("2026-01-02", "B", "A", 90, 100),
                # Same-day and future results must not leak into the forecast.
                ("2026-01-03", "A", "B", 200, 180),
                ("2026-01-04", "A", "B", 220, 210),
            ],
            columns=["game_date", "home_team", "away_team", "home_score", "away_score"],
        )

    def test_points_engine_accepts_target_date_and_excludes_same_day_and_future(self):
        history = self.history()
        with_target_date = points_engine.calculate_matchup_points(
            "A", "B", history_df=history, game_date="2026-01-03"
        )
        prior_only = points_engine.calculate_matchup_points(
            "A", "B", history_df=history.iloc[:2].copy()
        )

        self.assertEqual(with_target_date["h2h_games_used"], 2)
        self.assertEqual(with_target_date["projected_total"], prior_only["projected_total"])
        self.assertEqual(with_target_date["projected_home_points"], prior_only["projected_home_points"])
        self.assertEqual(with_target_date["projected_away_points"], prior_only["projected_away_points"])

    def test_target_date_requires_a_valid_history_date_column(self):
        with self.assertRaises(ValueError):
            points_engine.filter_history_before_date(
                pd.DataFrame({"home_team": ["A"]}), "2026-01-03"
            )
        with self.assertRaises(ValueError):
            points_engine.filter_history_before_date(self.history(), "not-a-date")

    def test_totals_model_passes_target_date_and_prior_only_history_to_points_engine(self):
        target = pd.Timestamp("2026-01-03")
        history = self.history()
        captured = {}

        def points_spy(**kwargs):
            captured["points_history"] = kwargs["history_df"].copy()
            captured["points_game_date"] = kwargs["game_date"]
            return {
                "projected_home_points": 112.0,
                "projected_away_points": 108.0,
                "projected_total": 220.0,
                "h2h_average_total": 210.0,
                "h2h_adjustment": -1.5,
                "h2h_games_used": 2,
            }

        pace_result = {
            "home_pace": 100.0,
            "away_pace": 99.0,
            "combined_pace": 99.5,
            "pace_adjustment": 0.0,
            "pace_gap": 1.0,
            "home_last_10_pace": 100.0,
            "home_last_20_pace": 100.0,
            "home_season_pace": 100.0,
            "home_venue_pace": 100.0,
            "away_last_10_pace": 99.0,
            "away_last_20_pace": 99.0,
            "away_season_pace": 99.0,
            "away_venue_pace": 99.0,
            "home_pace_games_used": 2,
            "away_pace_games_used": 2,
        }
        injury_result = {
            "home_injury_penalty": 0.0,
            "away_injury_penalty": 0.0,
            "injury_adjustment": 0.0,
            "home_missing_players": [],
            "away_missing_players": [],
        }

        def pace_spy(frame, *_args, **_kwargs):
            captured["pace_history"] = frame.copy()
            return pace_result

        with (
            mock.patch.object(totals_model, "load_history", return_value=history),
            mock.patch.object(totals_model, "calculate_matchup_points", side_effect=points_spy),
            mock.patch.object(totals_model, "calculate_matchup_pace", side_effect=pace_spy),
            mock.patch.object(totals_model, "get_injury_impact", return_value=injury_result),
            mock.patch("possession_total_model.get_model", side_effect=RuntimeError("force fallback")),
        ):
            result = totals_model.predict_game_total("A", "B", 220.5, game_date=target)

        self.assertEqual(captured["points_game_date"], target)
        for key in ("points_history", "pace_history"):
            observed_dates = pd.to_datetime(captured[key]["game_date"])
            self.assertTrue((observed_dates < target).all(), key)
            self.assertEqual(len(captured[key]), 2, key)
        self.assertEqual(result["history_rows"], 2)
        self.assertEqual(result["model_used"], "points_engine")


if __name__ == "__main__":
    unittest.main()
