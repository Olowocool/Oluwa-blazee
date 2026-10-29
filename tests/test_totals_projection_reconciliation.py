import unittest
from unittest.mock import patch

import pandas as pd

import totals_model


class TotalsProjectionReconciliationTests(unittest.TestCase):
    def test_team_projections_sum_to_final_total_and_rest_is_applied_once(self):
        history = pd.DataFrame({
            "game_date": ["2026-10-20"],
            "home_team": ["Home"],
            "away_team": ["Opponent"],
            "home_score": [110],
            "away_score": [100],
            "total_score": [210],
        })

        points_data = {
            "projected_home_points": 110.0,
            "projected_away_points": 112.0,
            # This baseline includes the Points Engine's 0.50 rest adjustment.
            "projected_total": 222.5,
            "h2h_average_total": 220.0,
            "h2h_adjustment": 0.0,
            "h2h_games_used": 3,
        }

        home_stats = {
            "last_5_scored": 117.0,
            "last_10_scored": 116.0,
            "last_10_allowed": 115.0,
            "home_split": 116.0,
            "away_split": 115.0,
            "rest_days": 2,
        }
        away_stats = {
            "last_5_scored": 113.0,
            "last_10_scored": 112.0,
            "last_10_allowed": 116.0,
            "home_split": 114.0,
            "away_split": 112.0,
            "rest_days": 1,
        }

        pace_data = {
            "home_pace": 100.0,
            "away_pace": 101.0,
            "combined_pace": 100.5,
            "pace_adjustment": 2.0,
            "pace_gap": 0.5,
            "home_last_10_pace": 100.0,
            "home_last_20_pace": 100.0,
            "home_season_pace": 100.0,
            "home_venue_pace": 100.0,
            "away_last_10_pace": 101.0,
            "away_last_20_pace": 101.0,
            "away_season_pace": 101.0,
            "away_venue_pace": 101.0,
            "home_pace_games_used": 10,
            "away_pace_games_used": 10,
        }

        injury_data = {
            "home_injury_penalty": 0.0,
            "away_injury_penalty": 0.0,
            "injury_adjustment": 0.0,
            "home_missing_players": [],
            "away_missing_players": [],
        }

        with patch("totals_model.load_history", return_value=history), \
             patch("totals_model.calculate_matchup_points", return_value=points_data), \
             patch("totals_model.get_team_recent_stats", side_effect=lambda history_df, team, game_date: home_stats if team == "Home" else away_stats), \
             patch("totals_model.calculate_matchup_pace", return_value=pace_data), \
             patch("totals_model.get_injury_impact", return_value=injury_data):
            result = totals_model.predict_game_total(
                "Home",
                "Away",
                bookmaker_total=219.95,
                game_date="2026-10-27",
            )
            experimental_result = totals_model.predict_game_total(
                "Home",
                "Away",
                bookmaker_total=219.95,
                game_date="2026-10-27",
                apply_experimental_adjustments=True,
            )

        # Official default uses the baseline: experimental adjustments are diagnostic only.
        self.assertEqual(result["points_engine_total"], 222.5)
        self.assertEqual(result["rest_adjustment"], 0.5)
        self.assertEqual(result["non_pace_adjustment"], 0.45)
        self.assertEqual(result["experimental_projected_total"], 224.95)
        self.assertFalse(result["experimental_adjustments_applied"])
        self.assertEqual(result["total_adjustment_after_baseline"], 0.0)
        self.assertEqual(result["projected_total"], 222.5)
        self.assertEqual(
            round(result["projected_home_points"] + result["projected_away_points"], 2),
            result["projected_total"],
        )

        # Official edge/recommendation use the baseline total.
        self.assertEqual(result["bookmaker_total"], 219.95)
        self.assertEqual(result["edge"], 2.55)
        self.assertEqual(result["recommendation"], "Lean Over")

        # The prior experimental version remains selectable for honest backtesting.
        self.assertTrue(experimental_result["experimental_adjustments_applied"])
        self.assertEqual(experimental_result["projected_total"], 224.95)
        self.assertEqual(experimental_result["edge"], 5.0)
        self.assertEqual(experimental_result["recommendation"], "Strong Over")
        self.assertEqual(
            round(
                experimental_result["projected_home_points"]
                + experimental_result["projected_away_points"],
                2,
            ),
            experimental_result["projected_total"],
        )

    def test_missing_or_invalid_market_total_fails_closed_to_no_bet(self):
        history = pd.DataFrame({
            "game_date": ["2026-10-20"],
            "home_team": ["Home"],
            "away_team": ["Away"],
            "home_score": [110],
            "away_score": [100],
        })

        for line in [None, "not-a-line", float("nan"), float("inf"), -float("inf"), 0, -1, True]:
            with self.subTest(line=line):
                result = totals_model.predict_game_total(
                    home_team="Home",
                    away_team="Away",
                    bookmaker_total=line,
                    game_date="2026-10-27",
                    history_df=history,
                )
                self.assertEqual(result["recommendation"], "No Bet")
                self.assertEqual(result["confidence_note"], "No valid sportsbook total line")
                self.assertIsNone(result["bookmaker_total"])
                self.assertIsNone(result["edge"])
                self.assertIsInstance(result["projected_total"], float)

    def test_invalid_edges_fail_closed_to_no_bet(self):
        for edge in [None, "bad-edge", float("nan"), float("inf"), -float("inf")]:
            with self.subTest(edge=edge):
                recommendation, note = totals_model.classify_total_edge(edge)
                self.assertEqual(recommendation, "No Bet")
                self.assertEqual(note, "Missing or invalid edge")

    def test_recommendation_thresholds_match_rounded_edge(self):
        cases = [
            (225.00, 220.00, "Strong Over"),
            (222.50, 220.00, "Lean Over"),
            (217.50, 220.00, "Lean Under"),
            (215.00, 220.00, "Strong Under"),
            (222.49, 220.00, "No Bet"),
        ]
        for projected_total, line, expected in cases:
            with self.subTest(projected_total=projected_total, line=line):
                edge = round(round(projected_total, 2) - round(line, 2), 2)
                actual, _ = totals_model.classify_total_edge(edge)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
