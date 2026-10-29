import unittest

import pandas as pd

from points_engine import calculate_matchup_points
from points_engine_core_ablation import (
    VARIANTS,
    VARIANT_BY_NAME,
    compute_core_components,
    predict_core_total,
    predict_from_components,
)


def synthetic_history():
    rows = []
    start = pd.Timestamp("2026-10-01")
    # Historical head-to-head results create a measurable H2H adjustment.
    for i in range(14):
        rows.append({
            "game_date": start + pd.Timedelta(days=i),
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 120 + (i % 4) * 3,
            "away_score": 110 + (i % 3) * 2,
            "total_score": 230 + (i % 4) * 3 + (i % 3) * 2,
        })
    return pd.DataFrame(rows)


class PointsEngineCoreAblationTests(unittest.TestCase):
    def test_baseline_variant_matches_official_points_engine_total(self):
        history = synthetic_history()
        target_date = "2026-10-20"

        official = calculate_matchup_points(
            "Home",
            "Away",
            history_df=history,
            bookmaker_total=None,
            game_date=target_date,
        )
        candidate = predict_core_total(
            "Home",
            "Away",
            history_df=history,
            game_date=target_date,
            variant=VARIANT_BY_NAME["baseline"],
        )

        self.assertEqual(candidate["projected_total"], official["projected_total"])
        self.assertEqual(candidate["history_rows"], len(history))

    def test_target_day_results_are_not_used_as_features(self):
        history = synthetic_history()
        target_row = pd.DataFrame([{
            "game_date": pd.Timestamp("2026-10-20"),
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 150,
            "away_score": 149,
            "total_score": 299,
        }])
        with_target_day = pd.concat([history, target_row], ignore_index=True)

        prediction_with_target_row = predict_core_total(
            "Home", "Away", with_target_day, "2026-10-20", VARIANT_BY_NAME["baseline"]
        )
        prediction_without_target_row = predict_core_total(
            "Home", "Away", history, "2026-10-20", VARIANT_BY_NAME["baseline"]
        )

        self.assertEqual(prediction_with_target_row["history_rows"], len(history))
        self.assertEqual(
            prediction_with_target_row["projected_total"],
            prediction_without_target_row["projected_total"],
        )

    def test_no_h2h_variant_removes_only_the_h2h_adjustment(self):
        components = {
            "home_last_5": 118.0,
            "home_last_10": 116.0,
            "home_opponent_allowed": 112.0,
            "home_venue_points": 117.0,
            "away_last_5": 110.0,
            "away_last_10": 112.0,
            "away_opponent_allowed": 119.0,
            "away_venue_points": 111.0,
            "home_rest_days": 2,
            "away_rest_days": 2,
            "h2h_average_total": 260.0,
            "h2h_games_used": 8,
            "history_rows": 100,
        }

        baseline = predict_from_components(components, VARIANT_BY_NAME["baseline"])
        no_h2h = predict_from_components(components, VARIANT_BY_NAME["no_h2h"])

        self.assertNotEqual(baseline["projected_total"], no_h2h["projected_total"])
        self.assertEqual(no_h2h["h2h_adjustment"], 0.0)
        self.assertEqual(no_h2h["rest_adjustment"], baseline["rest_adjustment"])

    def test_no_rest_variant_removes_only_the_rest_adjustment(self):
        components = {
            "home_last_5": 118.0,
            "home_last_10": 116.0,
            "home_opponent_allowed": 112.0,
            "home_venue_points": 117.0,
            "away_last_5": 110.0,
            "away_last_10": 112.0,
            "away_opponent_allowed": 119.0,
            "away_venue_points": 111.0,
            "home_rest_days": 3,
            "away_rest_days": 2,
            "h2h_average_total": 235.0,
            "h2h_games_used": 8,
            "history_rows": 100,
        }

        baseline = predict_from_components(components, VARIANT_BY_NAME["baseline"])
        no_rest = predict_from_components(components, VARIANT_BY_NAME["no_rest"])

        self.assertNotEqual(baseline["projected_total"], no_rest["projected_total"])
        self.assertEqual(no_rest["rest_adjustment"], 0.0)
        self.assertEqual(no_rest["h2h_adjustment"], baseline["h2h_adjustment"])

    def test_predeclared_variants_have_valid_weights(self):
        self.assertIn("baseline", {variant.name for variant in VARIANTS})
        self.assertGreaterEqual(len(VARIANTS), 5)
        for variant in VARIANTS:
            with self.subTest(variant=variant.name):
                variant.validate()
                self.assertAlmostEqual(sum(variant.weights), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
