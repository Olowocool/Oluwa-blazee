import unittest

import pandas as pd

from points_engine import calculate_team_points_features


class NeutralSitePointsEngineTests(unittest.TestCase):
    def test_neutral_games_count_for_form_but_not_venue_splits(self):
        history = pd.DataFrame([
            {
                "game_date": "2025-12-13",
                "home_team": "New York Knicks",
                "away_team": "Orlando Magic",
                "home_score": 132,
                "away_score": 120,
                "is_neutral_site": True,
            },
            {
                "game_date": "2025-12-10",
                "home_team": "New York Knicks",
                "away_team": "Boston Celtics",
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_date": "2025-12-08",
                "home_team": "Miami Heat",
                "away_team": "New York Knicks",
                "home_score": 90,
                "away_score": 100,
            },
        ])

        features = calculate_team_points_features(
            "New York Knicks",
            history_df=history,
            last_n=10,
        )

        # Neutral-court 132 points count in overall recent form:
        # (132 + 110 + 100) / 3 = 114.
        self.assertEqual(features["games_used"], 3)
        self.assertAlmostEqual(features["ppg"], 114.0)

        # But a neutral-court final must not be mislabelled as a home game.
        self.assertAlmostEqual(features["home_split"], 110.0)
        self.assertAlmostEqual(features["away_split"], 100.0)

    def test_missing_neutral_flag_keeps_legacy_history_behavior(self):
        history = pd.DataFrame([
            {
                "game_date": "2025-12-10",
                "home_team": "New York Knicks",
                "away_team": "Boston Celtics",
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_date": "2025-12-08",
                "home_team": "Miami Heat",
                "away_team": "New York Knicks",
                "home_score": 90,
                "away_score": 100,
            },
        ])

        features = calculate_team_points_features(
            "New York Knicks",
            history_df=history,
            last_n=10,
        )

        self.assertAlmostEqual(features["home_split"], 110.0)
        self.assertAlmostEqual(features["away_split"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
