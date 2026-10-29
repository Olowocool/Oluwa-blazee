import unittest

import pandas as pd

from pace_engine import (
    _prepare_history,
    _team_games as pace_team_games,
    _venue_games,
    calculate_team_advanced_pace,
)
from points_engine import (
    calculate_team_points_features,
    get_head_to_head_games,
    get_team_games,
)
from team_name_utils import normalize_history_team_columns, normalize_team_name
from totals_model import get_team_recent_stats


def mixed_alias_history():
    return pd.DataFrame(
        [
            {
                "game_date": "2025-11-01",
                "home_team": "LA Clippers",
                "away_team": "Boston Celtics",
                "home_score": 111,
                "away_score": 100,
            },
            {
                "game_date": "2025-11-03",
                "home_team": "Boston Celtics",
                "away_team": "Los Angeles Clippers",
                "home_score": 105,
                "away_score": 109,
            },
            {
                "game_date": "2025-11-05",
                "home_team": "LA Clippers",
                "away_team": "Los Angeles Lakers",
                "home_score": 120,
                "away_score": 118,
            },
            {
                "game_date": "2025-11-07",
                "home_team": "Los Angeles Lakers",
                "away_team": "Los Angeles Clippers",
                "home_score": 99,
                "away_score": 107,
            },
        ]
    )


class TeamNameNormalizationTests(unittest.TestCase):
    def test_known_aliases_share_canonical_names(self):
        cases = {
            "LA Clippers": "Los Angeles Clippers",
            "LOS ANGELES CLIPPERS": "Los Angeles Clippers",
            "LA Lakers": "Los Angeles Lakers",
            "NY Knicks": "New York Knicks",
            "OKC Thunder": "Oklahoma City Thunder",
            "Philadelphia Sixers": "Philadelphia 76ers",
            "Portland Blazers": "Portland Trail Blazers",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_team_name(raw), expected)

        # Unknown / historical franchise labels remain visible rather than
        # being silently forced into a current-team name.
        self.assertEqual(normalize_team_name("Seattle SuperSonics"), "Seattle SuperSonics")

    def test_history_normalizer_is_cached_and_idempotent(self):
        history = mixed_alias_history()
        normalized = normalize_history_team_columns(history)
        self.assertTrue(normalized.attrs.get("_team_names_normalized"))
        self.assertEqual(
            set(normalized["home_team"]) | set(normalized["away_team"]),
            {"Los Angeles Clippers", "Boston Celtics", "Los Angeles Lakers"},
        )
        second = normalize_history_team_columns(normalized)
        pd.testing.assert_frame_equal(normalized, second)
        self.assertEqual(
            int(
                (
                    (second["home_team"] == "Los Angeles Clippers")
                    | (second["away_team"] == "Los Angeles Clippers")
                ).sum()
            ),
            4,
        )

    def test_points_engine_recent_and_head_to_head_features_combine_aliases(self):
        history = mixed_alias_history()
        selected = get_team_games(history, "LA Clippers")
        self.assertEqual(len(selected), 4)
        self.assertTrue(
            selected["home_team"].eq("Los Angeles Clippers").any()
            or selected["away_team"].eq("Los Angeles Clippers").any()
        )

        features = calculate_team_points_features(
            "Los Angeles Clippers", history, last_n=10
        )
        self.assertEqual(features["games_used"], 4)
        self.assertAlmostEqual(features["ppg"], (111 + 109 + 120 + 107) / 4)

        h2h = get_head_to_head_games(
            history, "LA Clippers", "Boston Celtics", limit=10
        )
        self.assertEqual(len(h2h), 2)

    def test_pace_engine_combines_aliases_and_venue_splits(self):
        history = _prepare_history(mixed_alias_history())
        selected = pace_team_games(history, "LA Clippers")
        self.assertEqual(len(selected), 4)

        home = _venue_games(selected, "Los Angeles Clippers", "home")
        away = _venue_games(selected, "LA Clippers", "away")
        self.assertEqual(len(home), 2)
        self.assertEqual(len(away), 2)

        pace = calculate_team_advanced_pace(
            mixed_alias_history(), "LA Clippers", venue="home"
        )
        self.assertEqual(pace["games_used"], 4)

    def test_totals_recent_stats_combines_aliases(self):
        stats = get_team_recent_stats(
            mixed_alias_history(), "LA Clippers", game_date="2025-11-08"
        )
        self.assertAlmostEqual(stats["last_10_scored"], (111 + 109 + 120 + 107) / 4)
        self.assertAlmostEqual(stats["home_split"], (111 + 120) / 2)
        self.assertAlmostEqual(stats["away_split"], (109 + 107) / 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
