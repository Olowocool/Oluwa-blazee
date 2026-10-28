import unittest

import pandas as pd

from totals_full_model_backtest import (
    build_ablation_predictions,
    history_strictly_before,
    prepare_backtest_history,
    regression_metrics,
)
from points_engine import calculate_team_rest_days
from totals_model import get_team_recent_stats


class ChronologicalTotalsBacktestTests(unittest.TestCase):
    def test_prepare_history_sorts_and_rebuilds_total(self):
        history = pd.DataFrame({
            "game_date": ["2026-10-26", "2026-10-20", "not-a-date"],
            "home_team": ["B", "A", "Bad"],
            "away_team": ["D", "C", "Row"],
            "home_score": [110, 112, 100],
            "away_score": [105, 108, 99],
            "total_score": [1, 2, 3],  # recomputed from final scores
        })

        prepared = prepare_backtest_history(history)

        self.assertEqual(
            prepared["game_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-10-20", "2026-10-26"],
        )
        self.assertEqual(prepared["total_score"].tolist(), [220, 215])

    def test_history_snapshot_excludes_target_day_and_future(self):
        history = prepare_backtest_history(pd.DataFrame({
            "game_date": ["2026-10-20", "2026-10-26", "2026-10-27", "2026-10-28"],
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
            "home_score": [100, 101, 102, 103],
            "away_score": [99, 98, 97, 96],
        }))

        prior = history_strictly_before(history, "2026-10-27")
        self.assertEqual(
            prior["game_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-10-20", "2026-10-26"],
        )
        self.assertTrue((prior["game_date"] < pd.Timestamp("2026-10-27")).all())

    def test_rest_days_use_target_date_not_gap_between_prior_games(self):
        history = prepare_backtest_history(pd.DataFrame({
            "game_date": ["2026-10-20", "2026-10-25"],
            "home_team": ["Lakers", "Bulls"],
            "away_team": ["Bulls", "Lakers"],
            "home_score": [110, 105],
            "away_score": [100, 99],
        }))

        # Lakers last played Oct 25. For an Oct 27 game, Oct 26 is one rest day.
        # The old implementation incorrectly used the gap between Oct 20 and Oct 25.
        self.assertEqual(
            calculate_team_rest_days("Lakers", history, game_date="2026-10-27"),
            1,
        )
        stats = get_team_recent_stats(
            history,
            "Lakers",
            game_date="2026-10-27",
        )
        self.assertEqual(stats["rest_days"], 1)

        # A game the day immediately following Oct 25 gives zero rest days.
        self.assertEqual(
            calculate_team_rest_days("Lakers", history, game_date="2026-10-26"),
            0,
        )

    def test_ablation_predictions_remove_or_add_only_selected_component(self):
        baseline = 220.0
        full_prediction = {
            "projected_total": 225.0,
            "pace_adjustment": 2.0,
            "offensive_adjustment": 1.5,
            "defensive_adjustment": -0.5,
            "home_away_adjustment": 2.0,
        }

        predictions, components = build_ablation_predictions(
            baseline,
            full_prediction,
        )

        self.assertEqual(predictions["points_engine_projected_total"], 220.0)
        self.assertEqual(predictions["full_model_projected_total"], 225.0)
        self.assertEqual(predictions["full_without_pace_total"], 223.0)
        self.assertEqual(predictions["full_without_offense_total"], 223.5)
        self.assertEqual(predictions["full_without_defense_total"], 225.5)
        self.assertEqual(predictions["full_without_venue_total"], 223.0)
        self.assertEqual(predictions["baseline_plus_pace_total"], 222.0)
        self.assertEqual(predictions["baseline_plus_offense_total"], 221.5)
        self.assertEqual(predictions["baseline_plus_defense_total"], 219.5)
        self.assertEqual(predictions["baseline_plus_venue_total"], 222.0)
        self.assertEqual(components["venue"], 2.0)

    def test_metrics_calculate_errors_and_hit_ranges(self):
        metrics = regression_metrics(
            actual=[220, 230],
            predicted=[225, 225],
        )
        self.assertEqual(metrics["games"], 2)
        self.assertEqual(metrics["mae"], 5.0)
        self.assertEqual(metrics["rmse"], 5.0)
        self.assertEqual(metrics["bias"], 0.0)
        self.assertEqual(metrics["within_5_pct"], 100.0)
        self.assertEqual(metrics["within_10_pct"], 100.0)

    def test_metrics_handle_empty_inputs(self):
        metrics = regression_metrics(actual=[], predicted=[])
        self.assertEqual(metrics["games"], 0)
        self.assertTrue(pd.isna(metrics["mae"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
