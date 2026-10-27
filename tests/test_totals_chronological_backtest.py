import unittest

import pandas as pd

from totals_full_model_backtest import (
    history_strictly_before,
    prepare_backtest_history,
    regression_metrics,
)


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
