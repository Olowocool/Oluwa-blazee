import unittest

import numpy as np
import pandas as pd

from pregame_moneyline_pipeline import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_pregame_dataset,
    prepare_score_history,
    train_chronological_model,
)


class PregameMoneylinePipelineTests(unittest.TestCase):
    def test_same_day_results_never_enter_same_day_features(self):
        history = pd.DataFrame([
            {"game_date": "2026-01-01", "home_team": "Team A", "away_team": "Team B", "home_score": 100, "away_score": 90},
            {"game_date": "2026-01-02", "home_team": "Team A", "away_team": "Team C", "home_score": 111, "away_score": 88},
            # A different game on the same date must not influence Team C's features.
            {"game_date": "2026-01-02", "home_team": "Team D", "away_team": "Team C", "home_score": 180, "away_score": 70},
        ])

        dataset = build_pregame_dataset(history)
        target_a = dataset[
            (dataset["game_date"] == pd.Timestamp("2026-01-02"))
            & (dataset["home_team"] == "Team A")
        ].iloc[0]
        target_c = dataset[
            (dataset["game_date"] == pd.Timestamp("2026-01-02"))
            & (dataset["away_team"] == "Team C")
        ].iloc[0]

        self.assertEqual(target_a["home_points_for_last5"], 100.0)
        self.assertEqual(target_a["home_games_prior"], 1.0)
        self.assertEqual(target_c["away_games_prior"], 0.0)
        self.assertEqual(target_c["away_points_for_last5"], 114.0)

    def test_future_rows_do_not_change_earlier_features(self):
        base = pd.DataFrame([
            {"game_date": "2026-01-01", "home_team": "Team A", "away_team": "Team B", "home_score": 100, "away_score": 90},
            {"game_date": "2026-01-02", "home_team": "Team C", "away_team": "Team D", "home_score": 105, "away_score": 102},
            {"game_date": "2026-01-03", "home_team": "Team A", "away_team": "Team C", "home_score": 108, "away_score": 104},
        ])
        future = pd.concat([
            base,
            pd.DataFrame([
                {"game_date": "2026-01-10", "home_team": "Team A", "away_team": "Team B", "home_score": 250, "away_score": 60},
                {"game_date": "2026-01-11", "home_team": "Team C", "away_team": "Team D", "home_score": 40, "away_score": 240},
            ]),
        ], ignore_index=True)

        before = build_pregame_dataset(base)
        after = build_pregame_dataset(future)
        key = ["game_date", "home_team", "away_team"]
        merged = before.merge(after, on=key, suffixes=("_before", "_after"))
        for col in FEATURE_COLUMNS:
            np.testing.assert_allclose(
                merged[f"{col}_before"],
                merged[f"{col}_after"],
                err_msg=f"Future rows changed feature {col}",
            )

    def test_row_order_within_date_does_not_change_features(self):
        history = pd.DataFrame([
            {"game_date": "2026-01-01", "home_team": "A", "away_team": "B", "home_score": 100, "away_score": 90},
            {"game_date": "2026-01-02", "home_team": "C", "away_team": "D", "home_score": 120, "away_score": 101},
            {"game_date": "2026-01-02", "home_team": "A", "away_team": "C", "home_score": 105, "away_score": 104},
            {"game_date": "2026-01-03", "home_team": "B", "away_team": "D", "home_score": 110, "away_score": 98},
        ])
        shuffled = history.iloc[[0, 2, 1, 3]].reset_index(drop=True)

        left = build_pregame_dataset(history)
        right = build_pregame_dataset(shuffled)
        keys = ["game_date", "home_team", "away_team"]
        compare_cols = keys + FEATURE_COLUMNS
        pd.testing.assert_frame_equal(
            left[compare_cols].reset_index(drop=True),
            right[compare_cols].reset_index(drop=True),
        )

    def test_score_preparation_drops_invalid_and_duplicate_rows(self):
        history = pd.DataFrame([
            {"date": "2026-01-01", "home_team_name": "LA Lakers", "away_team_name": "Team B", "home_score": 100, "away_score": 90},
            {"date": "2026-01-01", "home_team": "Team B", "away_team": "Los Angeles Lakers", "home_score": 90, "away_score": 100},
            {"date": "bad date", "home_team": "A", "away_team": "B", "home_score": 90, "away_score": 100},
            {"date": "2026-01-02", "home_team": "A", "away_team": "A", "home_score": 110, "away_score": 100},
        ])

        # The two rows disagree on venue/home label for the same unordered
        # matchup/date, so neither row is safe to keep and preparation fails closed.
        with self.assertRaisesRegex(ValueError, "No valid completed games"):
            prepare_score_history(history)

    def test_identical_duplicate_records_are_collapsed(self):
        history = pd.DataFrame([
            {"date": "2026-01-01", "home_team": "Team A", "away_team": "Team B", "home_score": 100, "away_score": 90},
            {"date": "2026-01-01", "home_team": "Team A", "away_team": "Team B", "home_score": 100, "away_score": 90},
        ])

        prepared = prepare_score_history(history)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.iloc[0]["home_team"], "Team A")

    def test_training_uses_chronological_holdout_and_never_uses_postgame_fields(self):
        rows = []
        team_names = [f"Team {i}" for i in range(8)]
        for day in range(50):
            date_text = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            for game_no in range(4):
                home = team_names[game_no * 2]
                away = team_names[game_no * 2 + 1]
                # Vary home results and scores so both classes exist on both sides of the split.
                home_score = 105 + ((day + game_no) % 17)
                away_score = 104 + ((day * 3 + game_no * 2) % 17)
                if home_score == away_score:
                    home_score += 1
                rows.append({
                    "game_date": date_text,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "profit_loss": 999999,
                    "clv": 999999,
                })
        dataset = build_pregame_dataset(pd.DataFrame(rows))
        result = train_chronological_model(
            dataset,
            holdout_fraction=0.2,
            min_training_rows=20,
            min_unique_dates=20,
        )

        self.assertEqual(result["status"], "success")
        self.assertLess(result["training_through"], result["holdout_from"])
        self.assertEqual(result["holdout_from"], dataset.loc[
            dataset["game_date"] >= pd.Timestamp(result["holdout_from"]), "game_date"
        ].min().date().isoformat())
        self.assertEqual(result["feature_count"], len(FEATURE_COLUMNS))
        self.assertNotIn("profit_loss", result["feature_columns"])
        self.assertNotIn("clv", result["feature_columns"])
        self.assertNotIn(LABEL_COLUMN, result["feature_columns"])
        self.assertEqual(result["validation_status"], "research_only")
        self.assertFalse(result["model_written"])

    def test_insufficient_history_is_blocked_without_writing_a_model(self):
        dataset = build_pregame_dataset(pd.DataFrame([
            {"game_date": "2026-01-01", "home_team": "A", "away_team": "B", "home_score": 100, "away_score": 90},
            {"game_date": "2026-01-02", "home_team": "A", "away_team": "C", "home_score": 110, "away_score": 95},
        ]))
        result = train_chronological_model(
            dataset,
            min_training_rows=1,
            min_unique_dates=10,
        )

        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["model_written"])
        self.assertFalse(result["metrics_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
