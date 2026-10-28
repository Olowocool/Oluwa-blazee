import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from totals_betting_backtest import (
    join_market_to_scores,
    prepare_market_snapshots,
    prepare_score_history,
    select_entry_and_closing_snapshots,
    settle_total_bet,
    side_adjusted_clv,
    generate_predictions,
    summarize_backtest,
)


def snapshot_rows():
    return pd.DataFrame([
        {
            "game_date": "2026-01-02",
            "home_team": "Los Angeles Lakers",
            "away_team": "LA Clippers",
            "bookmaker": "Book A",
            "snapshot_timestamp_utc": "2026-01-02T22:30:00Z",
            "commence_time_utc": "2026-01-02T23:30:00Z",
            "total_line": 220.5,
            "over_odds": 1.91,
            "under_odds": 1.95,
        },
        {
            "game_date": "2026-01-02",
            "home_team": "Los Angeles Lakers",
            "away_team": "LA Clippers",
            "bookmaker": "Book A",
            "snapshot_timestamp_utc": "2026-01-02T23:20:00Z",
            "commence_time_utc": "2026-01-02T23:30:00Z",
            "total_line": 221.5,
            "over_odds": 1.88,
            "under_odds": 1.98,
        },
        {
            # Must be excluded because it is after tipoff.
            "game_date": "2026-01-02",
            "home_team": "Los Angeles Lakers",
            "away_team": "LA Clippers",
            "bookmaker": "Book A",
            "snapshot_timestamp_utc": "2026-01-02T23:35:00Z",
            "commence_time_utc": "2026-01-02T23:30:00Z",
            "total_line": 222.5,
            "over_odds": 1.80,
            "under_odds": 2.05,
        },
    ])


class HistoricalTotalsBettingBacktestTests(unittest.TestCase):
    def test_selects_real_entry_snapshot_and_separate_pregame_close(self):
        normalized, diagnostics = prepare_market_snapshots(snapshot_rows())
        selected, selection = select_entry_and_closing_snapshots(
            normalized,
            entry_minutes=60,
            max_snapshot_age_hours=6,
        )

        self.assertEqual(len(selected), 1)
        row = selected.iloc[0]
        # The entry must be at least 60 minutes before tipoff.
        self.assertEqual(row["entry_total"], 220.5)
        self.assertEqual(row["entry_over_odds"], 1.91)
        self.assertEqual(row["closing_total"], 221.5)
        self.assertEqual(row["closing_over_odds"], 1.88)
        self.assertEqual(row["entry_age_minutes"], 60.0)
        self.assertEqual(diagnostics["post_tipoff_rows_excluded"], 1)
        self.assertEqual(selection["games_without_a_separate_closing_snapshot"], 0)

    def test_does_not_fake_clv_when_only_one_snapshot_exists(self):
        one = snapshot_rows().iloc[[0]].copy()
        normalized, _ = prepare_market_snapshots(one)
        selected, diagnostics = select_entry_and_closing_snapshots(
            normalized,
            entry_minutes=60,
            max_snapshot_age_hours=6,
        )

        self.assertTrue(pd.isna(selected.iloc[0]["closing_total"]))
        self.assertEqual(diagnostics["games_without_a_separate_closing_snapshot"], 1)
        self.assertTrue(np.isnan(side_adjusted_clv("Over", 220.5, selected.iloc[0]["closing_total"])))

    def test_skips_market_prices_that_are_too_old(self):
        old = snapshot_rows().iloc[[0]].copy()
        old.loc[:, "commence_time_utc"] = "2026-01-03T02:30:00Z"
        normalized, _ = prepare_market_snapshots(old)
        selected, diagnostics = select_entry_and_closing_snapshots(
            normalized,
            entry_minutes=60,
            max_snapshot_age_hours=2,
        )

        self.assertTrue(selected.empty)
        self.assertEqual(diagnostics["skipped_stale_entry_snapshot"], 1)

    def test_requires_real_over_and_under_decimal_prices(self):
        missing_under_price = snapshot_rows().drop(columns=["under_odds"])
        with self.assertRaisesRegex(ValueError, "under_odds"):
            prepare_market_snapshots(missing_under_price)

    def test_requires_explicit_bookmaker_if_multiple_books_are_present(self):
        rows = snapshot_rows().iloc[[0]].copy()
        second = rows.copy()
        second.loc[:, "bookmaker"] = "Book B"
        with self.assertRaisesRegex(ValueError, "multiple bookmakers"):
            prepare_market_snapshots(pd.concat([rows, second], ignore_index=True))

    def test_unknown_bookmaker_error_lists_available_books(self):
        rows = snapshot_rows().iloc[[0]].copy()
        with self.assertRaisesRegex(ValueError, "Book A"):
            prepare_market_snapshots(rows, bookmaker="Unknown Book")

    def test_invalid_decimal_prices_are_rejected(self):
        rows = snapshot_rows().iloc[[0]].copy()
        rows.loc[:, "over_odds"] = 1.0
        with self.assertRaisesRegex(ValueError, "invalid rows"):
            prepare_market_snapshots(rows)

    def test_settlement_uses_actual_decimal_price_and_handles_push(self):
        self.assertEqual(settle_total_bet("Over", 230, 220.5, 1.90, 1.0), ("Win", 0.9, 1.0))
        self.assertEqual(settle_total_bet("Under", 230, 220.5, 1.90, 1.0), ("Loss", -1.0, 1.0))
        self.assertEqual(settle_total_bet("Over", 220, 220.0, 1.90, 1.0), ("Push", 0.0, 1.0))
        self.assertEqual(settle_total_bet(None, 230, 220.5, None, 1.0), ("No Bet", 0.0, 0.0))

    def test_clv_is_side_adjusted(self):
        self.assertEqual(side_adjusted_clv("Over", 220.5, 222.5), 2.0)
        self.assertEqual(side_adjusted_clv("Under", 220.5, 218.5), 2.0)
        self.assertTrue(np.isnan(side_adjusted_clv(None, 220.5, 222.5)))

    def test_score_join_uses_date_and_unordered_team_pair(self):
        scores = prepare_score_history(pd.DataFrame([
            {
                "game_date": "2026-01-02",
                "home_team": "Los Angeles Clippers",
                "away_team": "Los Angeles Lakers",
                "home_score": 110,
                "away_score": 108,
            }
        ]))
        markets, _ = prepare_market_snapshots(snapshot_rows().iloc[[0]])
        selected, _ = select_entry_and_closing_snapshots(
            markets, entry_minutes=60, max_snapshot_age_hours=6
        )
        joined, diagnostics = join_market_to_scores(selected, scores)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined.iloc[0]["home_team"], "Los Angeles Clippers")
        self.assertEqual(joined.iloc[0]["away_team"], "Los Angeles Lakers")
        self.assertEqual(joined.iloc[0]["actual_total"], 218)
        self.assertEqual(diagnostics["market_games_without_matching_final_score"], 0)

    def test_prediction_history_is_strictly_before_target_date(self):
        score_rows = []
        market_rows = []
        for day, home, away in [
            ("2026-01-01", "Team A", "Team B"),
            ("2026-01-02", "Team C", "Team D"),
            ("2026-01-03", "Team E", "Team F"),
        ]:
            score_rows.append({
                "game_date": day,
                "home_team": home,
                "away_team": away,
                "home_score": 110,
                "away_score": 108,
            })
            market_rows.append({
                "game_date": pd.Timestamp(day),
                "home_team": home,
                "away_team": away,
                "bookmaker": "Book A",
                "commence_time_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23),
                "entry_total": 220.5,
                "entry_over_odds": 1.91,
                "entry_under_odds": 1.91,
                "closing_total": np.nan,
                "actual_total": 218.0,
                "_game_key": day + home + away,
            })

        history = prepare_score_history(pd.DataFrame(score_rows))
        games = pd.DataFrame(market_rows)
        observed_prior_dates = []

        def fake_predict(**kwargs):
            prior = kwargs["history_df"]
            target = pd.Timestamp(kwargs["game_date"]).normalize()
            if not prior.empty:
                self.assertTrue((prior["game_date"] < target).all())
                observed_prior_dates.append(prior["game_date"].max())
            return {
                "projected_total": 221.0,
                "points_engine_total": 221.0,
                "experimental_adjustments_applied": False,
            }

        with patch("totals_betting_backtest.predict_game_total", side_effect=fake_predict):
            results, holdout_start = generate_predictions(
                games,
                history,
                holdout_games=1,
                min_prior_games=1,
                stake_units=1,
            )

        self.assertEqual(holdout_start, pd.Timestamp("2026-01-03"))
        self.assertEqual(results.loc[results["prediction_status"] == "scored"].shape[0], 2)
        self.assertGreater(len(observed_prior_dates), 0)
        self.assertTrue((results.loc[results["prediction_status"] == "scored", "prior_max_date"] <
                         results.loc[results["prediction_status"] == "scored", "game_date"].map(lambda d: pd.Timestamp(d).date().isoformat())).all())

    def test_summary_reports_roi_and_does_not_include_no_bets_in_stake(self):
        rows = pd.DataFrame([
            {
                "prediction_status": "scored",
                "evaluation_split": "reserved_holdout",
                "actual_total_error": -2.0,
                "market_line_error": 1.0,
                "bet_side": "Over",
                "recommendation": "Strong Over",
                "bet_outcome": "Win",
                "stake_units": 1.0,
                "profit_units": 0.9,
                "selected_decimal_odds": 1.9,
                "clv_points": 1.5,
            },
            {
                "prediction_status": "scored",
                "evaluation_split": "reserved_holdout",
                "actual_total_error": 2.0,
                "market_line_error": -1.0,
                "bet_side": "Under",
                "recommendation": "Strong Under",
                "bet_outcome": "Loss",
                "stake_units": 1.0,
                "profit_units": -1.0,
                "selected_decimal_odds": 2.0,
                "clv_points": -0.5,
            },
            {
                "prediction_status": "scored",
                "evaluation_split": "reserved_holdout",
                "actual_total_error": 1.0,
                "market_line_error": 0.0,
                "bet_side": "",
                "recommendation": "No Bet",
                "bet_outcome": "No Bet",
                "stake_units": 0.0,
                "profit_units": 0.0,
                "selected_decimal_odds": np.nan,
                "clv_points": np.nan,
            },
        ])

        summary = summarize_backtest(rows)
        holdout_all = summary[
            (summary["split"] == "reserved_holdout")
            & (summary["strategy"] == "all_qualifying_edges")
        ].iloc[0]

        self.assertEqual(holdout_all["bets"], 2)
        self.assertEqual(holdout_all["wins"], 1)
        self.assertEqual(holdout_all["losses"], 1)
        self.assertEqual(holdout_all["stake_units_risked"], 2.0)
        self.assertAlmostEqual(holdout_all["net_profit_units"], -0.1)
        self.assertAlmostEqual(holdout_all["roi_pct"], -5.0)
        self.assertEqual(holdout_all["clv_bets"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
