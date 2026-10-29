import argparse
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import dunkapi_totals_close_backtest as bt


def daily_payload():
    return {
        "count": 2,
        "games": [
            {
                "game_id": "game-1",
                "date": "2025-11-21",
                "starts_at": "2025-11-22T01:00:00.000Z",
                "status": "final",
                "home": {"name": "Los Angeles Clippers", "score": 120},
                "away": {"name": "Los Angeles Lakers", "score": 110},
                "books": [
                    {
                        "provider": "DraftKings",
                        "total_open": 224.5,
                        "total_close": 221.5,
                        "over_odds": -110,
                        "under_odds": -110,
                        "captured_at": "2025-11-22T00:50:00.000Z",
                    },
                    {
                        "provider": "BetMGM",
                        "total_open": 225.5,
                        "total_close": 222.5,
                        "over_odds": -105,
                        "under_odds": -115,
                        "captured_at": "2025-11-22T00:51:00.000Z",
                    },
                ],
            },
            {
                "game_id": "game-upcoming",
                "date": "2025-11-21",
                "starts_at": "2025-11-22T02:00:00.000Z",
                "status": "scheduled",
                "home": {"name": "Boston Celtics", "score": None},
                "away": {"name": "New York Knicks", "score": None},
                "books": [
                    {
                        "provider": "DraftKings",
                        "total_open": 220.5,
                        "total_close": 219.5,
                        "over_odds": -110,
                        "under_odds": -110,
                    },
                ],
            },
            {
                "game_id": "game-other-date",
                "date": "2025-11-22",
                "starts_at": "2025-11-23T01:00:00.000Z",
                "status": "final",
                "home": {"name": "Team A", "score": 100},
                "away": {"name": "Team B", "score": 99},
                "books": [
                    {
                        "provider": "DraftKings",
                        "total_open": 200.5,
                        "total_close": 201.5,
                        "over_odds": -110,
                        "under_odds": -110,
                    },
                ],
            },
        ],
    }


class DunkAPITotalsCloseBacktestTests(unittest.TestCase):
    def test_date_range_and_daily_credit_estimate(self):
        self.assertEqual(
            bt.parse_date_range("2025-11-21", "2025-11-25"),
            [date(2025, 11, d) for d in range(21, 26)],
        )
        self.assertEqual(bt.estimate_api_credits(5), 50)
        with self.assertRaisesRegex(ValueError, "on or after"):
            bt.parse_date_range("2025-11-26", "2025-11-25")

    def test_american_odds_convert_to_decimal(self):
        self.assertAlmostEqual(bt.american_to_decimal(-110), 1.909091, places=5)
        self.assertAlmostEqual(bt.american_to_decimal(125), 2.25, places=5)
        with self.assertRaises(ValueError):
            bt.american_to_decimal(0)

    def test_parser_selects_one_book_and_uses_real_closing_price(self):
        rows, diag = bt.parse_daily_payload(
            daily_payload(), date(2025, 11, 21), "draftkings"
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["game_id"], "game-1")
        self.assertEqual(row["game_date"], "2025-11-21")
        self.assertEqual(row["home_team"], "Los Angeles Clippers")
        self.assertEqual(row["away_team"], "Los Angeles Lakers")
        self.assertEqual(row["total_open"], 224.5)
        self.assertEqual(row["total_close"], 221.5)
        self.assertEqual(row["over_odds_american"], -110)
        self.assertEqual(row["under_odds_american"], -110)
        self.assertAlmostEqual(row["over_odds_decimal"], 1.909091, places=5)
        self.assertEqual(row["actual_total"], 230)
        self.assertEqual(row["market_source"], "DunkAPI get_odds historical opening/closing archive")
        self.assertEqual(diag["games_seen"], 3)
        self.assertEqual(diag["rows_for_selected_bookmaker"], 1)
        self.assertEqual(diag["rows_for_other_dates_excluded"], 1)
        self.assertEqual(diag["available_bookmakers"], ["BetMGM", "DraftKings"])

    def test_parser_accepts_documented_data_wrapped_response(self):
        payload = {"data": daily_payload()}
        rows, diag = bt.parse_daily_payload(
            payload, date(2025, 11, 21), "DraftKings"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game_id"], "game-1")
        self.assertEqual(rows[0]["total_close"], 221.5)
        self.assertEqual(diag["rows_for_selected_bookmaker"], 1)

    def test_parser_reports_keys_when_games_array_is_missing(self):
        with self.assertRaisesRegex(ValueError, "at the root or under 'data'"):
            bt.parse_daily_payload(
                {"error": "response shape unavailable"}, date(2025, 11, 21), "DraftKings"
            )

    def test_parser_does_not_use_upcoming_or_missing_price_rows(self):
        payload = daily_payload()
        payload["games"][0]["books"][0]["over_odds"] = None
        rows, diag = bt.parse_daily_payload(payload, date(2025, 11, 21), "DraftKings")
        self.assertEqual(rows, [])
        self.assertEqual(diag["games_missing_close_line_or_prices"], 1)

    def test_parser_allows_missing_open_total_without_fabricating_it(self):
        payload = daily_payload()
        payload["games"][0]["books"][0]["total_open"] = None
        rows, _ = bt.parse_daily_payload(payload, date(2025, 11, 21), "DraftKings")
        self.assertEqual(len(rows), 1)
        self.assertTrue(pd.isna(rows[0]["total_open"]))
        self.assertEqual(rows[0]["total_close"], 221.5)

    def test_archive_merge_can_enrich_missing_open_total(self):
        rows, _ = bt.parse_daily_payload(daily_payload(), date(2025, 11, 21), "DraftKings")
        with_missing_open = dict(rows[0])
        with_missing_open["total_open"] = float("nan")
        self.assertTrue(pd.isna(bt.merge_market_rows([], [with_missing_open])[0]["total_open"]))
        merged = bt.merge_market_rows([with_missing_open], rows)
        self.assertEqual(merged[0]["total_open"], 224.5)

    def test_merge_is_idempotent_and_conflicts_fail(self):
        rows, _ = bt.parse_daily_payload(daily_payload(), date(2025, 11, 21), "DraftKings")
        self.assertEqual(len(bt.merge_market_rows(rows, rows)), 1)
        conflict = dict(rows[0])
        conflict["total_close"] = 220.5
        with self.assertRaisesRegex(ValueError, "Conflicting historical market data"):
            bt.merge_market_rows(rows, [conflict])

    def test_http_400_shows_provider_message_without_exposing_headers(self):
        class FakeResponse:
            status_code = 400
            headers = {}
            text = '{"detail":"No archived games for this date"}'

            def json(self):
                return {"detail": "No archived games for this date"}

        with patch.object(bt.requests, "post", return_value=FakeResponse()):
            with self.assertRaisesRegex(
                RuntimeError,
                r"2025-11-27 \(HTTP 400\).*No archived games for this date",
            ):
                bt.fetch_daily_odds("do-not-log-this-key", date(2025, 11, 27))

    def test_plan_only_never_calls_provider(self):
        args = argparse.Namespace(
            start_date="2025-11-21",
            end_date="2025-11-25",
            bookmaker="DraftKings",
            output_file="unused.csv",
            plan_only=True,
            confirm_api_cost=False,
            max_api_credits=None,
        )
        with patch.object(bt, "fetch_daily_odds", side_effect=AssertionError("network called")):
            result = bt.collect_market_data(args)
        self.assertEqual(result["status"], "plan_only")
        self.assertEqual(result["daily_requests"], 5)
        self.assertEqual(result["estimated_credits"], 50)

    def test_budget_guard_stops_before_network(self):
        base = dict(
            start_date="2025-11-21",
            end_date="2025-11-25",
            bookmaker="DraftKings",
            output_file="unused.csv",
            plan_only=False,
            confirm_api_cost=False,
            max_api_credits=50,
        )
        with patch.object(bt, "fetch_daily_odds", side_effect=AssertionError("network called")):
            with self.assertRaisesRegex(ValueError, "No API calls made"):
                bt.collect_market_data(argparse.Namespace(**base))
        base["confirm_api_cost"] = True
        base["max_api_credits"] = 49
        with patch.object(bt, "fetch_daily_odds", side_effect=AssertionError("network called")):
            with self.assertRaisesRegex(ValueError, "exceeds --max-api-credits"):
                bt.collect_market_data(argparse.Namespace(**base))

    def test_close_line_backtest_uses_strict_prior_history_and_settles_odds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            score_rows = []
            # Build 101 completed prior games, all before the target period.
            start = pd.Timestamp("2025-07-01")
            for i in range(101):
                score_rows.append({
                    "game_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                    "home_team": f"History Home {i}",
                    "away_team": f"History Away {i}",
                    "home_score": 110,
                    "away_score": 108,
                })

            target_games = [
                {
                    "game_id": "target-1", "game_date": "2025-11-21",
                    "home_team": "Los Angeles Clippers", "away_team": "Los Angeles Lakers",
                    "home_score": 120, "away_score": 110, "actual_total": 230,
                    "total_open": 224.5, "total_close": 221.5,
                    "over_odds_american": -110, "under_odds_american": -110,
                    "over_odds_decimal": bt.american_to_decimal(-110),
                    "under_odds_decimal": bt.american_to_decimal(-110),
                    "bookmaker": "DraftKings", "game_id": "target-1",
                    "commence_time_utc": "2025-11-22T01:00:00Z",
                    "actual_home_points": 120, "actual_away_points": 110,
                    "source_captured_at": "", "market_source": "fixture",
                },
                {
                    "game_id": "target-2", "game_date": "2025-11-22",
                    "home_team": "Boston Celtics", "away_team": "New York Knicks",
                    "home_score": 100, "away_score": 105, "actual_total": 205,
                    "total_open": 218.5, "total_close": 220.5,
                    "over_odds_american": -110, "under_odds_american": -110,
                    "over_odds_decimal": bt.american_to_decimal(-110),
                    "under_odds_decimal": bt.american_to_decimal(-110),
                    "bookmaker": "DraftKings", "commence_time_utc": "2025-11-23T00:00:00Z",
                    "actual_home_points": 100, "actual_away_points": 105,
                    "source_captured_at": "", "market_source": "fixture",
                },
            ]
            for game in target_games:
                score_rows.append({
                    "game_date": game["game_date"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_score": game["actual_home_points"],
                    "away_score": game["actual_away_points"],
                })

            scores_path = tmp_path / "scores.csv"
            market_path = tmp_path / "market.csv"
            pd.DataFrame(score_rows).to_csv(scores_path, index=False)
            pd.DataFrame(target_games).to_csv(market_path, index=False)

            observed = []
            def fake_predict(**kwargs):
                prior = kwargs["history_df"]
                target = pd.Timestamp(kwargs["game_date"]).normalize()
                self.assertTrue((prior["game_date"] < target).all())
                observed.append((target, len(prior)))
                return {
                    "projected_total": 230.5,
                    "points_engine_total": 230.5,
                    "experimental_adjustments_applied": False,
                }

            args = argparse.Namespace(
                market_file=str(market_path),
                scores_file=str(scores_path),
                output_dir=str(tmp_path / "outputs"),
                holdout_games=1,
                min_prior_games=100,
                stake_units=1.0,
            )
            with patch.object(bt, "predict_game_total", side_effect=fake_predict):
                results, summary, diagnostics = bt.run_closing_line_backtest(args)

            scored = results.loc[results["prediction_status"] == "scored"]
            self.assertEqual(len(scored), 2)
            self.assertEqual(len(observed), 2)
            self.assertEqual(scored.iloc[0]["bet_outcome"], "Win")
            self.assertEqual(scored.iloc[0]["market_line_move_points"], -3.0)
            self.assertFalse(diagnostics["entry_time_clv_available"])
            self.assertIn("roi_pct", summary.columns)
            holdout = summary[
                (summary["split"] == "reserved_holdout")
                & (summary["strategy"] == "all_qualifying_edges")
            ].iloc[0]
            self.assertEqual(holdout["bets"], 1)
            self.assertEqual(holdout["losses"], 1)
            self.assertAlmostEqual(holdout["roi_pct"], -100.0)



    def test_backtest_applies_requested_date_and_bookmaker_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            score_rows = []
            start = pd.Timestamp("2025-06-01")
            for i in range(101):
                score_rows.append({
                    "game_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                    "home_team": f"Prior Home {i}",
                    "away_team": f"Prior Away {i}",
                    "home_score": 110,
                    "away_score": 108,
                })

            games = [
                ("g1", "2025-11-21", "LA Clippers", "LA Lakers", "DraftKings", 120, 110),
                ("g2", "2025-11-22", "Boston Celtics", "New York Knicks", "DraftKings", 100, 105),
                ("g3", "2025-11-22", "Chicago Bulls", "Miami Heat", "BetMGM", 100, 101),
                ("g4", "2025-11-25", "Atlanta Hawks", "Brooklyn Nets", "DraftKings", 108, 109),
            ]
            market_rows = []
            for game_id, game_date, home, away, book, home_score, away_score in games:
                score_rows.append({
                    "game_date": game_date, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                })
                market_rows.append({
                    "game_id": game_id, "game_date": game_date, "home_team": home,
                    "away_team": away, "bookmaker": book, "total_open": 220.5,
                    "total_close": 221.5, "over_odds_decimal": 1.91,
                    "under_odds_decimal": 1.91, "actual_home_points": home_score,
                    "actual_away_points": away_score, "actual_total": home_score + away_score,
                })

            scores_path = tmp_path / "scores.csv"
            market_path = tmp_path / "market.csv"
            pd.DataFrame(score_rows).to_csv(scores_path, index=False)
            pd.DataFrame(market_rows).to_csv(market_path, index=False)

            args = argparse.Namespace(
                market_file=str(market_path),
                scores_file=str(scores_path),
                output_dir=str(tmp_path / "outputs"),
                holdout_games=1,
                min_prior_games=100,
                stake_units=1.0,
                start_date="2025-11-21",
                end_date="2025-11-22",
                bookmaker="DraftKings",
            )

            def fake_predict(**kwargs):
                prior = kwargs["history_df"]
                target = pd.Timestamp(kwargs["game_date"]).normalize()
                self.assertTrue((prior["game_date"] < target).all())
                return {
                    "projected_total": 230.5,
                    "points_engine_total": 230.5,
                    "experimental_adjustments_applied": False,
                }

            with patch.object(bt, "predict_game_total", side_effect=fake_predict):
                results, summary, diagnostics = bt.run_closing_line_backtest(args)

            self.assertEqual(diagnostics["market_rows"], 2)
            self.assertEqual(diagnostics["games_scored"], 2)
            self.assertEqual(set(results["game_id"]), {"g1", "g2"})
            self.assertEqual(set(results["bookmaker"]), {"DraftKings"})
            self.assertTrue(results["game_date"].astype(str).str[:10].isin(
                ["2025-11-21", "2025-11-22"]
            ).all())

    def test_verified_score_overrides_separate_settlement_from_feature_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            score_rows = []
            start = pd.Timestamp("2025-06-01")
            for i in range(101):
                score_rows.append({
                    "game_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                    "home_team": f"Prior Home {i}",
                    "away_team": f"Prior Away {i}",
                    "home_score": 110,
                    "away_score": 108,
                })

            # The third target is present in normal score history. The first two
            # are validated by the override file only.
            score_rows.append({
                "game_date": "2025-11-23", "home_team": "Chicago Bulls",
                "away_team": "Miami Heat", "home_score": 105, "away_score": 106,
            })

            games = [
                ("override-regular", "2025-11-21", "LA Clippers", "LA Lakers", 120, 110),
                ("override-title", "2025-11-22", "Boston Celtics", "New York Knicks", 100, 110),
                ("historical", "2025-11-23", "Chicago Bulls", "Miami Heat", 105, 106),
            ]
            market_rows = []
            override_rows = []
            for game_id, game_date, home, away, home_score, away_score in games:
                market_rows.append({
                    "game_id": game_id, "game_date": game_date, "home_team": home,
                    "away_team": away, "bookmaker": "DraftKings", "total_open": 220.5,
                    "total_close": 221.5, "over_odds_decimal": 1.91,
                    "under_odds_decimal": 1.91, "actual_home_points": home_score,
                    "actual_away_points": away_score, "actual_total": home_score + away_score,
                })
            override_rows = [
                {
                    "game_date": "2025-11-21", "home_team": "LA Clippers",
                    "away_team": "LA Lakers", "home_score": 120, "away_score": 110,
                    "include_in_history": True,
                },
                {
                    "game_date": "2025-11-22", "home_team": "Boston Celtics",
                    "away_team": "New York Knicks", "home_score": 100, "away_score": 110,
                    "include_in_history": False,
                },
            ]

            scores_path = tmp_path / "scores.csv"
            market_path = tmp_path / "market.csv"
            overrides_path = tmp_path / "overrides.csv"
            pd.DataFrame(score_rows).to_csv(scores_path, index=False)
            pd.DataFrame(market_rows).to_csv(market_path, index=False)
            pd.DataFrame(override_rows).to_csv(overrides_path, index=False)

            args = argparse.Namespace(
                market_file=str(market_path),
                scores_file=str(scores_path),
                score_overrides_file=str(overrides_path),
                output_dir=str(tmp_path / "outputs"),
                holdout_games=1,
                min_prior_games=100,
                stake_units=1.0,
                start_date="2025-11-21",
                end_date="2025-11-23",
                bookmaker="DraftKings",
            )
            observed_prior = {}

            def fake_predict(**kwargs):
                prior = kwargs["history_df"]
                target = pd.Timestamp(kwargs["game_date"]).normalize()
                self.assertTrue((prior["game_date"] < target).all())
                observed_prior[target.date().isoformat()] = set(prior["_game_key"])
                return {
                    "projected_total": 230.5,
                    "points_engine_total": 230.5,
                    "experimental_adjustments_applied": False,
                }

            with patch.object(bt, "predict_game_total", side_effect=fake_predict):
                results, summary, diagnostics = bt.run_closing_line_backtest(args)

            first_key = bt.game_key("2025-11-21", "LA Clippers", "LA Lakers")
            second_key = bt.game_key("2025-11-22", "Boston Celtics", "New York Knicks")
            self.assertEqual(diagnostics["verified_score_override_rows_used"], 2)
            self.assertEqual(diagnostics["feature_history_override_rows_included"], 1)
            self.assertEqual(diagnostics["games_scored"], 3)
            self.assertEqual(
                results.sort_values("game_date")["score_validation_source"].tolist(),
                ["verified_score_override", "verified_score_override", "historical_score_history"],
            )
            self.assertNotIn(first_key, observed_prior["2025-11-21"])
            self.assertIn(first_key, observed_prior["2025-11-22"])
            self.assertIn(first_key, observed_prior["2025-11-23"])
            self.assertNotIn(second_key, observed_prior["2025-11-23"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
