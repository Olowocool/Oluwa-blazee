from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.evaluate_v3_market_archive import evaluate_market_archive


def test_market_archive_join_uses_canonical_team_names_and_reports_settlements():
    predictions = pd.DataFrame([
        {
            "game_date": "2025-11-01", "home_team": "LA Clippers",
            "away_team": "Boston Celtics", "projected_total": 230.0,
            "v3_2_projected_total": 228.0, "actual_total": 234.0,
        },
        {
            "game_date": "2025-11-02", "home_team": "Charlotte Hornets",
            "away_team": "New York Knicks", "projected_total": 220.0,
            "v3_2_projected_total": 222.0, "actual_total": 228.0,
        },
        {
            "game_date": "2025-11-03", "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers", "projected_total": 235.0,
            "v3_2_projected_total": 234.0, "actual_total": 240.0,
        },
        {
            "game_date": "2025-11-04", "home_team": "Miami Heat",
            "away_team": "Milwaukee Bucks", "projected_total": 235.0,
            "v3_2_projected_total": 233.0, "actual_total": 231.0,
        },
        {
            "game_date": "2025-11-05", "home_team": "Denver Nuggets",
            "away_team": "Utah Jazz", "projected_total": 220.0,
            "v3_2_projected_total": 221.0, "actual_total": 229.0,
        },
    ])
    market = pd.DataFrame([
        {
            "game_date": "2025-11-01", "home_team": "Los Angeles Clippers",
            "away_team": "Boston Celtics", "bookmaker": "DraftKings",
            "total_open": 224.5, "total_close": 225.5,
            "over_odds_decimal": 1.9, "under_odds_decimal": 1.9,
            "actual_total": 234.0,
            "source_captured_at": "2025-11-01T23:50:00Z",
            "commence_time_utc": "2025-11-02T00:00:00Z",
        },
        {
            "game_date": "2025-11-02", "home_team": "Charlotte Hornets",
            "away_team": "New York Knicks", "bookmaker": "DraftKings",
            "total_open": 225.5, "total_close": 224.5,
            "over_odds_decimal": 1.9, "under_odds_decimal": 1.9,
            "actual_total": 228.0,
            "source_captured_at": "2025-11-02T23:50:00Z",
            "commence_time_utc": "2025-11-03T00:00:00Z",
        },
        {
            "game_date": "2025-11-03", "home_team": "Boston Celtics",
            "away_team": "LA Lakers", "bookmaker": "DraftKings",
            "total_open": 230.5, "total_close": 230.5,
            "over_odds_decimal": 1.8, "under_odds_decimal": 2.0,
            "actual_total": 240.0,
            "source_captured_at": "2025-11-03T23:50:00Z",
            "commence_time_utc": "2025-11-04T00:00:00Z",
        },
        {
            "game_date": "2025-11-04", "home_team": "Miami Heat",
            "away_team": "Milwaukee Bucks", "bookmaker": "DraftKings",
            "total_open": 230.5, "total_close": 231.0,
            "over_odds_decimal": 1.9, "under_odds_decimal": 1.9,
            "actual_total": 231.0,
            "source_captured_at": "2025-11-04T23:50:00Z",
            "commence_time_utc": "2025-11-05T00:00:00Z",
        },
        {
            "game_date": "2025-11-05", "home_team": "Denver Nuggets",
            "away_team": "Utah Jazz", "bookmaker": "DraftKings",
            "total_open": 220.5, "total_close": 221.5,
            "over_odds_decimal": 1.9, "under_odds_decimal": 1.9,
            "actual_total": 230.0,
            "source_captured_at": "2025-11-05T23:50:00Z",
            "commence_time_utc": "2025-11-06T00:00:00Z",
        },
    ])

    summary, matched = evaluate_market_archive(predictions, market)

    assert len(matched) == 4
    assert summary["score_mismatch_rows_excluded"] == 1
    assert summary["sample_start"] == "2025-11-01"
    assert summary["sample_end"] == "2025-11-04"
    assert summary["market_capture_lead_minutes_median"] == 10.0

    threshold_25 = summary["threshold_simulations"][0]
    assert threshold_25["bets"] == 4
    assert threshold_25["wins"] == 2
    assert threshold_25["losses"] == 1
    assert threshold_25["pushes"] == 1
    assert threshold_25["net_profit_units"] == 0.7
    assert threshold_25["roi_pct_on_all_stakes_including_pushes"] == 17.5
