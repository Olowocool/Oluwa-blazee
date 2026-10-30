from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.v4_phase3_market_residual import (
    american_to_decimal,
    no_vig,
    paired_bootstrap_mae_difference,
    prepare_snapshots,
    select_entry_and_last,
    summarize,
    walk_forward,
)


def market_row(key, observed, start, total):
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return {
        "game_key": key,
        "game_date": start_dt.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        "commence_time_utc": start,
        "away_team": "Charlotte Hornets",
        "home_team": "Washington Wizards",
        "bookmaker": "DraftKings",
        "home_ml_american": -150,
        "away_ml_american": 130,
        "home_spread": -3.5,
        "home_spread_odds_american": -110,
        "total_line": total,
        "over_odds_american": -108,
        "under_odds_american": -112,
        "published_as_of_utc": observed,
        "observed_at_utc": observed,
    }


def test_american_price_conversion_and_no_vig():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    over, under, vig = no_vig(-110, -110)
    assert over == pytest.approx(0.5)
    assert under == pytest.approx(0.5)
    assert vig > 0


def test_fixed_entry_uses_only_pre_cutoff_and_retains_later_clv_observation():
    start = "2026-11-01T00:00:00Z"
    rows = [
        market_row("g1", "2026-10-31T19:00:00Z", start, 221.5),
        market_row("g1", "2026-10-31T21:30:00Z", start, 223.5),
        market_row("g1", "2026-10-31T23:00:00Z", start, 225.5),
        market_row("g1", "2026-11-01T00:05:00Z", start, 230.5),
    ]
    clean, _ = prepare_snapshots(pd.DataFrame(rows))
    chosen, diag = select_entry_and_last(clean, entry_minutes=120, max_age_minutes=240)
    assert len(chosen) == 1
    row = chosen.iloc[0]
    assert row.entry_total == 223.5
    assert row.first_observed_total == 221.5
    assert row.move_from_first_observed == 2.0
    assert row.last_observed_total == 225.5
    assert bool(row.has_later_pregame_observation)
    assert diag["games_with_entry_observation"] == 1


def test_stale_page_observation_is_rejected():
    start = "2026-11-01T00:00:00Z"
    row = market_row("g1", "2026-10-31T19:00:00Z", start, 221.5)
    row["observed_at_utc"] = "2026-11-01T00:00:00Z"
    with pytest.raises(ValueError, match="more than 180 minutes old"):
        prepare_snapshots(pd.DataFrame([row]))


def test_walk_forward_excludes_same_day_and_freezes_holdout():
    rows = []
    for day in range(1, 5):
        for game in range(2):
            entry = 220.0 + day + game
            rows.append({
                "game_date": pd.Timestamp(f"2026-11-{day:02d}"),
                "commence_time_utc": pd.Timestamp(f"2026-11-{day:02d}T20:00:00Z"),
                "baseline_edge": float(game),
                "entry_total": entry,
                "home_spread": -2.5,
                "move_from_first_observed": float(day - 1),
                "over_probability_no_vig": 0.5,
                "market_overround": 0.045,
                "entry_over_odds_decimal": 1.93,
                "entry_under_odds_decimal": 1.93,
                "home_spread_price_implied_probability": 0.5238,
                "market_residual_target": float(day + game - 2),
            })
    out = walk_forward(pd.DataFrame(rows), min_train_games=2, alpha=25.0)
    day1 = out.loc[pd.to_datetime(out.game_date).dt.day == 1]
    day2 = out.loc[pd.to_datetime(out.game_date).dt.day == 2]
    assert set(day1.residual_model_status) == {"warmup"}
    assert set(day2.residual_model_status) == {"walk_forward_scored"}
    assert set(day2.residual_training_games) == {2}
    holdout = out.loc[pd.to_datetime(out.game_date).dt.day == 4]
    assert set(holdout.residual_model_status) == {"frozen_holdout_scored"}
    assert set(holdout.residual_training_games) == {6}


def test_holdout_mae_bootstrap_is_paired_and_summary_uses_same_games_for_bets():
    records = []
    for i, status in enumerate([
        "walk_forward_scored", "walk_forward_scored", "frozen_holdout_scored", "warmup"
    ]):
        records.append({
            "game_date": pd.Timestamp(f"2026-11-0{i+1}"),
            "evaluation_split": "reserved_holdout" if i >= 2 else "development",
            "residual_model_status": status,
            "entry_total": 220.0,
            "points_engine_total": 223.0,
            "residual_model_total": 222.0 if status != "warmup" else np.nan,
            "actual_total": 224.0 + i,
            "points_engine_side": "Over",
            "points_engine_settlement": "Loss",
            "points_engine_profit_units": -1.0,
            "market_residual_model_side": "Over" if status != "warmup" else "",
            "market_residual_model_settlement": "Loss" if status != "warmup" else "No Bet",
            "market_residual_model_profit_units": -1.0 if status != "warmup" else 0.0,
            "last_observed_total": 221.0,
            "last_observed_over_probability_no_vig": 0.52,
            "last_observed_under_probability_no_vig": 0.48,
            "over_probability_no_vig": 0.50,
            "under_probability_no_vig": 0.50,
        })
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    bets = summary.loc[
        (summary.split == "reserved_holdout")
        & summary.predictor.isin(["points_engine_betting", "market_residual_model_betting"])
    ]
    assert set(bets.games_scored) == {1}
    boot_frame = pd.DataFrame({
        "evaluation_split": ["reserved_holdout"] * 4,
        "entry_total": [220.0] * 4,
        "residual_model_total": [223.0] * 4,
        "actual_total": [224.0] * 4,
    })
    boot = paired_bootstrap_mae_difference(boot_frame, n_resamples=500, seed=42)
    assert boot["observed_mae_difference_model_minus_market"] == pytest.approx(-3.0)
    assert boot["bootstrap_95_ci"] == pytest.approx([-3.0, -3.0])
