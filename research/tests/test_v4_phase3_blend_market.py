from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.v4_phase3_blend_market import (
    DEFAULT_BETA,
    attach_v4_features,
    summarize_bets,
)


def base_game():
    return {
        "game_date": pd.Timestamp("2026-10-30"),
        "home_team": "Washington Wizards",
        "away_team": "Charlotte Hornets",
        "actual_total": 174.0,
        "points_engine_total": 180.0,
        "entry_total": 175.5,
        "entry_over_odds_decimal": 1.925926,
        "entry_under_odds_decimal": 1.892857,
        "over_probability_no_vig": 0.51,
        "under_probability_no_vig": 0.49,
        "last_observed_total": 176.5,
        "last_observed_over_probability_no_vig": 0.52,
        "last_observed_under_probability_no_vig": 0.48,
    }


def v4_feature(**updates):
    row = {
        "game_date": pd.Timestamp("2026-10-30"),
        "home_team": "Washington Wizards",
        "away_team": "Charlotte Hornets",
        "actual_total": 174.0,
        "projected_total": 172.0,
        "max_history_date": pd.Timestamp("2026-10-29"),
        "home_games_prior": 5,
        "away_games_prior": 6,
        "season_blend_weight": 0.216,
    }
    row.update(updates)
    return row


def test_attach_v4_features_calculates_frozen_beta_blend_on_exact_match():
    out = attach_v4_features(
        pd.DataFrame([base_game()]),
        pd.DataFrame([v4_feature()]),
    )
    assert len(out) == 1
    assert bool(out.loc[0, "v4_prediction_valid"])
    expected = 180.0 + DEFAULT_BETA * (172.0 - 180.0)
    assert out.loc[0, "blend_total"] == pytest.approx(expected)
    assert out.loc[0, "v4_max_history_date"] == pd.Timestamp("2026-10-29")


def test_attach_v4_features_rejects_same_day_history():
    with pytest.raises(AssertionError, match="target-date/future history"):
        attach_v4_features(
            pd.DataFrame([base_game()]),
            pd.DataFrame([v4_feature(max_history_date=pd.Timestamp("2026-10-30"))]),
        )


def test_attach_v4_features_does_not_fill_missing_feature_or_score_mismatch():
    missing = attach_v4_features(
        pd.DataFrame([base_game()]),
        pd.DataFrame([v4_feature(home_team="Boston Celtics")]),
    )
    assert not bool(missing.loc[0, "v4_feature_matched"])
    assert pd.isna(missing.loc[0, "blend_total"])

    mismatch = attach_v4_features(
        pd.DataFrame([base_game()]),
        pd.DataFrame([v4_feature(actual_total=175.0)]),
    )
    assert bool(mismatch.loc[0, "v4_feature_matched"])
    assert not bool(mismatch.loc[0, "v4_score_consistent"])
    assert not bool(mismatch.loc[0, "v4_prediction_valid"])
    assert pd.isna(mismatch.loc[0, "blend_total"])


def test_attach_v4_features_rejects_duplicate_game_keys():
    with pytest.raises(ValueError, match="not unique by date/home/away"):
        attach_v4_features(
            pd.DataFrame([base_game()]),
            pd.DataFrame([v4_feature(), v4_feature()]),
        )


def test_default_beta_is_fixed_and_not_reestimated_on_market_sample():
    assert DEFAULT_BETA == pytest.approx(0.584068)


def test_bet_summary_uses_entry_side_and_later_line_only_as_clv():
    frame = pd.DataFrame([{
        "points_engine_side": "Over",
        "points_engine_settlement": "Win",
        "points_engine_profit_units": 0.925926,
        "entry_total": 175.5,
        "last_observed_total": 176.5,
        "over_probability_no_vig": 0.51,
        "last_observed_over_probability_no_vig": 0.52,
        "under_probability_no_vig": 0.49,
        "last_observed_under_probability_no_vig": 0.48,
        "v4_side": "",
        "v4_settlement": "No Bet",
        "v4_profit_units": 0.0,
        "blend_side": "Under",
        "blend_settlement": "Loss",
        "blend_profit_units": -1.0,
    }])
    summary = summarize_bets(frame).set_index("strategy")
    assert summary.loc["Points Engine V2", "bets"] == 1
    assert summary.loc["Points Engine V2", "net_profit_units"] == pytest.approx(0.925926)
    assert summary.loc["Points Engine V2", "mean_side_adjusted_line_clv_points"] == pytest.approx(1.0)
    assert summary.loc["V2 + V4 blend", "bets"] == 1
    assert summary.loc["V2 + V4 blend", "mean_side_adjusted_line_clv_points"] == pytest.approx(-1.0)
