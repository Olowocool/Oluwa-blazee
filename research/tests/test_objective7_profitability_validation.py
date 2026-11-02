import numpy as np
import pandas as pd
import pytest

from research.objective7_profitability_validation import (
    _validate_inputs,
    build_summary,
    settle_decisions,
)


def settled_fixture():
    return pd.DataFrame({
        "game_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
        "commence_time_utc": pd.to_datetime([
            "2026-01-01T23:00:00Z", "2026-01-01T23:30:00Z",
            "2026-01-02T23:00:00Z", "2026-01-02T23:30:00Z",
        ], utc=True),
        "decision": ["Over", "Under", "Over", "No Bet"],
        "stake_units": [1.0, 1.0, 0.5, 0.0],
        "entry_total": [220.5, 220.5, 220.0, 220.5],
        "actual_total": [221.0, 221.0, 220.0, 219.0],
        "entry_over_odds_decimal": [1.91, 1.91, 1.91, 1.91],
        "entry_under_odds_decimal": [1.91, 1.91, 1.91, 1.91],
        "over_probability_no_vig": [0.51, 0.51, 0.51, 0.51],
        "under_probability_no_vig": [0.49, 0.49, 0.49, 0.49],
        "last_observed_total": [220.0, 221.0, 219.5, 220.0],
    })


def test_settlement_correctly_handles_win_loss_push_and_no_bet():
    result = settle_decisions(settled_fixture())
    assert result["settlement"].tolist() == ["Win", "Loss", "Push", "No Bet"]
    assert result["net_profit_units"].tolist() == pytest.approx([0.91, -1.0, 0.0, 0.0])
    assert result["risked_units"].tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0])


def test_summary_reports_realized_roi_and_keeps_sample_gate_separate():
    result = build_summary(settle_decisions(settled_fixture()))
    assert result["settled_bets"] == 3
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["pushes"] == 1
    assert result["stake_risked_units"] == pytest.approx(2.5)
    assert result["net_profit_units"] == pytest.approx(-0.09)
    assert result["realized_roi"] == pytest.approx(-0.036)
    assert result["status"] == "insufficient_profitability_sample"
    assert result["interpretation"]["holdout_outcomes_used_to_select_or_change_bets"] is False


def test_no_bet_rows_cannot_have_stakes():
    frame = settled_fixture()
    frame.loc[3, "stake_units"] = 0.25
    with pytest.raises(ValueError, match="No Bet rows cannot carry stake"):
        settle_decisions(frame)


def test_active_bets_require_positive_stakes():
    frame = settled_fixture()
    frame.loc[0, "stake_units"] = 0.0
    with pytest.raises(ValueError, match="positive stake"):
        settle_decisions(frame)


def test_join_rejects_duplicate_game_keys():
    games = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["Home"], "away_team": ["Away"],
        "entry_total": [220.5], "actual_total": [221.0],
        "last_observed_total": [220.0],
        "entry_over_odds_decimal": [1.91], "entry_under_odds_decimal": [1.91],
        "over_probability_no_vig": [0.51], "under_probability_no_vig": [0.49],
    })
    decisions = pd.DataFrame({
        "game_date": ["2026-01-01", "2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"] * 2,
        "home_team": ["Home", "Home"], "away_team": ["Away", "Away"],
        "entry_total": [220.5, 220.5], "entry_over_odds_decimal": [1.91, 1.91],
        "entry_under_odds_decimal": [1.91, 1.91], "decision": ["Over", "Over"],
        "stake_units": [1.0, 1.0], "evaluation_split": ["reserved_holdout"] * 2,
    })
    with pytest.raises(ValueError, match="Duplicate game keys"):
        _validate_inputs(games, decisions, {"development_gate": {"passed": True}})


def test_failed_objective6_gate_cannot_be_used_to_evaluate_active_bets():
    games = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["Home"], "away_team": ["Away"],
        "entry_total": [220.5], "actual_total": [221.0],
        "last_observed_total": [220.0],
        "entry_over_odds_decimal": [1.91], "entry_under_odds_decimal": [1.91],
        "over_probability_no_vig": [0.51], "under_probability_no_vig": [0.49],
    })
    decisions = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["Home"], "away_team": ["Away"],
        "entry_total": [220.5], "entry_over_odds_decimal": [1.91],
        "entry_under_odds_decimal": [1.91], "decision": ["Over"],
        "stake_units": [1.0], "evaluation_split": ["reserved_holdout"],
    })
    with pytest.raises(ValueError, match="gate failed"):
        _validate_inputs(games, decisions, {"development_gate": {"passed": False}})



def test_empty_reserved_holdout_is_reported_as_insufficient_not_crashed():
    games = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["home"], "away_team": ["away"],
        "entry_total": [220.5], "actual_total": [221.0],
        "last_observed_total": [220.0],
        "entry_over_odds_decimal": [1.91], "entry_under_odds_decimal": [1.91],
        "over_probability_no_vig": [0.51], "under_probability_no_vig": [0.49],
    })
    decisions = pd.DataFrame(columns=[
        "game_date", "commence_time_utc", "home_team", "away_team", "entry_total",
        "entry_over_odds_decimal", "entry_under_odds_decimal", "decision",
        "stake_units", "evaluation_split",
    ])
    merged = _validate_inputs(
        games, decisions, {"development_gate": {"passed": False}}
    )
    result = build_summary(settle_decisions(merged))
    assert result["settled_bets"] == 0
    assert result["realized_roi"] is None
    assert result["status"] == "insufficient_profitability_sample"


def test_objective6_entry_prices_must_match_objective5_source_prices():
    games = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["Home"], "away_team": ["Away"],
        "entry_total": [220.5], "actual_total": [221.0],
        "last_observed_total": [220.0],
        "entry_over_odds_decimal": [1.91], "entry_under_odds_decimal": [1.91],
    })
    decisions = pd.DataFrame({
        "game_date": ["2026-01-01"],
        "commence_time_utc": ["2026-01-01T23:00:00Z"],
        "home_team": ["Home"], "away_team": ["Away"],
        "entry_total": [220.5], "entry_over_odds_decimal": [1.80],
        "entry_under_odds_decimal": [1.91], "decision": ["No Bet"],
        "stake_units": [0.0], "evaluation_split": ["reserved_holdout"],
    })
    with pytest.raises(ValueError, match="entry prices do not match"):
        _validate_inputs(games, decisions, {"development_gate": {"passed": False}})
