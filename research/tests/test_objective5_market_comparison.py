import numpy as np
import pandas as pd
import pytest

from research.objective5_market_comparison import (
    empirical_total_probabilities,
    probability_metrics,
    walk_forward_market_probabilities,
    build_summary,
)
from research.objective3_v5_market_residual import FEATURES


def test_empirical_probabilities_sum_to_one_and_half_point_has_no_push():
    over, under, push = empirical_total_probabilities(220.0, 220.5, [-10, -5, 0, 5, 10])
    assert over + under + push == pytest.approx(1.0)
    assert push == 0.0
    assert over == pytest.approx(4 / 7)
    assert under == pytest.approx(3 / 7)


def test_integer_line_allocates_push_probability():
    over, under, push = empirical_total_probabilities(220.0, 220.0, [-2, -0.25, 0.0, 0.25, 2])
    assert over + under + push == pytest.approx(1.0)
    assert push > 0


def synthetic_games(days=40, games_per_day=10):
    rows = []
    for day in range(days):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        for game in range(games_per_day):
            idx = day * games_per_day + game
            entry = 218.5 + (idx % 9) * 0.5
            edge = ((idx % 13) - 6) * 0.7
            spread = ((idx % 7) - 3) * 1.5
            move = ((idx % 5) - 2) * 0.5
            residual = 0.7 * edge - 0.15 * spread + 0.2 * move + ((idx % 11) - 5) * 0.8
            actual = entry + residual
            row = {
                "game_date": date,
                "commence_time_utc": date.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=game),
                "baseline_edge": edge,
                "entry_total": entry,
                "home_spread": spread,
                "move_from_first_observed": move,
                "over_probability_no_vig": 0.51 + (idx % 3) * 0.01,
                "market_overround": 0.04,
                "entry_over_odds_decimal": 1.91,
                "entry_under_odds_decimal": 1.91,
                "home_spread_price_implied_probability": 0.52,
                "market_residual_target": residual,
                "actual_total": actual,
                "v5_total": entry + edge,
                "last_observed_total": entry + move,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_calibration_uses_only_prior_game_dates_and_frozen_holdout():
    games = synthetic_games()
    result = walk_forward_market_probabilities(
        games, min_train_games=150, min_calibration_games=100, alpha=25.0
    )
    day_16 = result.loc[result.game_date.eq(pd.Timestamp("2026-01-17"))]
    assert len(day_16) == 10
    assert set(day_16.residual_training_games) == {150}
    assert set(day_16.residual_model_status) == {"walk_forward_scored"}

    holdout = result.loc[result.evaluation_split.eq("reserved_holdout")]
    assert len(holdout) == 80
    assert set(holdout.residual_model_status) == {"frozen_holdout_scored"}
    assert holdout.residual_model_over_probability.notna().all()
    assert holdout.v5_over_probability.notna().all()
    assert np.allclose(
        holdout.residual_model_over_probability
        + holdout.residual_model_under_probability
        + holdout.residual_model_push_probability,
        1.0,
    )


def test_holdout_outcomes_do_not_change_frozen_holdout_predictions_or_probabilities():
    games = synthetic_games()
    first = walk_forward_market_probabilities(
        games, min_train_games=150, min_calibration_games=100, alpha=25.0
    )
    changed = games.copy()
    holdout_start = first.loc[first.evaluation_split.eq("reserved_holdout"), "game_date"].min()
    changed.loc[changed.game_date >= holdout_start, "actual_total"] += 30.0
    changed.loc[changed.game_date >= holdout_start, "market_residual_target"] += 30.0
    second = walk_forward_market_probabilities(
        changed, min_train_games=150, min_calibration_games=100, alpha=25.0
    )
    cols = ["residual_model_total", "residual_model_over_probability",
            "residual_model_under_probability", "residual_model_push_probability"]
    a = first.loc[first.evaluation_split.eq("reserved_holdout"), cols].to_numpy(float)
    b = second.loc[second.evaluation_split.eq("reserved_holdout"), cols].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_probability_metrics_condition_on_non_push_outcomes():
    frame = pd.DataFrame({
        "actual_total": [220.0, 221.0, 219.0],
        "entry_total": [220.0, 220.0, 220.0],
        "residual_model_over_probability": [0.2, 0.6, 0.3],
        "residual_model_under_probability": [0.2, 0.3, 0.6],
        "residual_model_push_probability": [0.6, 0.1, 0.1],
    })
    metrics = probability_metrics(frame, "residual_model_over_probability")
    assert metrics["games"] == 2
    assert metrics["mean_probability"] == pytest.approx((0.6 / 0.9 + 0.3 / 0.9) / 2)


def test_summary_never_claims_roi_or_profitability():
    games = synthetic_games()
    result = walk_forward_market_probabilities(
        games, min_train_games=150, min_calibration_games=100, alpha=25.0
    )
    summary = build_summary(result)
    assert summary["status"] == "completed"
    assert summary["interpretation"]["roi_claim"] is False
    assert "no bet selection" in summary["market_residual_probability_ev_diagnostics"]["note"].lower()
