import numpy as np
import pandas as pd

from research.smart_total_probability import (
    FEATURES,
    chronological_splits,
    conformal_interval,
    empirical_over_probability,
    regression_metrics,
)


def test_empirical_over_probability_is_monotone_and_smoothed():
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    probabilities = empirical_over_probability(
        predicted_median=[100.0, 100.0, 100.0],
        market_line=[99.0, 100.0, 101.0],
        calibration_residuals=residuals,
    )
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_regression_metrics_are_correct():
    result = regression_metrics([220.0, 230.0], [222.0, 227.0])
    assert result["games"] == 2
    assert result["mae"] == 2.5
    assert result["bias_prediction_minus_actual"] == -0.5


def test_chronological_splits_are_disjoint_and_ordered():
    rows = []
    for date in ("2023-12-01", "2024-11-01", "2025-11-01"):
        row = {column: 0.0 for column in FEATURES}
        row.update({
            "game_date": pd.Timestamp(date),
            "game_type": "regular_season",
            "actual_total_source": 220.0,
        })
        rows.append(row)
    train, validation, holdout = chronological_splits(pd.DataFrame(rows))
    assert len(train) == len(validation) == len(holdout) == 1
    assert train.game_date.max() < validation.game_date.min()
    assert validation.game_date.max() < holdout.game_date.min()


def test_conformal_correction_is_nonnegative():
    lower = np.array([210.0, 211.0, 212.0])
    upper = np.array([240.0, 241.0, 242.0])
    actual = np.array([205.0, 230.0, 245.0])
    correction, _ = conformal_interval(lower, upper, actual, coverage=0.8)
    assert correction >= 0.0
