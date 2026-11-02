import numpy as np
import pandas as pd

from research.objective2_scoring_engine_competition import (
    COMPONENTS,
    apply_blend,
    date_cluster_bootstrap,
    fit_simplex_blend,
    metrics,
    simplex_weights,
)


def test_simplex_weights_are_nonnegative_and_sum_to_one():
    weights = list(simplex_weights(0.25))
    assert len(weights) == 35
    assert all(np.all(w >= 0) and np.isclose(w.sum(), 1.0) for w in weights)


def test_blend_is_fit_on_validation_and_applied_without_refitting():
    frame = pd.DataFrame({
        "actual_total": [220.0, 230.0, 240.0, 250.0],
        "v5_raw": [221.0, 231.0, 241.0, 251.0],
        "v2": [225.0, 235.0, 245.0, 255.0],
        "v4_phase1": [219.0, 229.0, 239.0, 249.0],
        "injury_rapm": [220.0, 230.0, 240.0, 250.0],
    })
    fit = fit_simplex_blend(frame)
    assert set(fit["weights"]) == set(COMPONENTS)
    assert np.isclose(sum(fit["weights"].values()), 1.0)
    scored = apply_blend(frame, fit)
    assert np.allclose(scored.simplex_blend, frame.actual_total)


def test_metrics_report_bias_and_accuracy():
    result = metrics([220.0, 230.0], [222.0, 227.0])
    assert result["games"] == 2
    assert result["mae"] == 2.5
    assert result["bias_prediction_minus_actual"] == -0.5


def test_date_cluster_bootstrap_is_paired_and_reproducible():
    frame = pd.DataFrame({
        "game_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
        "actual_total": [220.0, 230.0, 225.0, 240.0],
        "v5_raw": [225.0, 235.0, 230.0, 245.0],
        "candidate": [221.0, 231.0, 226.0, 241.0],
    })
    a = date_cluster_bootstrap(frame, "candidate", reps=300, seed=7)
    b = date_cluster_bootstrap(frame, "candidate", reps=300, seed=7)
    assert a == b
    assert a["candidate_minus_baseline_mae"] < 0
    assert a["game_dates"] == 3
