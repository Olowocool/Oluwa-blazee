from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v4_v2_residual_blend import (
    apply_blend, date_cluster_bootstrap, fit_blend_weight, run_evaluation,
)


def sample_rows():
    rows = []
    for season in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
        for index in range(4):
            v2 = 210.0 + index * 3 + season % 4
            gap = float(index - 1.5)
            actual = v2 + 0.4 * gap + (0.3 if index % 2 else -0.3)
            date = pd.Timestamp(f"{season - 1}-11-01") + pd.Timedelta(days=index)
            rows.append({
                "season": season, "game_id": f"{season}-{index}",
                "game_date": date, "actual_total": actual,
                "v2_points_engine_total": v2,
                "v4_pace_efficiency_total": v2 + gap,
                "v2_max_history_date": date - pd.Timedelta(days=2),
                "v4_max_history_date": date - pd.Timedelta(days=1),
            })
    return pd.DataFrame(rows)


def test_fit_blend_weight_recovers_known_convex_coefficient():
    frame = sample_rows().query("season in [2020, 2021, 2022]").copy()
    frame["actual_total"] = (
        frame["v2_points_engine_total"] + 0.4 * (
            frame["v4_pace_efficiency_total"] - frame["v2_points_engine_total"]
        )
    )
    fitted = fit_blend_weight(frame)
    assert fitted["beta"] == pytest.approx(0.4)
    assert fitted["raw_beta"] == pytest.approx(0.4)


def test_apply_blend_interpolates_between_v2_and_v4():
    frame = sample_rows().head(3)
    out = apply_blend(frame, 0.25)
    expected = frame["v2_points_engine_total"] + 0.25 * (
        frame["v4_pace_efficiency_total"] - frame["v2_points_engine_total"]
    )
    assert np.allclose(out["blend_prediction"], expected)


def test_holdout_outcomes_do_not_change_fitted_coefficient_or_predictions():
    frame = sample_rows()
    baseline = run_evaluation(frame)
    changed = frame.copy()
    changed.loc[changed["season"].isin([2024, 2025, 2026]), "actual_total"] += 1000
    altered = run_evaluation(changed)
    assert altered["final_fit"]["beta"] == pytest.approx(baseline["final_fit"]["beta"])
    assert altered["final_fit"]["games"] == baseline["final_fit"]["games"]
    assert altered["holdout_scored"]["blend_prediction"].equals(
        baseline["holdout_scored"]["blend_prediction"]
    )


def test_date_cluster_bootstrap_is_deterministic_and_paired():
    frame = pd.DataFrame({
        "game_date": pd.to_datetime([
            "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"
        ]),
        "candidate_abs": [1.0, 2.0, 1.0, 3.0],
        "baseline_abs": [2.0, 4.0, 2.0, 2.0],
    })
    first = date_cluster_bootstrap(frame, "candidate_abs", "baseline_abs", reps=300, seed=42)
    second = date_cluster_bootstrap(frame, "candidate_abs", "baseline_abs", reps=300, seed=42)
    assert first == second
    assert first["observed_mae_difference"] == pytest.approx(-0.75)
    assert first["game_dates"] == 3
    assert first["games"] == 4
