import pandas as pd
import pytest

from research.v4_phase2_rotation import attach_player_rapm_ratings


def test_missing_rapm_columns_are_populated_from_fitted_player_ratings():
    rotation = pd.DataFrame({"player_id": ["101", "202", "303"]})
    ratings = {
        "101": {"o_rapm": 1.25, "d_rapm": -0.75},
        "202": {"o_rapm": -0.40, "d_rapm": 0.60},
    }
    result = attach_player_rapm_ratings(rotation, ratings)
    assert result.o_rapm.tolist() == pytest.approx([1.25, -0.40, 0.0])
    assert result.d_rapm.tolist() == pytest.approx([-0.75, 0.60, 0.0])


def test_existing_non_null_rapm_values_are_preserved_and_nulls_are_filled():
    rotation = pd.DataFrame({
        "player_id": ["101", "202"],
        "o_rapm": [2.0, None],
        "d_rapm": [None, -1.5],
    })
    ratings = {
        "101": {"o_rapm": 1.25, "d_rapm": -0.75},
        "202": {"o_rapm": -0.40, "d_rapm": 0.60},
    }
    result = attach_player_rapm_ratings(rotation, ratings)
    assert result.o_rapm.tolist() == pytest.approx([2.0, -0.40])
    assert result.d_rapm.tolist() == pytest.approx([-0.75, -1.5])
