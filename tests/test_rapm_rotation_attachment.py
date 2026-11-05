import sys
from pathlib import Path

import pandas as pd
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from research.v4_phase2_rotation import attach_rapm_ratings


def test_attach_rapm_ratings_populates_missing_columns_from_player_map():
    rotation = pd.DataFrame({"player_id": ["p1", "p2", "p3"]})
    ratings = {
        "p1": {"o_rapm": 1.25, "d_rapm": -0.75},
        "p2": {"o_rapm": -0.40, "d_rapm": 0.30},
    }

    result = attach_rapm_ratings(rotation, ratings)

    assert result.o_rapm.tolist() == pytest.approx([1.25, -0.40, 0.0])
    assert result.d_rapm.tolist() == pytest.approx([-0.75, 0.30, 0.0])


def test_attach_rapm_ratings_fills_nan_but_preserves_valid_existing_values():
    rotation = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "o_rapm": [None, 9.0],
        "d_rapm": [None, -8.0],
    })
    ratings = {
        "p1": {"o_rapm": 1.5, "d_rapm": -0.5},
        "p2": {"o_rapm": 2.5, "d_rapm": -1.5},
    }

    result = attach_rapm_ratings(rotation, ratings)

    assert result.o_rapm.tolist() == pytest.approx([1.5, 9.0])
    assert result.d_rapm.tolist() == pytest.approx([-0.5, -8.0])


def test_attach_rapm_ratings_rejects_missing_player_id():
    with pytest.raises(ValueError, match="player_id"):
        attach_rapm_ratings(pd.DataFrame({"name": ["Player"]}), {})
