import unittest

import pandas as pd

from research.prospective_rotation_projection import validate_rotation_projection


class ProspectiveRotationProjectionTests(unittest.TestCase):
    def valid_rows(self):
        return pd.DataFrame([
            {"game_id": "g1", "team_id": "t1", "player_id": f"p{i}",
             "rotation_rank": i, "expected_minutes": 30.0,
             "baseline_minutes": 30.0, "status": "NOT_REPORTED",
             "availability_probability": 1.0, "is_rotation_tail": False}
            for i in range(1, 9)
        ])

    def test_full_player_level_rotation_reconciles_to_240(self):
        validate_rotation_projection(self.valid_rows(), "g1", "t1")

    def test_underallocated_rotation_is_rejected(self):
        rows = self.valid_rows()
        rows.loc[0, "expected_minutes"] = 29.0
        with self.assertRaisesRegex(ValueError, "do not reconcile to 240"):
            validate_rotation_projection(rows, "g1", "t1")

    def test_over_48_minute_player_is_rejected(self):
        rows = self.valid_rows()
        rows.loc[0, "expected_minutes"] = 49.0
        rows.loc[1:, "expected_minutes"] = 191.0 / 7.0
        with self.assertRaisesRegex(ValueError, "48-minute"):
            validate_rotation_projection(rows, "g1", "t1")

    def test_synthetic_tail_is_rejected(self):
        rows = self.valid_rows()
        rows.loc[0, "is_rotation_tail"] = True
        with self.assertRaisesRegex(ValueError, "tail"):
            validate_rotation_projection(rows, "g1", "t1")

    def test_duplicate_player_is_rejected(self):
        rows = self.valid_rows()
        rows.loc[1, "player_id"] = rows.loc[0, "player_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate player"):
            validate_rotation_projection(rows, "g1", "t1")

    def test_non_finite_minutes_are_rejected(self):
        rows = self.valid_rows()
        rows.loc[0, "expected_minutes"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_rotation_projection(rows, "g1", "t1")


if __name__ == "__main__":
    unittest.main()
