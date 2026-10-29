import ast
import unittest
from pathlib import Path

from legacy_training_guard import (
    LEGACY_PIPELINE_BLOCK_REASON,
    blocked_legacy_pipeline_result,
)


ROOT = Path(__file__).resolve().parents[1]


class LegacyTrainingGuardTests(unittest.TestCase):
    def test_legacy_pipeline_result_is_fail_closed(self):
        result = blocked_legacy_pipeline_result("ensemble training")

        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["evaluation_allowed"])
        self.assertFalse(result["model_written"])
        self.assertFalse(result["metrics_valid"])
        self.assertEqual(result["reason_code"], "unverified_point_in_time_features")
        self.assertIn("chronological holdout", result["message"])
        self.assertEqual(result["message"], LEGACY_PIPELINE_BLOCK_REASON)

    def test_unsafe_entry_points_return_guard_before_loading_data_or_training(self):
        entry_points = {
            "train_ensemble_model.py": "train_ensemble_model",
            "model_evaluation.py": "evaluate_ensemble_model",
            "retrain_model.py": "retrain_pipeline",
        }

        for filename, function_name in entry_points.items():
            with self.subTest(file=filename, function=function_name):
                module = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
                function = next(
                    node for node in module.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function_name
                )
                first_statement = function.body[0]
                self.assertIsInstance(first_statement, ast.Return)
                call = first_statement.value
                self.assertIsInstance(call, ast.Call)
                self.assertIsInstance(call.func, ast.Name)
                self.assertEqual(call.func.id, "blocked_legacy_pipeline_result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
