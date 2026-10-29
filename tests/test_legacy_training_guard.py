import ast
import unittest
from pathlib import Path

from legacy_training_guard import (
    LEGACY_PIPELINE_BLOCK_REASON,
    blocked_legacy_pipeline_result,
    finalize_automation_status,
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

    def test_automation_is_partial_when_any_step_is_blocked_or_fails(self):
        summary = {
            "status": "success",
            "steps": {
                "result_sync": {"status": "success"},
                "ensemble_training": {"status": "blocked"},
                "model_evaluation": {"status": "error"},
            },
        }

        result = finalize_automation_status(summary)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["incomplete_steps"], ["ensemble_training", "model_evaluation"])
        self.assertIn("do not treat this run as a complete model update", result["message"])
        self.assertEqual(summary["status"], "success")

    def test_automation_stays_success_when_all_steps_succeed(self):
        result = finalize_automation_status({
            "status": "success",
            "steps": {
                "result_sync": {"status": "success"},
                "learning_dataset": {"status": "success"},
            },
        })

        self.assertEqual(result["status"], "success")
        self.assertNotIn("incomplete_steps", result)

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

    def test_direct_legacy_trainer_is_also_blocked(self):
        module = ast.parse(
            (ROOT / "retrain_model.py").read_text(encoding="utf-8")
        )
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_model"
        )
        first_statement = function.body[0]
        self.assertIsInstance(first_statement, ast.Raise)
        self.assertIsInstance(first_statement.exc, ast.Call)
        self.assertIsInstance(first_statement.exc.func, ast.Name)
        self.assertEqual(first_statement.exc.func.id, "RuntimeError")

    def test_post_result_fields_are_not_candidate_predictors(self):
        source = (ROOT / "retrain_model.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        train_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_training_data"
        )
        candidate_assignments = [
            node for node in ast.walk(train_function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "feature_candidates"
                for target in node.targets
            )
        ]
        self.assertEqual(len(candidate_assignments), 1)
        candidate_names = [
            node.value for node in ast.walk(candidate_assignments[0])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertNotIn("clv", candidate_names)
        self.assertNotIn("profit_loss", candidate_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
