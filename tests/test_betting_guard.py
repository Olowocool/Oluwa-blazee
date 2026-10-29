import unittest

from betting_guard import (
    BETTING_VALIDATION_COMPLETE,
    actionable_bet_allowed,
    betting_gate_message,
)


class BettingGuardTests(unittest.TestCase):
    def test_release_defaults_to_research_only(self):
        self.assertFalse(BETTING_VALIDATION_COMPLETE)
        self.assertFalse(actionable_bet_allowed(True))
        self.assertIn("research-only", betting_gate_message())

    def test_candidate_filters_cannot_override_incomplete_validation(self):
        self.assertFalse(
            actionable_bet_allowed(
                True,
                validation_complete=False,
                test_mode=False,
            )
        )

    def test_test_mode_never_bypasses_validation_gate(self):
        self.assertFalse(
            actionable_bet_allowed(
                True,
                validation_complete=False,
                test_mode=True,
            )
        )
        self.assertFalse(
            actionable_bet_allowed(
                True,
                validation_complete=True,
                test_mode=True,
            )
        )

    def test_validated_candidate_still_must_pass_filters(self):
        self.assertFalse(
            actionable_bet_allowed(
                False,
                validation_complete=True,
                test_mode=False,
            )
        )
        self.assertTrue(
            actionable_bet_allowed(
                True,
                validation_complete=True,
                test_mode=False,
            )
        )

    def test_gate_message_does_not_claim_validation_when_closed(self):
        self.assertIn(
            "no actionable betting recommendation is authorized",
            betting_gate_message(validation_complete=False),
        )
        self.assertIn(
            "ordinary value/risk filters still apply",
            betting_gate_message(validation_complete=True),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
