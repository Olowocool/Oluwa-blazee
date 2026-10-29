"""Fail-closed gate for unvalidated model betting recommendations.

Paper tracking and research signals remain available while validation is
incomplete. This module does not place sportsbook wagers.
"""

BETTING_VALIDATION_COMPLETE = False


def actionable_bet_allowed(
    candidate_passes_filters,
    *,
    validation_complete=BETTING_VALIDATION_COMPLETE,
    test_mode=False,
):
    """Allow an actionable recommendation only after validation is approved.

    Test mode is never an override for the validation gate. The app may still
    show research-only model outputs or record paper candidates while blocked.
    """
    return bool(
        validation_complete
        and not test_mode
        and candidate_passes_filters
    )


def betting_gate_message(*, validation_complete=BETTING_VALIDATION_COMPLETE):
    if validation_complete:
        return "Validation gate is open; ordinary value/risk filters still apply."
    return (
        "Betting validation is incomplete. Treat model outputs as research-only; "
        "no actionable betting recommendation is authorized."
    )
