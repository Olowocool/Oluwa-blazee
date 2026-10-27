"""Totals AI confidence is intentionally disabled pending leakage-free validation.

Do not load legacy model artifacts here: older artifacts may have been trained
with post-game columns such as actual_total and profit_loss.
"""


def totals_ai_prediction(projected_total, sportsbook_total, edge):
    return {
        "status": "disabled",
        "message": (
            "AI totals probabilities are disabled until a leakage-free model "
            "passes chronological out-of-sample validation and calibration."
        ),
        "over_probability": None,
        "under_probability": None,
        "confidence": None,
    }
