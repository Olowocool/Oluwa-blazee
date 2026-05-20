import numpy as np


def classify_uncertainty(
    ensemble_probability,
    disagreement,
    probability_range,
    expected_value
):

    risk_score = 0

    # =========================
    # DISAGREEMENT RISK
    # =========================

    if disagreement >= 0.10:
        risk_score += 3

    elif disagreement >= 0.06:
        risk_score += 2

    elif disagreement >= 0.03:
        risk_score += 1

    # =========================
    # MODEL RANGE RISK
    # =========================

    if probability_range >= 0.20:
        risk_score += 3

    elif probability_range >= 0.12:
        risk_score += 2

    elif probability_range >= 0.06:
        risk_score += 1

    # =========================
    # EXTREME CONFIDENCE RISK
    # =========================

    if ensemble_probability >= 0.90:
        risk_score += 2

    elif ensemble_probability >= 0.82:
        risk_score += 1

    # =========================
    # LOW EDGE RISK
    # =========================

    if expected_value <= 0:
        risk_score += 3

    elif expected_value <= 0.03:
        risk_score += 2

    elif expected_value <= 0.06:
        risk_score += 1

    # =========================
    # FINAL CLASSIFICATION
    # =========================

    if risk_score >= 8:

        return {
            "uncertainty_level": "Extreme",
            "recommendation": "Avoid Bet",
            "risk_score": risk_score
        }

    if risk_score >= 5:

        return {
            "uncertainty_level": "High",
            "recommendation": "Very Risky",
            "risk_score": risk_score
        }

    if risk_score >= 3:

        return {
            "uncertainty_level": "Moderate",
            "recommendation": "Caution",
            "risk_score": risk_score
        }

    return {
        "uncertainty_level": "Low",
        "recommendation": "Stable",
        "risk_score": risk_score
    }
