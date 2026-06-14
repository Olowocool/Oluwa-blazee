def classify_uncertainty(
    ensemble_probability,
    disagreement,
    probability_range,
    expected_value
):

    risk_score = 0

    if disagreement >= 0.20:
        risk_score += 3
    elif disagreement >= 0.15:
        risk_score += 2
    elif disagreement >= 0.10:
        risk_score += 1

    if probability_range >= 0.35:
        risk_score += 3
    elif probability_range >= 0.25:
        risk_score += 2
    elif probability_range >= 0.15:
        risk_score += 1

    if ensemble_probability >= 0.92:
        risk_score += 1

    if expected_value <= -0.03:
        risk_score += 3
    elif expected_value <= 0:
        risk_score += 1

    if risk_score >= 7:
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
