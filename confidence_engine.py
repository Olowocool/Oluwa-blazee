def classify_confidence(
    model_probability,
    expected_value,
    kelly,
    disagreement=0,
    line_movement_diff=0,
    sharp_support_pct=0
):
    model_probability = float(model_probability)
    expected_value = float(expected_value)
    kelly = float(kelly)
    disagreement = float(disagreement)
    line_movement_diff = float(line_movement_diff)
    sharp_support_pct = float(sharp_support_pct)

    score = 0

    if model_probability >= 0.60:
        score += 2
    elif model_probability >= 0.55:
        score += 1

    if expected_value >= 0.08:
        score += 2
    elif expected_value >= 0.03:
        score += 1

    if kelly >= 0.05:
        score += 2
    elif kelly >= 0.02:
        score += 1

    if sharp_support_pct >= 0.65:
        score += 1

    if abs(line_movement_diff) >= 3:
        score += 1

    if disagreement >= 0.20:
        score -= 2
    elif disagreement >= 0.12:
        score -= 1

    if score >= 6:
        tier = "Elite"
        action = "Bet"
    elif score >= 4:
        tier = "Strong"
        action = "Bet Small"
    elif score >= 2:
        tier = "Watchlist"
        action = "Monitor"
    else:
        tier = "Avoid"
        action = "No Bet"

    return {
        "confidence_score": score,
        "confidence_tier": tier,
        "recommended_action": action
    }
