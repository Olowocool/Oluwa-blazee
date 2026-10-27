def select_best_bet_v2(
    moneyline_confidence,
    moneyline_pick,
    totals_confidence,
    totals_pick
):
    """Select only among confidence values that are actually available.

    A missing totals confidence means the totals AI is disabled/unvalidated,
    not that the totals market has zero risk or a valid score of zero.
    """
    try:
        moneyline_confidence = float(moneyline_confidence)
    except (TypeError, ValueError):
        moneyline_confidence = None

    try:
        totals_confidence = float(totals_confidence)
    except (TypeError, ValueError):
        totals_confidence = None

    if totals_confidence is None:
        return {
            "market": "Moneyline",
            "pick": moneyline_pick,
            "confidence": round(moneyline_confidence, 1) if moneyline_confidence is not None else None,
            "reason": "Totals AI confidence is disabled pending validation.",
        }

    if moneyline_confidence is None or totals_confidence > moneyline_confidence:
        return {
            "market": "Totals",
            "pick": totals_pick,
            "confidence": round(totals_confidence, 1),
        }

    return {
        "market": "Moneyline",
        "pick": moneyline_pick,
        "confidence": round(moneyline_confidence, 1),
    }
