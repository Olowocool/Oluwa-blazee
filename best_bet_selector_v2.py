def select_best_bet_v2(
    moneyline_confidence,
    moneyline_pick,
    totals_confidence,
    totals_pick
):

    moneyline_confidence = float(moneyline_confidence)
    totals_confidence = float(totals_confidence)

    if moneyline_confidence >= totals_confidence:

        return {
            "market": "Moneyline",
            "pick": moneyline_pick,
            "confidence": round(moneyline_confidence, 1)
        }

    return {
        "market": "Totals",
        "pick": totals_pick,
        "confidence": round(totals_confidence, 1)
    }
