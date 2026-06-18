def select_best_bet(
    moneyline_pick,
    moneyline_ev,
    moneyline_confidence,
    totals_edge,
    totals_recommendation
):
    moneyline_score = (
        (moneyline_ev * 100)
        + moneyline_confidence
    )

    totals_score = abs(totals_edge)

    if totals_recommendation == "No Bet":
        totals_score = 0

    if totals_score > moneyline_score:
        return {
            "market": "Totals",
            "pick": totals_recommendation
        }

    if moneyline_score > 0:
        return {
            "market": "Moneyline",
            "pick": moneyline_pick
        }

    return {
        "market": "No Bet",
        "pick": "No Bet"
    }
