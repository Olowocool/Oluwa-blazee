"""
betting_guards.py

Guard rails for the moneyline and totals recommendations.
All tunables are constants at the top. NONE of them has been fitted on real
sportsbook data yet (the project had no historical totals lines or odds).
Re-fit them once you have a few thousand logged games with closing lines.
"""

# ---- Moneyline -------------------------------------------------------------
# How much of the model's disagreement with the no-vig market to trust.
# 0 = trust the market fully, 1 = trust the model fully.
MARKET_SHRINK_K = 0.25

# If the model differs from the no-vig market by more than this (in probability),
# assume the model is missing information (injuries, lineups, etc.) and do not bet.
MAX_MODEL_MARKET_GAP = 0.10

# Staking: fractional Kelly, hard-capped as a share of bankroll.
KELLY_MULTIPLIER = 0.10
MAX_STAKE_FRACTION = 0.03

# ---- Totals ----------------------------------------------------------------
TOTALS_LEAN_EDGE = 2.5      # unchanged from the original code
TOTALS_STRONG_EDGE = 5.0    # unchanged from the original code
TOTALS_MAX_CREDIBLE_EDGE = 7.0   # bigger gaps vs the market are treated as model error

# The totals model stacked offense/defense/home-away/rest adjustments on top of a
# points engine that already uses the same inputs (rest was applied twice).
# On 600 recent games this raised RMSE from 18.89 to 19.22 and widened the spread
# of projections from SD 6.5 to 10.2. Set True to restore the old behaviour.
USE_STACKED_ADJUSTMENTS = False

# A team's "last 10 games" in the first weeks of a season are mostly last season.
# Below this many games in the last EARLY_SEASON_WINDOW_DAYS days, no totals bet.
# Set EARLY_SEASON_MIN_GAMES = 0 to disable.
EARLY_SEASON_MIN_GAMES = 8
EARLY_SEASON_WINDOW_DAYS = 45

# totals_ai_predictor loads a model trained with actual_total / profit_loss as
# features (label leakage) and feeds it zeros at prediction time. Keep disabled.
USE_TOTALS_AI_MODEL = False


def devig_two_way(odds_a, odds_b):
    """Decimal odds for both sides -> (fair_prob_a, fair_prob_b) with the vig removed."""
    ia, ib = 1.0 / float(odds_a), 1.0 / float(odds_b)
    total = ia + ib
    return ia / total, ib / total


def shrink_to_market(model_prob, fair_prob, k=MARKET_SHRINK_K):
    """Pull the model probability toward the no-vig market probability."""
    return fair_prob + k * (model_prob - fair_prob)


def _side(model_prob, fair_prob, odds):
    adj = shrink_to_market(model_prob, fair_prob)
    b = float(odds) - 1.0
    ev = adj * b - (1.0 - adj)                      # EV per 1 unit, vig included via odds
    kelly = max(ev / b, 0.0) if b > 0 else 0.0
    stake = min(kelly * KELLY_MULTIPLIER, MAX_STAKE_FRACTION)
    return {
        "model_prob": model_prob,
        "fair_prob": fair_prob,
        "adj_prob": adj,
        "edge": adj - fair_prob,
        "ev": ev,
        "kelly": kelly,
        "stake_fraction": stake,
        "gap": model_prob - fair_prob,
    }


def assess_moneyline(model_home_prob, home_odds, away_odds):
    fair_home, fair_away = devig_two_way(home_odds, away_odds)
    home = _side(model_home_prob, fair_home, home_odds)
    away = _side(1.0 - model_home_prob, fair_away, away_odds)

    gap = abs(model_home_prob - fair_home)
    blocked = gap > MAX_MODEL_MARKET_GAP
    reason = ""
    if blocked:
        reason = (
            f"Model differs from the no-vig market by {gap * 100:.1f} points "
            f"(limit {MAX_MODEL_MARKET_GAP * 100:.0f}). That is far more likely to be "
            f"missing information (injuries, lineups, early-season sample) than a "
            f"market mistake. No bet."
        )
    if blocked:
        # never suggest a stake on a blocked game
        home["stake_fraction"] = 0.0
        away["stake_fraction"] = 0.0
    return {"home": home, "away": away, "gap": gap, "blocked": blocked, "reason": reason}


def classify_totals_edge(edge):
    """edge = projected_total - market_total. Returns (recommendation, note)."""
    e = float(edge)
    if abs(e) > TOTALS_MAX_CREDIBLE_EDGE:
        return (
            "No Bet",
            f"Model is {abs(e):.1f} pts from the market (limit {TOTALS_MAX_CREDIBLE_EDGE:.0f}) "
            f"- treated as model error, not an edge",
        )
    if e >= TOTALS_STRONG_EDGE:
        return "Strong Over", "Strong Over edge"
    if e >= TOTALS_LEAN_EDGE:
        return "Lean Over", "Small Over edge"
    if e <= -TOTALS_STRONG_EDGE:
        return "Strong Under", "Strong Under edge"
    if e <= -TOTALS_LEAN_EDGE:
        return "Lean Under", "Small Under edge"
    return "No Bet", "Edge too small"
