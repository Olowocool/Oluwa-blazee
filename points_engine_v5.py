"""Research-only V5 raw-total engine: 55% V3 + 45% V4 (90-day).

This is a composition of the already-tested V3 and V4-90 components.
It does not modify either engine.

Raw total includes:
- V3 current-season opponent-adjusted offense/defense and recent form;
- V4 90-day recency-weighted opponent-adjusted ratings and learned home effect.

It deliberately excludes pace, H2H, rest, bookmaker lines, and market inputs.
Both component engines enforce a strict pre-game-date history cutoff.
"""
from points_engine_v3 import calculate_matchup_points_v3
from points_engine_v4 import calculate_matchup_points_v4


V5_V3_WEIGHT = 0.55
V5_V4_WEIGHT = 0.45
V5_V4_HALF_LIFE_DAYS = 90.0
V5_V4_PRIOR_GAMES = 8.0


def calculate_matchup_points_v5(
    home_team,
    away_team,
    history_df,
    game_date,
    _prepared_cache=None,
):
    """Return reconciled V5 home/away points and raw game total.

    Pass the same _prepared_cache dictionary across games/dates in a
    chronological backtest to reuse V3 and V4 date-level calculations.
    """
    if history_df is None:
        raise ValueError("V5 requires a historical score dataframe.")
    if game_date is None:
        raise ValueError("V5 requires an explicit target game_date.")

    cache = _prepared_cache if _prepared_cache is not None else {}
    v3_cache = cache.setdefault("v3", {})
    v4_cache = cache.setdefault("v4", {})

    v3 = calculate_matchup_points_v3(
        home_team,
        away_team,
        history_df,
        game_date=game_date,
        _prepared_cache=v3_cache,
    )
    v4 = calculate_matchup_points_v4(
        home_team,
        away_team,
        history_df,
        game_date=game_date,
        half_life_days=V5_V4_HALF_LIFE_DAYS,
        prior_games=V5_V4_PRIOR_GAMES,
        _prepared_cache=v4_cache,
    )

    home_points = (
        V5_V3_WEIGHT * float(v3["projected_home_points"])
        + V5_V4_WEIGHT * float(v4["projected_home_points"])
    )
    away_points = (
        V5_V3_WEIGHT * float(v3["projected_away_points"])
        + V5_V4_WEIGHT * float(v4["projected_away_points"])
    )

    # Keep the displayed component projections and game total internally
    # consistent. The component engines expose scores rounded to two decimals.
    home_points = round(home_points, 2)
    away_points = round(away_points, 2)
    raw_total = round(home_points + away_points, 2)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "projected_home_points": home_points,
        "projected_away_points": away_points,
        "raw_total": raw_total,
        "v3_raw_total": float(v3["raw_total"]),
        "v4_90_raw_total": float(v4["raw_total"]),
        "v3_weight": V5_V3_WEIGHT,
        "v4_weight": V5_V4_WEIGHT,
        "v4_half_life_days": V5_V4_HALF_LIFE_DAYS,
        "v4_prior_games": V5_V4_PRIOR_GAMES,
        "season_start_year": v3["season_start_year"],
        "v3_season_games_used": int(v3["season_games_used"]),
        "v3_home_recent_games_used": int(v3["home_recent_games_used"]),
        "v3_away_recent_games_used": int(v3["away_recent_games_used"]),
        "v3_home_offense_rating": float(v3["home_offense_rating"]),
        "v3_away_offense_rating": float(v3["away_offense_rating"]),
        "v3_home_defense_weakness": float(v3["home_defense_weakness"]),
        "v3_away_defense_weakness": float(v3["away_defense_weakness"]),
        "v4_home_advantage": float(v4["home_advantage"]),
        "v4_weighted_league_avg_team_points": float(
            v4["weighted_league_avg_team_points"]
        ),
        "v4_games_used": int(v4["games_used"]),
        "history_cutoff": str(game_date),
    }
