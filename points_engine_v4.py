"""Research-only V4 raw-total engine.

V4 estimates each team's scoring contribution and opponent defensive effect
with a recency-weighted, regularized linear model. It includes a learned
home-court scoring effect. It uses game scores only: no pace correction,
H2H adjustment, rest adjustment, odds, or betting-market inputs.

All training rows are filtered to dates strictly before the requested game.
"""
import numpy as np
import pandas as pd

from team_name_utils import normalize_history_team_columns, normalize_team_name


FALLBACK_TEAM_POINTS = 114.0
DEFAULT_HALF_LIFE_DAYS = 365.0
DEFAULT_PRIOR_GAMES = 8.0
DEFAULT_HOME_ADVANTAGE_PRIOR = 2.0
DEFAULT_HOME_ADVANTAGE_PRIOR_GAMES = 40.0


def _clean_full_history(history_df):
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=[
            "game_date", "home_team", "away_team", "home_score", "away_score"
        ])
    history = normalize_history_team_columns(history_df.copy())
    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [col for col in required if col not in history.columns]
    if missing:
        raise ValueError(f"V4 history is missing required columns: {missing}")
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.dropna(subset=required)
    history = history[
        (history["home_score"] > 0)
        & (history["away_score"] > 0)
        & (history["home_team"].astype(str).str.strip() != "")
        & (history["away_team"].astype(str).str.strip() != "")
    ].sort_values("game_date").reset_index(drop=True)
    return history


def _fit_weighted_ratings(
    history,
    target_date,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    prior_games=DEFAULT_PRIOR_GAMES,
    lookback_half_lives=4.0,
    home_advantage_prior=DEFAULT_HOME_ADVANTAGE_PRIOR,
    home_advantage_prior_games=DEFAULT_HOME_ADVANTAGE_PRIOR_GAMES,
):
    """Fit weighted score model using only games before target_date."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target game date: {target_date!r}")
    target = target.normalize()

    half_life_days = float(half_life_days)
    prior_games = float(prior_games)
    if half_life_days <= 0 or prior_games < 0:
        raise ValueError("half_life_days must be positive and prior_games nonnegative.")

    # Strict calendar-date cutoff: no game on the target date can enter ratings.
    past = history.loc[history["game_date"] < target].copy()
    if past.empty:
        return {
            "intercept": FALLBACK_TEAM_POINTS,
            "home_advantage": float(home_advantage_prior),
            "offense": {},
            "defense_weakness": {},
            "games_used": 0,
            "weighted_league_avg": FALLBACK_TEAM_POINTS,
        }

    age_days = (target - past["game_date"]).dt.days.astype(float)
    max_age = half_life_days * float(lookback_half_lives)
    keep = age_days <= max_age
    past = past.loc[keep].copy()
    age_days = age_days.loc[keep]
    if past.empty:
        return {
            "intercept": FALLBACK_TEAM_POINTS,
            "home_advantage": float(home_advantage_prior),
            "offense": {},
            "defense_weakness": {},
            "games_used": 0,
            "weighted_league_avg": FALLBACK_TEAM_POINTS,
        }

    weights = np.exp(-np.log(2.0) * age_days.to_numpy(dtype=float) / half_life_days)
    home_scores = past["home_score"].to_numpy(dtype=float)
    away_scores = past["away_score"].to_numpy(dtype=float)
    weighted_league_avg = float(
        np.sum(weights * (home_scores + away_scores))
        / (2.0 * np.sum(weights))
    )

    home_teams = past["home_team"].astype(str).to_numpy()
    away_teams = past["away_team"].astype(str).to_numpy()
    teams = sorted(set(home_teams) | set(away_teams))
    team_index = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(past)
    n_features = 2 + 2 * n_teams

    home_idx = np.fromiter((team_index[t] for t in home_teams), dtype=int, count=n_games)
    away_idx = np.fromiter((team_index[t] for t in away_teams), dtype=int, count=n_games)

    # Two observations per historical game:
    # Home score = intercept + home advantage + home offense + away defensive weakness
    # Away score = intercept + away offense + home defensive weakness
    feature_idx = np.full((2 * n_games, 4), -1, dtype=int)
    feature_idx[:n_games, 0] = 0
    feature_idx[:n_games, 1] = 1
    feature_idx[:n_games, 2] = 2 + home_idx
    feature_idx[:n_games, 3] = 2 + n_teams + away_idx

    feature_idx[n_games:, 0] = 0
    feature_idx[n_games:, 1] = 2 + away_idx
    feature_idx[n_games:, 2] = 2 + n_teams + home_idx

    outcomes = np.concatenate([home_scores, away_scores])
    obs_weights = np.concatenate([weights, weights])

    # Accumulate X'WX and X'Wy from the sparse four-feature rows without
    # constructing a large dense design matrix.
    gram = np.zeros((n_features, n_features), dtype=float)
    rhs = np.zeros(n_features, dtype=float)
    i_idx = np.broadcast_to(feature_idx[:, :, None], (2 * n_games, 4, 4))
    j_idx = np.broadcast_to(feature_idx[:, None, :], (2 * n_games, 4, 4))
    pair_mask = (i_idx >= 0) & (j_idx >= 0)
    pair_weights = np.broadcast_to(obs_weights[:, None, None], (2 * n_games, 4, 4))
    np.add.at(gram, (i_idx[pair_mask], j_idx[pair_mask]), pair_weights[pair_mask])

    valid_features = feature_idx >= 0
    rhs_values = np.broadcast_to((obs_weights * outcomes)[:, None], feature_idx.shape)
    np.add.at(rhs, feature_idx[valid_features], rhs_values[valid_features])

    # Ridge priors shrink team offense/defense toward league average (zero).
    # The home-court coefficient is softly anchored to a modest prior.
    diagonal = np.arange(n_features)
    gram[diagonal[2:], diagonal[2:]] += prior_games
    gram[1, 1] += float(home_advantage_prior_games)
    rhs[1] += float(home_advantage_prior_games) * float(home_advantage_prior)

    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    offense = {
        team: float(coefficients[2 + idx])
        for team, idx in team_index.items()
    }
    defense = {
        team: float(coefficients[2 + n_teams + idx])
        for team, idx in team_index.items()
    }
    return {
        "intercept": float(coefficients[0]),
        "home_advantage": float(coefficients[1]),
        "offense": offense,
        "defense_weakness": defense,
        "games_used": int(n_games),
        "weighted_league_avg": weighted_league_avg,
    }


def calculate_matchup_points_v4(
    home_team,
    away_team,
    history_df,
    game_date,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    prior_games=DEFAULT_PRIOR_GAMES,
    _prepared_cache=None,
):
    """Return V4 raw points prediction for one matchup."""
    target = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"V4 requires a valid game_date; received {game_date!r}")
    target = target.normalize()
    cache_key = (
        target.date().isoformat(),
        float(half_life_days),
        float(prior_games),
    )

    prepared = _prepared_cache.get(cache_key) if _prepared_cache is not None else None
    if prepared is None:
        # Reuse a single normalized copy of the full history across all dates.
        history_key = "__v4_clean_history__"
        history = _prepared_cache.get(history_key) if _prepared_cache is not None else None
        if history is None:
            history = _clean_full_history(history_df)
            if _prepared_cache is not None:
                _prepared_cache[history_key] = history
        prepared = _fit_weighted_ratings(
            history,
            target,
            half_life_days=half_life_days,
            prior_games=prior_games,
        )
        if _prepared_cache is not None:
            _prepared_cache[cache_key] = prepared

    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)
    offense = prepared["offense"]
    defense = prepared["defense_weakness"]

    home_points = (
        prepared["intercept"]
        + prepared["home_advantage"]
        + offense.get(home_key, 0.0)
        + defense.get(away_key, 0.0)
    )
    away_points = (
        prepared["intercept"]
        + offense.get(away_key, 0.0)
        + defense.get(home_key, 0.0)
    )
    return {
        "home_team": home_team,
        "away_team": away_team,
        "projected_home_points": round(float(home_points), 2),
        "projected_away_points": round(float(away_points), 2),
        "raw_total": round(float(home_points + away_points), 2),
        "intercept": round(float(prepared["intercept"]), 3),
        "home_advantage": round(float(prepared["home_advantage"]), 3),
        "weighted_league_avg_team_points": round(
            float(prepared["weighted_league_avg"]), 2
        ),
        "games_used": int(prepared["games_used"]),
        "home_offense_rating": round(float(offense.get(home_key, 0.0)), 3),
        "away_offense_rating": round(float(offense.get(away_key, 0.0)), 3),
        "home_defense_weakness": round(float(defense.get(home_key, 0.0)), 3),
        "away_defense_weakness": round(float(defense.get(away_key, 0.0)), 3),
        "half_life_days": float(half_life_days),
        "prior_games": float(prior_games),
    }
