"""Research-only Points Engine V3 candidate.

This module deliberately does not replace points_engine.py. It provides a
separate raw-total candidate for chronological comparison against V2.

Design:
- current-season, opponent-adjusted offensive and defensive scoring ratings;
- recent form measured as residuals against those ratings (not overlapping
  last-5 and last-10 raw PPG averages);
- residuals shrunk toward zero for small samples;
- no fixed +1.5/-1.0 venue constants and no H2H/rest adjustments here.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from points_engine import history_strictly_before_target_date
from team_name_utils import normalize_history_team_columns, normalize_team_name


NBA_FALLBACK_TEAM_POINTS = 114.0
RATING_PRIOR_GAMES = 8.0
FORM_WINDOW = 10
FORM_PRIOR_GAMES = 5.0
FORM_WEIGHT = 0.25
RATING_MAX_ITERATIONS = 100
RATING_TOLERANCE = 1e-4


def nba_season_start_year(value):
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"Invalid NBA season date: {value!r}")
    return int(date.year if date.month >= 10 else date.year - 1)


def _clean_history(history_df, game_date):
    if history_df is None:
        return pd.DataFrame(columns=[
            "game_date", "home_team", "away_team", "home_score", "away_score"
        ])
    history = history_strictly_before_target_date(history_df, game_date)
    if history is None or history.empty:
        return pd.DataFrame(columns=[
            "game_date", "home_team", "away_team", "home_score", "away_score"
        ])
    history = normalize_history_team_columns(history).copy()
    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [col for col in required if col not in history.columns]
    if missing:
        raise ValueError(f"V3 history is missing required columns: {missing}")
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


def _fit_season_ratings(season_games):
    """Fit additive opponent-adjusted offense and defensive-weakness ratings."""
    if season_games.empty:
        return {
            "league_avg": NBA_FALLBACK_TEAM_POINTS,
            "offense": {},
            "defense_weakness": {},
            "games": {},
            "iterations": 0,
        }

    league_avg = float(
        (season_games["home_score"].sum() + season_games["away_score"].sum())
        / (2 * len(season_games))
    )
    teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
    offense = {team: 0.0 for team in teams}
    defense = {team: 0.0 for team in teams}
    game_counts = defaultdict(int)

    observations = []
    for row in season_games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        hs = float(row.home_score)
        aws = float(row.away_score)
        observations.append((home, away, hs, aws))
        game_counts[home] += 1
        game_counts[away] += 1

    iterations = 0
    for iteration in range(RATING_MAX_ITERATIONS):
        old_offense = offense.copy()
        old_defense = defense.copy()

        offense_sum = defaultdict(float)
        offense_n = defaultdict(int)
        defense_sum = defaultdict(float)
        defense_n = defaultdict(int)

        for home, away, home_score, away_score in observations:
            offense_sum[home] += home_score - league_avg - defense[away]
            offense_n[home] += 1
            offense_sum[away] += away_score - league_avg - defense[home]
            offense_n[away] += 1

            # Positive defensive_weakness means the team allows more than expected.
            defense_sum[away] += home_score - league_avg - offense[home]
            defense_n[away] += 1
            defense_sum[home] += away_score - league_avg - offense[away]
            defense_n[home] += 1

        for team in teams:
            offense[team] = offense_sum[team] / (
                offense_n[team] + RATING_PRIOR_GAMES
            )
            defense[team] = defense_sum[team] / (
                defense_n[team] + RATING_PRIOR_GAMES
            )

        # Center both ratings so the league intercept remains interpretable.
        off_mean = float(np.mean(list(offense.values()))) if offense else 0.0
        def_mean = float(np.mean(list(defense.values()))) if defense else 0.0
        offense = {team: value - off_mean for team, value in offense.items()}
        defense = {team: value - def_mean for team, value in defense.items()}

        max_change = max(
            [abs(offense[t] - old_offense[t]) for t in teams]
            + [abs(defense[t] - old_defense[t]) for t in teams]
        ) if teams else 0.0
        iterations = iteration + 1
        if max_change < RATING_TOLERANCE:
            break

    return {
        "league_avg": league_avg,
        "offense": offense,
        "defense_weakness": defense,
        "games": dict(game_counts),
        "iterations": iterations,
    }


def _recent_form(history, team, ratings):
    team_key = normalize_team_name(team).casefold()
    team_games = history[
        (history["home_team"].astype(str).str.casefold() == team_key)
        | (history["away_team"].astype(str).str.casefold() == team_key)
    ].tail(FORM_WINDOW)

    if team_games.empty:
        return {"offense_residual": 0.0, "defense_residual": 0.0, "games_used": 0}

    league_avg = ratings["league_avg"]
    offense = ratings["offense"]
    defense = ratings["defense_weakness"]
    off_residuals = []
    def_residuals = []

    for row in team_games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_score = float(row.home_score)
        away_score = float(row.away_score)

        if home.casefold() == team_key:
            scored, allowed, opponent = home_score, away_score, away
        elif away.casefold() == team_key:
            scored, allowed, opponent = away_score, home_score, home
        else:
            continue

        expected_for = (
            league_avg + offense.get(team, offense.get(normalize_team_name(team), 0.0))
            + defense.get(opponent, 0.0)
        )
        expected_allowed = (
            league_avg + offense.get(opponent, 0.0)
            + defense.get(team, defense.get(normalize_team_name(team), 0.0))
        )
        off_residuals.append(scored - expected_for)
        def_residuals.append(allowed - expected_allowed)

    n = len(off_residuals)
    if n == 0:
        return {"offense_residual": 0.0, "defense_residual": 0.0, "games_used": 0}

    shrink = n / (n + FORM_PRIOR_GAMES)
    return {
        "offense_residual": float(np.mean(off_residuals)) * shrink,
        "defense_residual": float(np.mean(def_residuals)) * shrink,
        "games_used": n,
    }


def calculate_matchup_points_v3(home_team, away_team, history_df=None, game_date=None):
    """Return V3 raw team and total projections using strictly prior games."""
    history = _clean_history(history_df, game_date)

    if game_date is None:
        if history.empty:
            season_start = None
        else:
            season_start = nba_season_start_year(pd.Timestamp.now().date())
    else:
        season_start = nba_season_start_year(game_date)

    if season_start is None:
        season_games = history.iloc[0:0].copy()
    else:
        season_games = history[
            history["game_date"].map(nba_season_start_year) == season_start
        ].copy()

    ratings = _fit_season_ratings(season_games)
    league_avg = ratings["league_avg"]

    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)
    home_form = _recent_form(season_games, home_key, ratings)
    away_form = _recent_form(season_games, away_key, ratings)

    home_base = (
        league_avg
        + ratings["offense"].get(home_key, 0.0)
        + ratings["defense_weakness"].get(away_key, 0.0)
    )
    away_base = (
        league_avg
        + ratings["offense"].get(away_key, 0.0)
        + ratings["defense_weakness"].get(home_key, 0.0)
    )

    # Recent form is a residual against opponent-adjusted season ratings.
    # This avoids separately adding overlapping raw last-5 and last-10 averages.
    home_points = home_base + FORM_WEIGHT * (
        home_form["offense_residual"] + away_form["defense_residual"]
    )
    away_points = away_base + FORM_WEIGHT * (
        away_form["offense_residual"] + home_form["defense_residual"]
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "projected_home_points": round(float(home_points), 2),
        "projected_away_points": round(float(away_points), 2),
        "raw_total": round(float(home_points + away_points), 2),
        "league_avg_team_points": round(float(league_avg), 2),
        "season_start_year": season_start,
        "season_games_used": int(len(season_games)),
        "home_season_games": int(ratings["games"].get(home_key, 0)),
        "away_season_games": int(ratings["games"].get(away_key, 0)),
        "home_recent_games_used": home_form["games_used"],
        "away_recent_games_used": away_form["games_used"],
        "home_offense_rating": round(float(ratings["offense"].get(home_key, 0.0)), 3),
        "away_offense_rating": round(float(ratings["offense"].get(away_key, 0.0)), 3),
        "home_defense_weakness": round(float(ratings["defense_weakness"].get(home_key, 0.0)), 3),
        "away_defense_weakness": round(float(ratings["defense_weakness"].get(away_key, 0.0)), 3),
        "history_rows": int(len(history)),
        "rating_iterations": int(ratings["iterations"]),
    }
