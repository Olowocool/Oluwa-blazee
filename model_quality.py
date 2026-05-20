import pandas as pd


def safe_mean(values, default=0):
    try:
        values = pd.Series(values).dropna()
        if values.empty:
            return default
        return float(values.mean())
    except Exception:
        return default


def get_team_score(game, team_name):
    if game.get("home_team_name") == team_name:
        return game.get("home_pts", game.get("home_score", 0)), game.get("away_pts", game.get("away_score", 0))

    if game.get("away_team_name") == team_name:
        return game.get("away_pts", game.get("away_score", 0)), game.get("home_pts", game.get("home_score", 0))

    return 0, 0


def calculate_recent_form(team_games, team_name=None, last_n=10):
    recent = team_games.sort_values("date").tail(last_n)

    if recent.empty:
        return {
            "recent_win_rate": 0.5,
            "recent_points": 100,
            "recent_allowed": 100,
            "recent_margin": 0
        }

    if team_name is None:
        first_row = recent.iloc[-1]
        team_name = first_row.get("home_team_name", "")

    wins = []
    scored = []
    allowed = []

    for _, game in recent.iterrows():
        team_score, opp_score = get_team_score(game, team_name)

        scored.append(team_score)
        allowed.append(opp_score)
        wins.append(1 if team_score > opp_score else 0)

    return {
        "recent_win_rate": safe_mean(wins, 0.5),
        "recent_points": safe_mean(scored, 100),
        "recent_allowed": safe_mean(allowed, 100),
        "recent_margin": safe_mean(scored, 100) - safe_mean(allowed, 100)
    }


def calculate_home_away_strength(team_games, team_name):
    home_games = team_games[team_games["home_team_name"] == team_name]
    away_games = team_games[team_games["away_team_name"] == team_name]

    home_wins = []
    away_wins = []

    for _, game in home_games.iterrows():
        home_wins.append(1 if game.get("home_pts", 0) > game.get("away_pts", 0) else 0)

    for _, game in away_games.iterrows():
        away_wins.append(1 if game.get("away_pts", 0) > game.get("home_pts", 0) else 0)

    return {
        "home_strength": safe_mean(home_wins, 0.5),
        "away_strength": safe_mean(away_wins, 0.5)
    }


def calculate_rest_days(team_games):
    recent_games = team_games.sort_values("date").tail(2)

    if len(recent_games) < 2:
        return 2

    try:
        dates = pd.to_datetime(recent_games["date"]).reset_index(drop=True)
        return max((dates.iloc[1] - dates.iloc[0]).days, 0)
    except Exception:
        return 2


def fatigue_penalty(rest_days):
    if rest_days <= 0:
        return -0.04
    if rest_days == 1:
        return -0.02
    if rest_days >= 3:
        return 0.015
    return 0


def calibrate_probability(prob):
    prob = max(0.05, min(0.95, float(prob)))
    return max(0.05, min(0.95, 0.5 + ((prob - 0.5) * 0.82)))


def quality_adjust_probability(
    raw_prob,
    home_recent_form,
    away_recent_form,
    home_rest_days,
    away_rest_days,
    injury_adjustment
):
    prob = float(raw_prob)

    recent_form_edge = home_recent_form["recent_win_rate"] - away_recent_form["recent_win_rate"]
    recent_margin_edge = (home_recent_form["recent_margin"] - away_recent_form["recent_margin"]) * 0.002
    rest_edge = fatigue_penalty(home_rest_days) - fatigue_penalty(away_rest_days)

    prob += recent_form_edge * 0.08
    prob += recent_margin_edge
    prob += rest_edge
    prob += injury_adjustment

    return calibrate_probability(prob)
