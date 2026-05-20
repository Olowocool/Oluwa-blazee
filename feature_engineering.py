import pandas as pd
import numpy as np


def safe_float(value, default=0):
    try:
        return float(value)
    except Exception:
        return default


def normalize_probability(prob):
    prob = safe_float(prob)

    if prob < 0:
        return 0

    if prob > 1:
        return 1

    return prob


# =========================
# REST / FATIGUE
# =========================

def calculate_rest_advantage(
    home_rest_days,
    away_rest_days
):

    home_rest_days = safe_float(home_rest_days)
    away_rest_days = safe_float(away_rest_days)

    return home_rest_days - away_rest_days


def fatigue_penalty(rest_days):

    rest_days = safe_float(rest_days)

    if rest_days <= 0:
        return 0.08

    if rest_days == 1:
        return 0.04

    return 0


# =========================
# HOME / AWAY SPLITS
# =========================

def home_court_adjustment(
    home_win_pct,
    away_win_pct
):

    home_win_pct = normalize_probability(
        home_win_pct
    )

    away_win_pct = normalize_probability(
        away_win_pct
    )

    return (
        home_win_pct - away_win_pct
    )


# =========================
# RECENT FORM
# =========================

def recent_form_score(
    recent_wins,
    recent_games=10
):

    recent_wins = safe_float(
        recent_wins
    )

    recent_games = safe_float(
        recent_games
    )

    if recent_games <= 0:
        return 0

    return recent_wins / recent_games


# =========================
# OFFENSIVE / DEFENSIVE
# =========================

def net_rating(
    offensive_rating,
    defensive_rating
):

    offensive_rating = safe_float(
        offensive_rating
    )

    defensive_rating = safe_float(
        defensive_rating
    )

    return (
        offensive_rating
        -
        defensive_rating
    )


# =========================
# PACE FACTOR
# =========================

def pace_factor(team_pace):

    team_pace = safe_float(team_pace)

    league_average = 100

    return (
        team_pace - league_average
    ) / 100


# =========================
# TRAVEL FATIGUE
# =========================

def travel_fatigue_penalty(
    travel_distance_km
):

    travel_distance_km = safe_float(
        travel_distance_km
    )

    if travel_distance_km >= 3000:
        return 0.05

    if travel_distance_km >= 1500:
        return 0.03

    return 0


# =========================
# MARKET MOVEMENT
# =========================

def market_movement_strength(
    line_move_pct
):

    line_move_pct = safe_float(
        line_move_pct
    )

    return abs(line_move_pct)


# =========================
# SHARP CONSENSUS
# =========================

def sharp_consensus_signal(
    sharp_books_support,
    total_books
):

    sharp_books_support = safe_float(
        sharp_books_support
    )

    total_books = safe_float(
        total_books
    )

    if total_books <= 0:
        return 0

    return (
        sharp_books_support
        /
        total_books
    )


# =========================
# MASTER FEATURE ENGINE
# =========================

def build_feature_vector(data):

    features = {}

    # Base probabilities

    features["home_probability"] = normalize_probability(
        data.get("home_probability", 0.5)
    )

    features["away_probability"] = normalize_probability(
        data.get("away_probability", 0.5)
    )

    features["model_confidence"] = max(
        features["home_probability"],
        features["away_probability"]
    )

    # Rest / fatigue

    features["rest_advantage"] = calculate_rest_advantage(
        data.get("home_rest_days", 2),
        data.get("away_rest_days", 2)
    )

    features["home_fatigue_penalty"] = fatigue_penalty(
        data.get("home_rest_days", 2)
    )

    features["away_fatigue_penalty"] = fatigue_penalty(
        data.get("away_rest_days", 2)
    )

    # Home / away

    features["home_court_edge"] = home_court_adjustment(
        data.get("home_home_win_pct", 0.5),
        data.get("away_away_win_pct", 0.5)
    )

    # Recent form

    features["home_recent_form"] = recent_form_score(
        data.get("home_recent_wins", 5)
    )

    features["away_recent_form"] = recent_form_score(
        data.get("away_recent_wins", 5)
    )

    # Net ratings

    features["home_net_rating"] = net_rating(
        data.get("home_off_rating", 110),
        data.get("home_def_rating", 110)
    )

    features["away_net_rating"] = net_rating(
        data.get("away_off_rating", 110),
        data.get("away_def_rating", 110)
    )

    # Pace

    features["home_pace_factor"] = pace_factor(
        data.get("home_pace", 100)
    )

    features["away_pace_factor"] = pace_factor(
        data.get("away_pace", 100)
    )

    # Travel

    features["home_travel_penalty"] = travel_fatigue_penalty(
        data.get("home_travel_km", 0)
    )

    features["away_travel_penalty"] = travel_fatigue_penalty(
        data.get("away_travel_km", 0)
    )

    # Market movement

    features["home_market_strength"] = market_movement_strength(
        data.get("home_line_move_pct", 0)
    )

    features["away_market_strength"] = market_movement_strength(
        data.get("away_line_move_pct", 0)
    )

    # Sharp consensus

    features["sharp_support"] = sharp_consensus_signal(
        data.get("sharp_books_support", 0),
        data.get("total_books", 1)
    )

    return features
