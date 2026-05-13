import streamlit as st
import requests
import csv
import os
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

API_URL = "https://oluwa-blazee.onrender.com"

ODDS_API_KEY = "462ebe76301cb50ce7a9f125c077f9e2"

# =========================================================
# TEAM LOGOS
# =========================================================

TEAM_LOGOS = {
    "Atlanta Hawks": "https://cdn.nba.com/logos/nba/1610612737/global/L/logo.svg",
    "Boston Celtics": "https://cdn.nba.com/logos/nba/1610612738/global/L/logo.svg",
    "Brooklyn Nets": "https://cdn.nba.com/logos/nba/1610612751/global/L/logo.svg",
    "Charlotte Hornets": "https://cdn.nba.com/logos/nba/1610612766/global/L/logo.svg",
    "Chicago Bulls": "https://cdn.nba.com/logos/nba/1610612741/global/L/logo.svg",
    "Cleveland Cavaliers": "https://cdn.nba.com/logos/nba/1610612739/global/L/logo.svg",
    "Dallas Mavericks": "https://cdn.nba.com/logos/nba/1610612742/global/L/logo.svg",
    "Denver Nuggets": "https://cdn.nba.com/logos/nba/1610612743/global/L/logo.svg",
    "Detroit Pistons": "https://cdn.nba.com/logos/nba/1610612765/global/L/logo.svg",
    "Golden State Warriors": "https://cdn.nba.com/logos/nba/1610612744/global/L/logo.svg",
    "Houston Rockets": "https://cdn.nba.com/logos/nba/1610612745/global/L/logo.svg",
    "Indiana Pacers": "https://cdn.nba.com/logos/nba/1610612754/global/L/logo.svg",
    "LA Clippers": "https://cdn.nba.com/logos/nba/1610612746/global/L/logo.svg",
    "Los Angeles Lakers": "https://cdn.nba.com/logos/nba/1610612747/global/L/logo.svg",
    "Memphis Grizzlies": "https://cdn.nba.com/logos/nba/1610612763/global/L/logo.svg",
    "Miami Heat": "https://cdn.nba.com/logos/nba/1610612748/global/L/logo.svg",
    "Milwaukee Bucks": "https://cdn.nba.com/logos/nba/1610612749/global/L/logo.svg",
    "Minnesota Timberwolves": "https://cdn.nba.com/logos/nba/1610612750/global/L/logo.svg",
    "New Orleans Pelicans": "https://cdn.nba.com/logos/nba/1610612740/global/L/logo.svg",
    "New York Knicks": "https://cdn.nba.com/logos/nba/1610612752/global/L/logo.svg",
    "Oklahoma City Thunder": "https://cdn.nba.com/logos/nba/1610612760/global/L/logo.svg",
    "Orlando Magic": "https://cdn.nba.com/logos/nba/1610612753/global/L/logo.svg",
    "Philadelphia 76ers": "https://cdn.nba.com/logos/nba/1610612755/global/L/logo.svg",
    "Phoenix Suns": "https://cdn.nba.com/logos/nba/1610612756/global/L/logo.svg",
    "Portland Trail Blazers": "https://cdn.nba.com/logos/nba/1610612757/global/L/logo.svg",
    "Sacramento Kings": "https://cdn.nba.com/logos/nba/1610612758/global/L/logo.svg",
    "San Antonio Spurs": "https://cdn.nba.com/logos/nba/1610612759/global/L/logo.svg",
    "Toronto Raptors": "https://cdn.nba.com/logos/nba/1610612761/global/L/logo.svg",
    "Utah Jazz": "https://cdn.nba.com/logos/nba/1610612762/global/L/logo.svg",
    "Washington Wizards": "https://cdn.nba.com/logos/nba/1610612764/global/L/logo.svg"
}

# =========================================================
# HELPERS
# =========================================================

@st.cache_data(ttl=300)
def load_teams():

    try:
        response = requests.get(
            f"{API_URL}/teams",
            timeout=60
        )

        if response.status_code != 200:
            st.error("Failed to load teams.")
            return []

        return response.json()["teams"]

    except Exception as e:
        st.error(f"Backend connection error: {e}")
        return []


@st.cache_data(ttl=300)
def get_odds():

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            st.error(f"Odds API Error: {response.status_code}")
            return {}

        games = response.json()

        odds_map = {}

        for game in games:

            home_team = game["home_team"]
            away_team = game["away_team"]

            bookmakers = game.get("bookmakers", [])

            if bookmakers:

                markets = bookmakers[0].get("markets", [])

                if markets:

                    outcomes = markets[0].get("outcomes", [])

                    current_odds = {}

                    for outcome in outcomes:
                        current_odds[outcome["name"]] = outcome["price"]

                    odds_map[(home_team, away_team)] = current_odds

        return odds_map

    except Exception as e:
        st.error(f"Odds fetch failed: {e}")
        return {}


def calculate_ev(model_prob, decimal_odds):

    implied_prob = 1 / decimal_odds

    ev = (
        (model_prob * (decimal_odds - 1))
        - (1 - model_prob)
    )

    return ev, implied_prob


def kelly_fraction(probability, decimal_odds):

    b = decimal_odds - 1
    q = 1 - probability

    kelly = ((b * probability) - q) / b

    return max(kelly, 0)


def save_prediction_log(game, game_date):

    file_exists = os.path.isfile("prediction_history.csv")

    with open("prediction_history.csv", "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "game_date",
                "home_team",
                "away_team",
                "prediction",
                "home_probability",
                "away_probability"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            game_date,
            game["home_team"],
            game["away_team"],
            game["prediction"],
            game["home_win_probability"],
            game["away_win_probability"]
        ])

# =========================================================
# LOAD DATA
# =========================================================

teams = load_teams()
odds_map = get_odds()

# =========================================================
# UI
# =========================================================

st.title("Today's NBA Games")

date_input = st.text_input(
    "Game Date (MM/DD/YYYY)",
    value="12/25/2025",
    key="daily_games_date"
)

# =========================================================
# DAILY PREDICTIONS
# =========================================================

if st.button("Load Daily Predictions"):

    try:

        response = requests.get(
            f"{API_URL}/predict_today",
            params={"date": date_input},
            timeout=30
        )

        if response.status_code != 200:
            st.error("Prediction API failed.")
            st.stop()

        data = response.json()

    except Exception as e:

        st.error(f"Prediction request failed: {e}")
        st.stop()

    if "games" in data and len(data["games"]) > 0:

        for game in data["games"]:

            # =====================================================
            # HEADER
            # =====================================================

            col_logo1, col_text, col_logo2 = st.columns([1, 3, 1])

            with col_logo1:

                away_logo = TEAM_LOGOS.get(game["away_team"])

                if away_logo:
                    st.image(away_logo, width=70)

            with col_text:

                st.subheader(
                    f"{game['away_team']} @ {game['home_team']}"
                )

            with col_logo2:

                home_logo = TEAM_LOGOS.get(game["home_team"])

                if home_logo:
                    st.image(home_logo, width=70)

            # =====================================================
            # CONFIDENCE
            # =====================================================

            confidence = max(
                game["home_win_probability"],
                game["away_win_probability"]
            )

            if confidence >= 0.70:

                confidence_label = "Strong Favorite"
                betting_note = "High-confidence model pick"

            elif confidence >= 0.60:

                confidence_label = "Lean"
                betting_note = "Moderate model edge"

            elif confidence >= 0.55:

                confidence_label = "Slight Lean"
                betting_note = "Small edge, use caution"

            else:

                confidence_label = "Avoid"
                betting_note = "Too close to call"

            # =====================================================
            # PREDICTION
            # =====================================================

            st.caption(
                f"Predicted Winner — {confidence_label}"
            )

            st.header(game["prediction"])

            st.progress(confidence)

            st.info(betting_note)

            save_prediction_log(game, date_input)

            # =====================================================
            # PROBABILITY METRICS
            # =====================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label=game["home_team"],
                    value=f"{game['home_win_probability'] * 100:.1f}%"
                )

            with col2:

                st.metric(
                    label=game["away_team"],
                    value=f"{game['away_win_probability'] * 100:.1f}%"
                )

            # =====================================================
            # ODDS + EV
            # =====================================================

            odds = {}

            for (home, away), value in odds_map.items():

                if (
                    home.lower() == game["home_team"].lower()
                    and away.lower() == game["away_team"].lower()
                ):

                    odds = value
                    break

            home_odds = odds.get(game["home_team"])
            away_odds = odds.get(game["away_team"])

            if home_odds and away_odds:

                st.subheader("Betting Analytics")

                home_ev, home_implied = calculate_ev(
                    game["home_win_probability"],
                    home_odds
                )

                away_ev, away_implied = calculate_ev(
                    game["away_win_probability"],
                    away_odds
                )

                best_bet = None
                best_ev = 0
                
                if home_ev > away_ev and home_ev > 0.05:
                    best_bet = game["home_team"]
                    best_ev = home_ev
                
                elif away_ev > home_ev and away_ev > 0.05:
                    best_bet = game["away_team"]
                    best_ev = away_ev
                
                if best_bet:
                    st.success(
                        f"🔥 BEST BET: {best_bet} | Expected Value: {best_ev * 100:.1f}%"
                    )
                else:
                    st.warning("No strong value bet detected.")

                home_kelly = kelly_fraction(
                    game["home_win_probability"],
                    home_odds
                )

                away_kelly = kelly_fraction(
                    game["away_win_probability"],
                    away_odds
                )

                analytics_col1, analytics_col2 = st.columns(2)

                with analytics_col1:

                    st.metric(
                        f"{game['home_team']} Odds",
                        f"{home_odds}"
                    )

                    st.metric(
                        "Expected Value",
                        f"{home_ev:.3f}"
                    )

                    st.metric(
                        "Kelly %",
                        f"{home_kelly * 100:.1f}%"
                    )

                with analytics_col2:

                    st.metric(
                        f"{game['away_team']} Odds",
                        f"{away_odds}"
                    )

                    st.metric(
                        "Expected Value",
                        f"{away_ev:.3f}"
                    )

                    st.metric(
                        "Kelly %",
                        f"{away_kelly * 100:.1f}%"
                    )

            else:

                st.warning(
                    "No sportsbook odds found for this matchup."
                )

            st.divider()

    else:

        st.warning("No games returned from API.")

# =========================================================
# MATCHUP PREDICTION
# =========================================================

st.title("Single Matchup Prediction")

home_team = st.selectbox(
    "Home Team",
    teams,
    key="home_team"
)

away_team = st.selectbox(
    "Away Team",
    teams,
    key="away_team"
)

if st.button("Predict Matchup"):

    try:

        response = requests.post(
            f"{API_URL}/predict_matchup",
            json={
                "home_team": home_team,
                "away_team": away_team
            },
            timeout=30
        )

        if response.status_code != 200:
            st.error("Matchup prediction failed.")
            st.stop()

        result = response.json()

    except Exception as e:

        st.error(f"Prediction error: {e}")
        st.stop()

    col_logo1, col_text, col_logo2 = st.columns([1, 3, 1])

    with col_logo1:

        away_logo = TEAM_LOGOS.get(away_team)

        if away_logo:
            st.image(away_logo, width=70)

    with col_text:

        st.subheader(
            f"{away_team} @ {home_team}"
        )

    with col_logo2:

        home_logo = TEAM_LOGOS.get(home_team)

        if home_logo:
            st.image(home_logo, width=70)

    confidence = max(
        result["home_win_probability"],
        result["away_win_probability"]
    )

    st.header(result["prediction"])

    st.progress(confidence)

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            home_team,
            f"{result['home_win_probability'] * 100:.1f}%"
        )

    with metric_col2:

        st.metric(
            away_team,
            f"{result['away_win_probability'] * 100:.1f}%"
        )
