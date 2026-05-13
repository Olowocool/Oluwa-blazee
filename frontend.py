import streamlit as st
import requests
from datetime import datetime
import csv
import os

API_URL = "https://oluwa-blazee.onrender.com"
ODDS_API_KEY = "462ebe76301cb50ce7a9f125c077f9e2"

TEAM_LOGOS = {
    "Boston Celtics": "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg",
    "Los Angeles Lakers": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",
    "New York Knicks": "https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg",
    "Cleveland Cavaliers": "https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg",
    "Golden State Warriors": "https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg",
    "Dallas Mavericks": "https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg",
    "Oklahoma City Thunder": "https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg",
    "San Antonio Spurs": "https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg",
}


def get_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"Odds API Error: {response.status_code}")
        return {}

    games = response.json()
    odds_map = {}

    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        if not game.get("bookmakers"):
            continue

        bookmaker = game["bookmakers"][0]

        if not bookmaker.get("markets"):
            continue

        outcomes = bookmaker["markets"][0]["outcomes"]
        odds = {}

        for outcome in outcomes:
            odds[outcome["name"]] = outcome["price"]

        odds_map[(home_team, away_team)] = odds

    return odds_map

def save_prediction_log(game):
    file_exists = os.path.isfile("prediction_history.csv")

    with open("prediction_history.csv", "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "home_team",
                "away_team",
                "prediction",
                "home_probability",
                "away_probability"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            game["home_team"],
            game["away_team"],
            game["prediction"],
            game["home_win_probability"],
            game["away_win_probability"]
        ])
        
def calculate_ev(model_prob, decimal_odds):
    implied_prob = 1 / decimal_odds
    ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)

    return {
        "implied_probability": implied_prob,
        "expected_value": ev
    }


def kelly_fraction(probability, odds):
    b = odds - 1
    q = 1 - probability

    if b <= 0:
        return 0

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

st.title("NBA Prediction Dashboard")

teams_response = requests.get(f"{API_URL}/teams")
teams = teams_response.json()["teams"]

home_team = st.selectbox(
    "Home Team",
    teams,
    index=teams.index("Boston Celtics")
)

away_team = st.selectbox(
    "Away Team",
    teams,
    index=teams.index("Los Angeles Lakers")
)

if st.button("Predict Matchup"):
    response = requests.post(
        f"{API_URL}/predict_matchup",
        json={
            "home_team": home_team,
            "away_team": away_team
        }
    )

    result = response.json()

    st.subheader("Prediction Result")
    st.metric("Predicted Winner", result["prediction"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            result["home_team"],
            f"{result['home_win_probability'] * 100:.1f}%"
        )

    with col2:
        st.metric(
            result["away_team"],
            f"{result['away_win_probability'] * 100:.1f}%"
        )

st.divider()

st.header("Today's NBA Games")

date_input = st.text_input(
    "Game Date (MM/DD/YYYY)",
    datetime.today().strftime("%m/%d/%Y"),
    key="daily_games_date"
)

if st.button("Load Daily Predictions"):
    response = requests.get(
        f"{API_URL}/predict_today",
        params={"date": date_input}
    )

    data = response.json()
    odds_map = get_odds()

    if "games" in data and len(data["games"]) > 0:
        for game in data["games"]:
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

            confidence = max(
                game["home_win_probability"],
                game["away_win_probability"]
            )

            odds = {}

            for (home, away), value in odds_map.items():
                home_match = home.lower() == game["home_team"].lower()
                away_match = away.lower() == game["away_team"].lower()

                if (
                    game["home_team"].lower() in home.lower()
                    and game["away_team"].lower() in away.lower()
                ):
                    odds = value
                    break

            home_odds = odds.get(game["home_team"])
            away_odds = odds.get(game["away_team"])

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

            st.metric(
                label=f"Predicted Winner — {confidence_label}",
                value=game["prediction"]
            )

            st.progress(confidence)
            st.info(betting_note)
            save_prediction_log(game, date_input)
            save_prediction_log(game)

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

            if home_odds and away_odds:
                implied_home = 1 / home_odds
                implied_away = 1 / away_odds

                model_home = game["home_win_probability"]
                model_away = game["away_win_probability"]

                home_edge = model_home - implied_home
                away_edge = model_away - implied_away

                home_ev_data = calculate_ev(model_home, home_odds)
                away_ev_data = calculate_ev(model_away, away_odds)

                home_kelly = kelly_fraction(model_home, home_odds)
                away_kelly = kelly_fraction(model_away, away_odds)

                st.markdown("### Sportsbook Odds")

                col_odds1, col_odds2 = st.columns(2)

                with col_odds1:
                    st.metric(
                        "Home Odds",
                        f"{home_odds:.2f}"
                    )

                    st.metric(
                        "Home Implied Probability",
                        f"{implied_home * 100:.1f}%"
                    )

                    st.metric(
                        "Home Model Edge",
                        f"{home_edge * 100:.1f}%"
                    )

                with col_odds2:
                    st.metric(
                        "Away Odds",
                        f"{away_odds:.2f}"
                    )

                    st.metric(
                        "Away Implied Probability",
                        f"{implied_away * 100:.1f}%"
                    )

                    st.metric(
                        "Away Model Edge",
                        f"{away_edge * 100:.1f}%"
                    )

                st.markdown("### Betting Analytics")

                bet_col1, bet_col2 = st.columns(2)

                with bet_col1:
                    st.metric(
                        "Home EV",
                        f"{home_ev_data['expected_value'] * 100:.1f}%"
                    )

                    st.metric(
                        "Home Kelly",
                        f"{home_kelly * 100:.1f}%"
                    )

                with bet_col2:
                    st.metric(
                        "Away EV",
                        f"{away_ev_data['expected_value'] * 100:.1f}%"
                    )

                    st.metric(
                        "Away Kelly",
                        f"{away_kelly * 100:.1f}%"
                    )

                if home_ev_data["expected_value"] > 0.05:
                    st.success(
                        f"Value bet detected on {game['home_team']}"
                    )
                elif away_ev_data["expected_value"] > 0.05:
                    st.success(
                        f"Value bet detected on {game['away_team']}"
                    )
                else:
                    st.warning(
                        "No positive EV bet found"
                    )

            else:
                st.warning(
                    "No sportsbook odds found for this matchup."
                )

            st.divider()

    else:
        st.warning("No games returned from API.")
