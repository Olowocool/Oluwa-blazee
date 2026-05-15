import streamlit as st
import requests
import csv
import os
import pandas as pd
from datetime import datetime

API_URL = "https://oluwa-blazee-new.onrender.com"
ODDS_API_KEY = "462ebe76301cb50ce7a9f125c077f9e2"
STAKE = 100


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
    "Los Angeles Clippers": "https://cdn.nba.com/logos/nba/1610612746/global/L/logo.svg",
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
    "Washington Wizards": "https://cdn.nba.com/logos/nba/1610612764/global/L/logo.svg",
}


TEAM_NAME_FIXES = {
    "Philadelphia Sixers": "Philadelphia 76ers",
    "LA Clippers": "Los Angeles Clippers",
}


def normalize_team_name(name):
    fixed = TEAM_NAME_FIXES.get(name, name)
    return fixed.strip()


@st.cache_data(ttl=300)
def load_teams():
    try:
        response = requests.get(f"{API_URL}/teams", timeout=120)

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
        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            st.warning(f"Odds API Error: {response.status_code}")
            return {}

        games = response.json()
        odds_map = {}

        for game in games:
            home_team = normalize_team_name(game["home_team"])
            away_team = normalize_team_name(game["away_team"])

            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])

            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])

            current_odds = {}

            for outcome in outcomes:
                fixed_name = normalize_team_name(outcome["name"])
                current_odds[fixed_name] = outcome["price"]

            odds_map[(home_team, away_team)] = current_odds

        return odds_map

    except Exception as e:
        st.warning(f"Odds fetch failed: {e}")
        return {}


def calculate_ev(model_prob, decimal_odds):
    implied_prob = 1 / decimal_odds
    ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
    return ev, implied_prob
    
def calibrate_probability(probability, strength=0.75, min_prob=0.05, max_prob=0.95):
    probability = max(min(probability, max_prob), min_prob)
    calibrated = 0.5 + ((probability - 0.5) * strength)
    return calibrated

def kelly_fraction(probability, decimal_odds):
    b = decimal_odds - 1
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


def save_bet_pick(game, game_date, best_bet, odds, model_prob, expected_value, kelly):
    file_exists = os.path.isfile("bet_history.csv")

    with open("bet_history.csv", "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "game_date",
                "home_team",
                "away_team",
                "best_bet",
                "odds",
                "model_probability",
                "expected_value",
                "kelly",
                "stake",
                "result",
                "profit_loss"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            game_date,
            game["home_team"],
            game["away_team"],
            best_bet,
            odds,
            model_prob,
            expected_value,
            kelly,
            STAKE,
            "Pending",
            0
        ])


def calculate_profit_loss(row):
    result = str(row.get("result", "Pending")).lower()
    odds = float(row.get("odds", 0))
    stake = float(row.get("stake", STAKE))

    if result == "win":
        return (odds - 1) * stake

    if result == "loss":
        return -stake

    return 0


def load_bet_history():
    if not os.path.isfile("bet_history.csv"):
        return None

    df = pd.read_csv("bet_history.csv")

    if "stake" not in df.columns:
        df["stake"] = STAKE

    if "result" not in df.columns:
        df["result"] = "Pending"

    if "profit_loss" not in df.columns:
        df["profit_loss"] = 0

    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["stake"] = df["stake"].fillna(STAKE)
    df.loc[df["stake"] <= 0, "stake"] = STAKE

    df["result"] = df["result"].fillna("Pending")
    df["profit_loss"] = df.apply(calculate_profit_loss, axis=1)

    return df


def save_bet_history(df):
    df.to_csv("bet_history.csv", index=False)


def auto_grade_bet(row):
    try:
        response = requests.get(
            f"{API_URL}/score_result",
            params={
                "date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "best_bet": row["best_bet"]
            },
            timeout=30
        )

        if response.status_code != 200:
            return row

        data = response.json()

        if data.get("status") == "completed":
            row["result"] = data["result"]
            row["profit_loss"] = calculate_profit_loss({
                "result": data["result"],
                "odds": row["odds"],
                "stake": row["stake"]
            })

        return row

    except Exception:
        return row


teams = load_teams()

if "daily_data" not in st.session_state:
    st.session_state["daily_data"] = None

if "last_loaded_date" not in st.session_state:
    st.session_state["last_loaded_date"] = None


st.title("Today's NBA Games")

date_input = st.text_input(
    "Game Date (MM/DD/YYYY)",
    value="05/15/2026"
)

if st.button("Load Daily Predictions"):
    try:
        response = requests.get(
            f"{API_URL}/predict_today",
            params={"date": date_input},
            timeout=60
        )

        if response.status_code != 200:
            st.error(f"Prediction API failed with status {response.status_code}")
            st.write(response.text)
            st.stop()

        data = response.json()
        st.session_state["daily_data"] = data
        st.session_state["last_loaded_date"] = date_input

    except Exception as e:
        st.error(f"Prediction request failed: {e}")
        st.stop()


data = st.session_state["daily_data"]
active_date = st.session_state["last_loaded_date"] or date_input

if data and "games" in data and len(data["games"]) > 0:
    odds_map = get_odds()

    for game in data["games"]:
        col_logo1, col_text, col_logo2 = st.columns([1, 3, 1])

        with col_logo1:
            away_logo = TEAM_LOGOS.get(game["away_team"])
            if away_logo:
                st.image(away_logo, width=70)

        with col_text:
            st.subheader(f"{game['away_team']} @ {game['home_team']}")

        with col_logo2:
            home_logo = TEAM_LOGOS.get(game["home_team"])
            if home_logo:
                st.image(home_logo, width=70)

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

        st.caption(f"Predicted Winner — {confidence_label}")
        st.header(game["prediction"])
        st.progress(confidence)
        st.info(betting_note)

        save_prediction_log(game, active_date)

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

        st.subheader("Injury Impact")

        injury_col1, injury_col2, injury_col3 = st.columns(3)

        with injury_col1:
            st.metric(
                "Home Penalty",
                game.get("home_injury_penalty", 0)
            )

        with injury_col2:
            st.metric(
                "Away Penalty",
                game.get("away_injury_penalty", 0)
            )

        with injury_col3:
            st.metric(
                "Injury Diff",
                game.get("injury_diff", 0)
            )

        odds = {}

        game_home = normalize_team_name(game["home_team"]).lower()
        game_away = normalize_team_name(game["away_team"]).lower()

        for (home, away), value in odds_map.items():
            odds_home = normalize_team_name(home).lower()
            odds_away = normalize_team_name(away).lower()

            teams_match = sorted([game_home, game_away]) == sorted([odds_home, odds_away])

            if teams_match:
                odds = value
                break

        home_odds = None
        away_odds = None

        for team_name, price in odds.items():
            normalized_team = normalize_team_name(team_name).lower()

            if normalized_team == game_home:
                home_odds = price

            elif normalized_team == game_away:
                away_odds = price

        if home_odds and away_odds:
            st.subheader("Betting Analytics")
            
            calibrated_home_prob = calibrate_probability(game["home_win_probability"])
            calibrated_away_prob = calibrate_probability(game["away_win_probability"])

            home_ev, home_implied = calculate_ev(
                game["home_win_probability"],
                home_odds
            )

            away_ev, away_implied = calculate_ev(
                game["away_win_probability"],
                away_odds
            )

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
                st.metric(f"{game['home_team']} Odds", f"{home_odds:.2f}")
                st.metric("Implied Probability", f"{home_implied * 100:.1f}%")
                st.metric("Expected Value", f"{home_ev * 100:.1f}%")
                st.metric("Kelly %", f"{home_kelly * 100:.1f}%")

            with analytics_col2:
                st.metric(f"{game['away_team']} Odds", f"{away_odds:.2f}")
                st.metric("Implied Probability", f"{away_implied * 100:.1f}%")
                st.metric("Expected Value", f"{away_ev * 100:.1f}%")
                st.metric("Kelly %", f"{away_kelly * 100:.1f}%")

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

                if best_bet == game["home_team"]:
                    selected_odds = home_odds
                    selected_prob = game["home_win_probability"]
                    selected_kelly = home_kelly
                else:
                    selected_odds = away_odds
                    selected_prob = game["away_win_probability"]
                    selected_kelly = away_kelly

                button_key = f"button_{game['home_team']}_{game['away_team']}_{best_bet}"
                saved_key = f"saved_{game['home_team']}_{game['away_team']}_{best_bet}"

                if saved_key not in st.session_state:
                    st.session_state[saved_key] = False

                if st.button(
                    f"Save Bet Pick: {best_bet}",
                    key=button_key
                ):
                    save_bet_pick(
                        game,
                        active_date,
                        best_bet,
                        selected_odds,
                        selected_prob,
                        best_ev,
                        selected_kelly
                    )

                    st.session_state[saved_key] = True

                if st.session_state[saved_key]:
                    st.success("Bet pick saved successfully!")

            else:
                st.warning("No strong value bet detected.")

        else:
            st.warning("No sportsbook odds found for this matchup.")

            with st.expander("Debug odds matching"):
                st.write("Prediction matchup:")
                st.write(game["away_team"], "@", game["home_team"])

                st.write("Available Odds API matchups:")
                st.write(list(odds_map.keys())[:20])

        st.divider()

elif data:
    st.warning("No games returned from API.")


st.title("Bet Performance Dashboard")

bet_history = load_bet_history()

if bet_history is None or bet_history.empty:
    st.info("No saved bet picks yet.")

else:
    bet_history["expected_value"] = pd.to_numeric(
        bet_history["expected_value"],
        errors="coerce"
    )

    bet_history["kelly"] = pd.to_numeric(
        bet_history["kelly"],
        errors="coerce"
    )

    bet_history["profit_loss"] = pd.to_numeric(
        bet_history["profit_loss"],
        errors="coerce"
    )

    updated_df = bet_history.copy()

    updated_df = updated_df.apply(auto_grade_bet, axis=1)
    save_bet_history(updated_df)

    total_bets = len(updated_df)
    wins = len(updated_df[updated_df["result"].str.lower() == "win"])
    losses = len(updated_df[updated_df["result"].str.lower() == "loss"])
    settled_bets = wins + losses

    win_rate = (wins / settled_bets * 100) if settled_bets > 0 else 0
    total_profit = updated_df["profit_loss"].sum()
    total_staked = settled_bets * STAKE
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    avg_ev = updated_df["expected_value"].mean() * 100
    avg_kelly = updated_df["kelly"].mean() * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Picks", total_bets)
        st.metric("Wins", wins)

    with col2:
        st.metric("Losses", losses)
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col3:
        st.metric("Profit/Loss", f"${total_profit:.2f}")
        st.metric("ROI", f"{roi:.1f}%")

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Average EV", f"{avg_ev:.1f}%")

    with col5:
        st.metric("Average Kelly", f"{avg_kelly:.1f}%")

    st.subheader("Update Bet Results")

    for index, row in updated_df.iterrows():
        current_result = row["result"]

        if current_result not in ["Pending", "Win", "Loss"]:
            current_result = "Pending"

        st.write(
            f"{row['game_date']} — {row['best_bet']} "
            f"({row['away_team']} @ {row['home_team']})"
        )

        updated_df.at[index, "result"] = st.selectbox(
            "Result",
            ["Pending", "Win", "Loss"],
            index=["Pending", "Win", "Loss"].index(current_result),
            key=f"result_{index}"
        )

    if st.button("Save Updated Results"):
        updated_df["profit_loss"] = updated_df.apply(calculate_profit_loss, axis=1)
        save_bet_history(updated_df)
        st.success("Bet results updated successfully!")

    st.subheader("Bankroll Growth")

    chart_df = updated_df.copy()

    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
    chart_df = chart_df.sort_values("timestamp")
    chart_df["cumulative_profit"] = chart_df["profit_loss"].cumsum()

    st.line_chart(
        chart_df.set_index("timestamp")["cumulative_profit"]
    )

    st.subheader("Saved Bet Picks")

    def highlight_results(row):
        result = str(row["result"]).lower()

        if result == "win":
            return ["background-color: #d4edda"] * len(row)

        elif result == "loss":
            return ["background-color: #f8d7da"] * len(row)

        return ["background-color: #fff3cd"] * len(row)

    styled_df = updated_df.style.apply(
        highlight_results,
        axis=1
    )

    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )
