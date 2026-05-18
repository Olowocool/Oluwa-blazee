import streamlit as st
import requests
import csv
import os
import pandas as pd
from datetime import date, datetime

API_URL = "https://oluwa-blazee-new.onrender.com"
STAKE = 100


def load_odds_api_key():
    env_key = os.getenv("ODDS_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""


ODDS_API_KEY = load_odds_api_key()


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
    return TEAM_NAME_FIXES.get(str(name).strip(), str(name).strip())


def parse_game_date(date_text):
    try:
        return datetime.strptime(date_text, "%m/%d/%Y").date()
    except ValueError:
        return None


def should_fetch_live_odds(date_text):
    game_date = parse_game_date(date_text)
    if game_date is None:
        return True

    return game_date >= date.today()


@st.cache_data(ttl=300)
def get_odds():
    if not ODDS_API_KEY:
        st.warning("Odds API key is not configured.")
        return {}

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
            home_team = normalize_team_name(game["home_team"]).lower()
            away_team = normalize_team_name(game["away_team"]).lower()

            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            current_odds = {}

            for bookmaker in bookmakers:
                bookmaker_name = bookmaker.get("title", "Unknown Sportsbook")
                markets = bookmaker.get("markets", [])

                if not markets:
                    continue

                outcomes = markets[0].get("outcomes", [])

                for outcome in outcomes:
                    fixed_name = normalize_team_name(outcome["name"]).lower()
                    price = float(outcome["price"])

                    if fixed_name not in current_odds:
                        current_odds[fixed_name] = {
                            "price": price,
                            "bookmaker": bookmaker_name
                        }
                    elif price > current_odds[fixed_name]["price"]:
                        current_odds[fixed_name] = {
                            "price": price,
                            "bookmaker": bookmaker_name
                        }

            odds_map[(home_team, away_team)] = current_odds

        return odds_map

    except Exception as e:
        st.warning(f"Odds fetch failed: {e}")
        return {}


def get_historical_odds(game_date):
    if not os.path.isfile("historical_odds.csv"):
        return {}

    try:
        df = pd.read_csv("historical_odds.csv")
    except Exception as e:
        st.warning(f"Historical odds file error: {e}")
        return {}
        
def save_live_odds_to_history(game_date, odds_map):
    rows = []

    for (home_team, away_team), odds in odds_map.items():
        home_data = odds.get(home_team)
        away_data = odds.get(away_team)

        if not home_data or not away_data:
            continue

        rows.append({
            "game_date": game_date,
            "home_team": home_team.title(),
            "away_team": away_team.title(),
            "home_odds": home_data["price"],
            "away_odds": away_data["price"]
        })

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if os.path.isfile("historical_odds.csv"):
        old_df = pd.read_csv("historical_odds.csv")
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=["game_date", "home_team", "away_team"],
            keep="last"
        )
    else:
        final_df = new_df

    final_df.to_csv("historical_odds.csv", index=False)
    
    required_cols = ["game_date", "home_team", "away_team", "home_odds", "away_odds"]

    for col in required_cols:
        if col not in df.columns:
            st.warning(f"historical_odds.csv missing column: {col}")
            return {}

    df["game_date"] = df["game_date"].astype(str).str.strip()
    filtered = df[df["game_date"] == str(game_date).strip()]

    odds_map = {}

    for _, row in filtered.iterrows():
        home_team = normalize_team_name(row["home_team"]).lower()
        away_team = normalize_team_name(row["away_team"]).lower()

        odds_map[(home_team, away_team)] = {
            home_team: {
                "price": float(row["home_odds"]),
                "bookmaker": "Historical Odds"
            },
            away_team: {
                "price": float(row["away_odds"]),
                "bookmaker": "Historical Odds"
            }
        }

    return odds_map


def calculate_ev(model_prob, decimal_odds):
    implied_prob = 1 / decimal_odds
    ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
    return ev, implied_prob


def calculate_model_edge(model_prob, implied_prob):
    return model_prob - implied_prob


def classify_edge(edge):
    if edge >= 0.10:
        return "Elite Edge"
    if edge >= 0.06:
        return "Strong Edge"
    if edge >= 0.03:
        return "Playable Edge"
    if edge > 0:
        return "Small Edge"
    return "No Edge"


def calibrate_probability(probability, strength=0.75, min_prob=0.05, max_prob=0.95):
    probability = max(min(probability, max_prob), min_prob)
    return 0.5 + ((probability - 0.5) * strength)


def kelly_fraction(probability, decimal_odds):
    b = decimal_odds - 1
    q = 1 - probability

    if b <= 0:
        return 0

    kelly = ((b * probability) - q) / b
    return max(kelly, 0)


def calculate_profit_loss(row):
    result = str(row.get("result", "Pending")).lower()
    odds = float(row.get("odds", 0))
    stake = float(row.get("stake", STAKE))

    if result == "win":
        return (odds - 1) * stake

    if result == "loss":
        return -stake

    return 0


def calculate_clv(saved_odds, closing_odds):
    try:
        saved_odds = float(saved_odds)
        closing_odds = float(closing_odds)

        if saved_odds <= 0 or closing_odds <= 0:
            return ""

        clv = (saved_odds / closing_odds) - 1
        return round(clv, 4)

    except Exception:
        return ""


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
                "profit_loss",
                "closing_odds",
                "clv"
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
            0,
            "",
            ""
        ])


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

    if "closing_odds" not in df.columns:
        df["closing_odds"] = ""

    if "clv" not in df.columns:
        df["clv"] = ""

    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["stake"] = df["stake"].fillna(STAKE)
    df.loc[df["stake"] <= 0, "stake"] = STAKE

    df["result"] = df["result"].fillna("Pending")
    df["profit_loss"] = df.apply(calculate_profit_loss, axis=1)

    df["clv"] = df.apply(
        lambda row: calculate_clv(row["odds"], row["closing_odds"]),
        axis=1
    )

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


if "daily_data" not in st.session_state:
    st.session_state["daily_data"] = None

if "last_loaded_date" not in st.session_state:
    st.session_state["last_loaded_date"] = None


st.title("NBA Games")

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
live_odds_mode = should_fetch_live_odds(active_date)

if data and "games" in data and len(data["games"]) > 0:
    if live_odds_mode:
        odds_map = get_odds()
    
        if odds_map:
            save_live_odds_to_history(active_date, odds_map)
            st.success("Live odds saved into historical odds file.")
    else:
        odds_map = get_historical_odds(active_date)

        if odds_map:
            st.success("Historical odds loaded.")
        else:
            st.info("No stored historical odds found for this date.")

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

        if confidence >= 0.75:
            confidence_label = "Elite"
            betting_note = "Strong model position"
            confidence_color = "green"
        elif confidence >= 0.65:
            confidence_label = "Good"
            betting_note = "Moderate confidence"
            confidence_color = "orange"
        elif confidence >= 0.55:
            confidence_label = "Risky"
            betting_note = "Weak betting profile"
            confidence_color = "red"
        else:
            confidence_label = "Avoid"
            betting_note = "No predictive edge"
            confidence_color = "red"

        st.markdown(
            f"""
            <div style="
                padding:10px;
                border-radius:10px;
                background-color:{confidence_color};
                color:white;
                font-weight:bold;
                width:fit-content;
            ">
                Predicted Winner — {confidence_label}
            </div>
            """,
            unsafe_allow_html=True
        )

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
            st.metric("Home Penalty", game.get("home_injury_penalty", 0))

        with injury_col2:
            st.metric("Away Penalty", game.get("away_injury_penalty", 0))

        with injury_col3:
            st.metric("Injury Diff", game.get("injury_diff", 0))

        st.metric(
            "Probability Adjustment",
            f"{game.get('injury_probability_adjustment', 0) * 100:.1f}%"
        )

        odds = {}

        game_home = normalize_team_name(game["home_team"]).lower()
        game_away = normalize_team_name(game["away_team"]).lower()
    if not isinstance(odds_map, dict):
        odds_map = {}
    for (home, away), value in odds_map.items():
            odds_home = normalize_team_name(home).lower()
            odds_away = normalize_team_name(away).lower()

            teams_match = (
                odds_home == game_home
                and odds_away == game_away
            )

            if teams_match:
                odds = value
                break

        home_odds_data = odds.get(game_home)
        away_odds_data = odds.get(game_away)

        home_odds = home_odds_data["price"] if home_odds_data else None
        away_odds = away_odds_data["price"] if away_odds_data else None

        home_bookmaker = home_odds_data["bookmaker"] if home_odds_data else "N/A"
        away_bookmaker = away_odds_data["bookmaker"] if away_odds_data else "N/A"

        if home_odds and away_odds:
            st.subheader("Betting Analytics")

            calibrated_home_prob = calibrate_probability(
                game["home_win_probability"]
            )

            calibrated_away_prob = calibrate_probability(
                game["away_win_probability"]
            )

            home_ev, home_implied = calculate_ev(
                calibrated_home_prob,
                home_odds
            )

            away_ev, away_implied = calculate_ev(
                calibrated_away_prob,
                away_odds
            )

            home_edge = calculate_model_edge(
                calibrated_home_prob,
                home_implied
            )

            away_edge = calculate_model_edge(
                calibrated_away_prob,
                away_implied
            )

            home_edge_label = classify_edge(home_edge)
            away_edge_label = classify_edge(away_edge)

            home_kelly = kelly_fraction(
                calibrated_home_prob,
                home_odds
            )

            away_kelly = kelly_fraction(
                calibrated_away_prob,
                away_odds
            )

            analytics_col1, analytics_col2 = st.columns(2)

            with analytics_col1:
                st.metric(f"{game['home_team']} Odds", f"{home_odds:.2f}")
                st.caption(f"Best book: {home_bookmaker}")
                st.metric("Implied Probability", f"{home_implied * 100:.1f}%")
                st.metric("Model Edge", f"{home_edge * 100:.1f}%")
                st.caption(home_edge_label)
                st.metric("Expected Value", f"{home_ev * 100:.1f}%")
                st.metric("Kelly %", f"{home_kelly * 100:.1f}%")

            with analytics_col2:
                st.metric(f"{game['away_team']} Odds", f"{away_odds:.2f}")
                st.caption(f"Best book: {away_bookmaker}")
                st.metric("Implied Probability", f"{away_implied * 100:.1f}%")
                st.metric("Model Edge", f"{away_edge * 100:.1f}%")
                st.caption(away_edge_label)
                st.metric("Expected Value", f"{away_ev * 100:.1f}%")
                st.metric("Kelly %", f"{away_kelly * 100:.1f}%")

            best_bet = None
            best_ev = 0
            best_confidence = confidence

            MIN_EV = 0.05
            MIN_EDGE = 0.03
            MIN_KELLY = 0.01
            MIN_CONFIDENCE = 0.60

            if home_ev > away_ev:
                candidate_bet = game["home_team"]
                candidate_ev = home_ev
                candidate_edge = home_edge
                candidate_kelly = home_kelly
            else:
                candidate_bet = game["away_team"]
                candidate_ev = away_ev
                candidate_edge = away_edge
                candidate_kelly = away_kelly

            passes_filter = (
                candidate_ev >= MIN_EV
                and candidate_edge >= MIN_EDGE
                and candidate_kelly >= MIN_KELLY
                and best_confidence >= MIN_CONFIDENCE
            )

            if passes_filter:
                best_bet = candidate_bet
                best_ev = candidate_ev

            if best_bet:
                st.success(
                    f"🔥 BEST BET: {best_bet} | Expected Value: {best_ev * 100:.1f}%"
                )

                if best_bet == game["home_team"]:
                    selected_odds = home_odds
                    selected_prob = calibrated_home_prob
                    selected_kelly = home_kelly
                else:
                    selected_odds = away_odds
                    selected_prob = calibrated_away_prob
                    selected_kelly = away_kelly

                button_key = f"button_{game['home_team']}_{game['away_team']}_{best_bet}"
                saved_key = f"saved_{game['home_team']}_{game['away_team']}_{best_bet}"

                if saved_key not in st.session_state:
                    st.session_state[saved_key] = False

                if st.button(f"Save Bet Pick: {best_bet}", key=button_key):
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
                st.error("🚫 NO BET — failed professional value filter")

                with st.expander("Why this game was rejected"):
                    st.write(f"Required EV: at least {MIN_EV * 100:.1f}%")
                    st.write(f"Required Edge: at least {MIN_EDGE * 100:.1f}%")
                    st.write(f"Required Kelly: at least {MIN_KELLY * 100:.1f}%")
                    st.write(f"Required Confidence: at least {MIN_CONFIDENCE * 100:.1f}%")

                    st.write("---")
                    st.write(f"Best Candidate: {candidate_bet}")
                    st.write(f"Candidate EV: {candidate_ev * 100:.1f}%")
                    st.write(f"Candidate Edge: {candidate_edge * 100:.1f}%")
                    st.write(f"Candidate Kelly: {candidate_kelly * 100:.1f}%")
                    st.write(f"Model Confidence: {best_confidence * 100:.1f}%")

        else:
            if live_odds_mode:
                st.warning("No sportsbook odds found for this matchup.")
            else:
                st.warning("Stored historical odds not found for this exact matchup.")

                with st.expander("Debug historical odds matching"):
                    st.write("Prediction matchup:")
                    st.write(game_away, "@", game_home)
                    st.write("Available historical odds matchups:")
                    st.write(list(odds_map.keys()))

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

    updated_df["clv"] = updated_df.apply(
        lambda row: calculate_clv(row["odds"], row["closing_odds"]),
        axis=1
    )

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

    numeric_clv = pd.to_numeric(updated_df["clv"], errors="coerce")
    avg_clv = numeric_clv.mean() * 100 if not numeric_clv.dropna().empty else 0

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

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Average EV", f"{avg_ev:.1f}%")

    with col5:
        st.metric("Average Kelly", f"{avg_kelly:.1f}%")

    with col6:
        st.metric("Average CLV", f"{avg_clv:.1f}%")

    st.subheader("Update Bet Results + CLV")

    for index, row in updated_df.iterrows():
        current_result = row["result"]

        if current_result not in ["Pending", "Win", "Loss"]:
            current_result = "Pending"

        st.write(
            f"{row['game_date']} — {row['best_bet']} "
            f"({row['away_team']} @ {row['home_team']})"
        )

        updated_df.at[index, "closing_odds"] = st.text_input(
            "Closing Odds",
            value=str(row.get("closing_odds", "")),
            key=f"closing_odds_{index}"
        )

        updated_df.at[index, "result"] = st.selectbox(
            "Result",
            ["Pending", "Win", "Loss"],
            index=["Pending", "Win", "Loss"].index(current_result),
            key=f"result_{index}"
        )

    if st.button("Save Updated Results"):
        updated_df["profit_loss"] = updated_df.apply(calculate_profit_loss, axis=1)

        updated_df["clv"] = updated_df.apply(
            lambda row: calculate_clv(row["odds"], row["closing_odds"]),
            axis=1
        )

        save_bet_history(updated_df)
        st.success("Bet results and CLV updated successfully!")

    st.subheader("Bankroll Growth")

    chart_df = updated_df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
    chart_df = chart_df.sort_values("timestamp")
    chart_df["cumulative_profit"] = chart_df["profit_loss"].cumsum()

    st.line_chart(chart_df.set_index("timestamp")["cumulative_profit"])

    st.subheader("Saved Bet Picks")

    def highlight_results(row):
        result = str(row["result"]).lower()

        if result == "win":
            return ["background-color: #d4edda"] * len(row)

        if result == "loss":
            return ["background-color: #f8d7da"] * len(row)

        return ["background-color: #fff3cd"] * len(row)

    styled_df = updated_df.style.apply(highlight_results, axis=1)

    st.write(
        styled_df.to_html(),
        unsafe_allow_html=True
    )
