import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "models/basketball_xgb_calibrated_v3.joblib"
DATA_PATH = "outputs/training_dataset.parquet"

STARTING_BANKROLL = 1000
STAKE = 100
MIN_EV_THRESHOLD = 0.05
MARKET_NOISE_STD = 0.04
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

history = pd.read_parquet(DATA_PATH)


def calculate_ev(model_prob, decimal_odds):
    return (model_prob * (decimal_odds - 1)) - (1 - model_prob)


def calibrate_probability(probability, strength=0.75):
    probability = max(min(probability, 0.95), 0.05)
    return 0.5 + ((probability - 0.5) * strength)


def simulate_market_odds(home_prob):
    market_home_prob = min(
        max(
            home_prob + np.random.normal(0, MARKET_NOISE_STD),
            0.05
        ),
        0.95
    )

    market_away_prob = 1 - market_home_prob

    home_odds = 1 / market_home_prob
    away_odds = 1 / market_away_prob

    return market_home_prob, market_away_prob, home_odds, away_odds


def simulate_backtest():
    bankroll = STARTING_BANKROLL
    results = []

    df = history.copy()

    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    probs = model.predict_proba(X)[:, 1]

    df["raw_home_prob"] = probs
    df["home_prob"] = df["raw_home_prob"].apply(calibrate_probability)
    df["away_prob"] = 1 - df["home_prob"]

    debug_counter = 0

    for _, row in df.iterrows():
        home_prob = row["home_prob"]
        away_prob = row["away_prob"]

        market_home_prob, market_away_prob, home_odds, away_odds = simulate_market_odds(
            home_prob
        )

        home_ev = calculate_ev(home_prob, home_odds)
        away_ev = calculate_ev(away_prob, away_odds)

        if debug_counter < 5:
            print(
                "DEBUG:",
                "home_prob=", round(home_prob, 4),
                "market_home_prob=", round(market_home_prob, 4),
                "home_odds=", round(home_odds, 3),
                "home_ev=", round(home_ev, 4),
                "away_ev=", round(away_ev, 4)
            )
            debug_counter += 1

        bet_team = None
        bet_side = None
        bet_odds = None
        bet_ev = None
        bet_model_prob = None
        bet_market_prob = None

        if home_ev > away_ev and home_ev >= MIN_EV_THRESHOLD:
            bet_team = row["home_team_name"]
            bet_side = "home"
            bet_odds = home_odds
            bet_ev = home_ev
            bet_model_prob = home_prob
            bet_market_prob = market_home_prob

        elif away_ev > home_ev and away_ev >= MIN_EV_THRESHOLD:
            bet_team = row["away_team_name"]
            bet_side = "away"
            bet_odds = away_odds
            bet_ev = away_ev
            bet_model_prob = away_prob
            bet_market_prob = market_away_prob

        if bet_team is None:
            continue

        if bet_side == "home":
            won = row["home_win"] == 1
        else:
            won = row["home_win"] == 0

        profit = (bet_odds - 1) * STAKE if won else -STAKE
        bankroll += profit

        results.append({
            "date": row.get("date"),
            "home_team": row["home_team_name"],
            "away_team": row["away_team_name"],
            "bet_team": bet_team,
            "bet_side": bet_side,
            "bet_odds": bet_odds,
            "model_prob": bet_model_prob,
            "market_prob": bet_market_prob,
            "model_edge": bet_model_prob - bet_market_prob,
            "expected_value": bet_ev,
            "result": "Win" if won else "Loss",
            "profit": profit,
            "bankroll": bankroll
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No bets matched the strategy.")
        return

    total_bets = len(results_df)
    wins = len(results_df[results_df["result"] == "Win"])
    losses = len(results_df[results_df["result"] == "Loss"])
    win_rate = wins / total_bets
    total_profit = results_df["profit"].sum()
    roi = total_profit / (total_bets * STAKE)
    final_bankroll = bankroll

    print("===== BACKTEST RESULTS =====")
    print(f"Total Bets: {total_bets}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.2%}")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"Average EV: {results_df['expected_value'].mean():.2%}")
    print(f"Average Edge: {results_df['model_edge'].mean():.2%}")

    results_df.to_csv("outputs/backtest_results.csv", index=False)
    print("Saved: outputs/backtest_results.csv")


if __name__ == "__main__":
    simulate_backtest()
