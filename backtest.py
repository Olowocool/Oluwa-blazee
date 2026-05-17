import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "models/basketball_xgb_calibrated_v3.joblib"
DATA_PATH = "outputs/training_dataset.parquet"

STARTING_BANKROLL = 1000
FLAT_STAKE = 100

MARKET_NOISE_STD = 0.015
RANDOM_SEED = 42

FRACTIONAL_KELLY_MULTIPLIER = 0.10
MAX_BANKROLL_RISK_PER_BET = 0.03

MIN_BET_SIZE = 10
MAX_BET_SIZE = 500

EV_THRESHOLDS = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]


np.random.seed(RANDOM_SEED)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

history = pd.read_parquet(DATA_PATH)


def calculate_ev(model_prob, decimal_odds):
    return (model_prob * (decimal_odds - 1)) - (1 - model_prob)


def kelly_fraction(probability, decimal_odds):
    b = decimal_odds - 1
    q = 1 - probability

    if b <= 0:
        return 0

    kelly = ((b * probability) - q) / b

    return max(kelly, 0)


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


def calculate_dynamic_bet_size(bankroll, model_prob, decimal_odds):
    kelly = kelly_fraction(model_prob, decimal_odds)

    fractional_kelly = kelly * FRACTIONAL_KELLY_MULTIPLIER

    fractional_kelly = min(
        fractional_kelly,
        MAX_BANKROLL_RISK_PER_BET
    )

    bet_size = bankroll * fractional_kelly

    bet_size = max(min(bet_size, MAX_BET_SIZE), MIN_BET_SIZE)

    return bet_size, kelly, fractional_kelly


def calculate_max_loss_streak(results_series):
    loss_streak = 0
    max_loss_streak = 0

    for result in results_series:
        if result == "Loss":
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    return max_loss_streak


def run_backtest_for_threshold(threshold):
    np.random.seed(RANDOM_SEED)

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

    for _, row in df.iterrows():
        home_prob = row["home_prob"]
        away_prob = row["away_prob"]

        market_home_prob, market_away_prob, home_odds, away_odds = simulate_market_odds(
            home_prob
        )

        home_ev = calculate_ev(home_prob, home_odds)
        away_ev = calculate_ev(away_prob, away_odds)

        bet_team = None
        bet_side = None
        bet_odds = None
        bet_ev = None
        bet_model_prob = None
        bet_market_prob = None

        if home_ev > away_ev and home_ev >= threshold:
            bet_team = row["home_team_name"]
            bet_side = "home"
            bet_odds = home_odds
            bet_ev = home_ev
            bet_model_prob = home_prob
            bet_market_prob = market_home_prob

        elif away_ev > home_ev and away_ev >= threshold:
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

        bet_size, kelly, fractional_kelly = calculate_dynamic_bet_size(
            bankroll,
            bet_model_prob,
            bet_odds
        )

        profit = (
            (bet_odds - 1) * bet_size
            if won
            else -bet_size
        )

        bankroll += profit

        results.append({
            "threshold": threshold,
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
            "kelly": kelly,
            "fractional_kelly": fractional_kelly,
            "bet_size": bet_size,
            "result": "Win" if won else "Loss",
            "profit": profit,
            "bankroll": bankroll
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            "threshold": threshold,
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_profit": 0,
            "roi": 0,
            "final_bankroll": STARTING_BANKROLL,
            "avg_ev": 0,
            "avg_edge": 0,
            "avg_bet_size": 0,
            "avg_kelly": 0,
            "max_drawdown": 0,
            "max_loss_streak": 0,
            "profit_volatility": 0
        }, results_df

    results_df["running_peak"] = results_df["bankroll"].cummax()

    results_df["drawdown"] = (
        results_df["bankroll"] - results_df["running_peak"]
    ) / results_df["running_peak"]

    max_drawdown = results_df["drawdown"].min()
    max_loss_streak = calculate_max_loss_streak(results_df["result"])
    profit_std = results_df["profit"].std()

    total_bets = len(results_df)
    wins = len(results_df[results_df["result"] == "Win"])
    losses = len(results_df[results_df["result"] == "Loss"])
    win_rate = wins / total_bets
    total_profit = results_df["profit"].sum()
    total_staked = results_df["bet_size"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0

    summary = {
        "threshold": threshold,
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "roi": roi,
        "final_bankroll": bankroll,
        "avg_ev": results_df["expected_value"].mean(),
        "avg_edge": results_df["model_edge"].mean(),
        "avg_bet_size": results_df["bet_size"].mean(),
        "avg_kelly": results_df["kelly"].mean(),
        "max_drawdown": max_drawdown,
        "max_loss_streak": max_loss_streak,
        "profit_volatility": profit_std
    }

    return summary, results_df


def run_threshold_sweep():
    summaries = []
    all_results = []

    for threshold in EV_THRESHOLDS:
        summary, results_df = run_backtest_for_threshold(threshold)
        summaries.append(summary)

        if not results_df.empty:
            all_results.append(results_df)

    summary_df = pd.DataFrame(summaries)

    print("===== EV THRESHOLD + RISK-CAPPED KELLY SWEEP RESULTS =====")

    display_df = summary_df.copy()
    display_df["win_rate"] = display_df["win_rate"].map(lambda x: f"{x:.2%}")
    display_df["roi"] = display_df["roi"].map(lambda x: f"{x:.2%}")
    display_df["avg_ev"] = display_df["avg_ev"].map(lambda x: f"{x:.2%}")
    display_df["avg_edge"] = display_df["avg_edge"].map(lambda x: f"{x:.2%}")
    display_df["avg_kelly"] = display_df["avg_kelly"].map(lambda x: f"{x:.2%}")
    display_df["max_drawdown"] = display_df["max_drawdown"].map(lambda x: f"{x:.2%}")
    display_df["avg_bet_size"] = display_df["avg_bet_size"].map(lambda x: f"${x:.2f}")
    display_df["profit_volatility"] = display_df["profit_volatility"].map(lambda x: f"${x:.2f}")
    display_df["total_profit"] = display_df["total_profit"].map(lambda x: f"${x:.2f}")
    display_df["final_bankroll"] = display_df["final_bankroll"].map(lambda x: f"${x:.2f}")

    print(display_df.to_string(index=False))

    summary_df.to_csv("outputs/backtest_threshold_summary.csv", index=False)

    if all_results:
        full_results_df = pd.concat(all_results, ignore_index=True)
        full_results_df.to_csv("outputs/backtest_results.csv", index=False)

    print("Saved: outputs/backtest_threshold_summary.csv")
    print("Saved: outputs/backtest_results.csv")


if __name__ == "__main__":
    run_threshold_sweep()
