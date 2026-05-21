import os
import requests
import pandas as pd

from auto_learning import summarize_learning

API_URL = "https://oluwa-blazee-new.onrender.com"
BET_HISTORY_FILE = "bet_history.csv"


def safe_read_csv(path):
    if not os.path.isfile(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def calculate_profit_loss(result, odds, stake):
    try:
        odds = float(odds)
        stake = float(stake)
    except Exception:
        return 0

    result = str(result).lower()

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

        return round((saved_odds / closing_odds) - 1, 4)

    except Exception:
        return ""


def fetch_game_result(row):
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
            return None

        data = response.json()

        if data.get("status") != "completed":
            return None

        return data.get("result")

    except Exception:
        return None


def update_bet_results():
    df = safe_read_csv(BET_HISTORY_FILE)

    if df.empty:
        return {
            "status": "empty",
            "message": "No bet history found."
        }

    for col in ["result", "profit_loss", "closing_odds", "clv", "stake"]:
        if col not in df.columns:
            if col == "result":
                df[col] = "Pending"
            elif col == "stake":
                df[col] = 100
            else:
                df[col] = ""

    updated_rows = 0

    for index, row in df.iterrows():
        current_result = str(row.get("result", "Pending")).lower()

        if current_result in ["win", "loss"]:
            continue

        latest_result = fetch_game_result(row)

        if latest_result is None:
            continue

        df.loc[index, "result"] = latest_result

        df.loc[index, "profit_loss"] = calculate_profit_loss(
            latest_result,
            row.get("odds", 0),
            row.get("stake", 100)
        )

        df.loc[index, "clv"] = calculate_clv(
            row.get("odds", 0),
            row.get("closing_odds", "")
        )

        updated_rows += 1

    df.to_csv(BET_HISTORY_FILE, index=False)

    learning_summary = summarize_learning()

    return {
        "status": "success",
        "updated_rows": updated_rows,
        "total_rows": len(df),
        "learning_summary": learning_summary
    }


if __name__ == "__main__":
    print(update_bet_results())
