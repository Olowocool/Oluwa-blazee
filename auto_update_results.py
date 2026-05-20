import os
import requests
import pandas as pd

API_URL = "https://oluwa-blazee-new.onrender.com"

BET_HISTORY_FILE = "bet_history.csv"
LEARNING_DATASET_FILE = "learning_dataset.csv"


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
            return None

        return round(
            ((saved_odds / closing_odds) - 1),
            4
        )

    except Exception:
        return None


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

    if "result" not in df.columns:
        df["result"] = "Pending"

    if "profit_loss" not in df.columns:
        df["profit_loss"] = 0

    if "closing_odds" not in df.columns:
        df["closing_odds"] = ""

    if "clv" not in df.columns:
        df["clv"] = ""

    updated_rows = 0

    for index, row in df.iterrows():

        current_result = str(
            row.get("result", "Pending")
        ).lower()

        if current_result in ["win", "loss"]:
            continue

        latest_result = fetch_game_result(row)

        if latest_result is None:
            continue

        df.loc[index, "result"] = latest_result

        profit_loss = calculate_profit_loss(
            latest_result,
            row["odds"],
            row.get("stake", 100)
        )

        df.loc[index, "profit_loss"] = profit_loss

        try:

            closing_odds = row.get("closing_odds")

            if pd.notna(closing_odds) and closing_odds != "":

                clv = calculate_clv(
                    row["odds"],
                    closing_odds
                )

                df.loc[index, "clv"] = clv

        except Exception:
            pass

        updated_rows += 1

    df.to_csv(BET_HISTORY_FILE, index=False)

    return {

        "status": "success",

        "updated_rows": updated_rows,

        "total_rows": len(df)
    }


if __name__ == "__main__":

    result = update_bet_results()

    print(result)
