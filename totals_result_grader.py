import os
import pandas as pd
import requests


TOTALS_HISTORY_FILE = "totals_history.csv"
API_URL = "https://oluwa-blazee-new.onrender.com"
STAKE = 100
WIN_PROFIT = 91


def fetch_final_score(game_date, home_team, away_team):

    try:
        response = requests.get(
            f"{API_URL}/score_result",
            params={
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "best_bet": home_team
            },
            timeout=30
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("status") != "completed":
            return None

        return {
            "home_score": float(data.get("home_score", 0)),
            "away_score": float(data.get("away_score", 0))
        }

    except Exception:
        return None


def grade_totals_results():

    if not os.path.exists(TOTALS_HISTORY_FILE):
        return {
            "status": "error",
            "message": "totals_history.csv not found"
        }

    df = pd.read_csv(TOTALS_HISTORY_FILE)

    if df.empty:
        return {
            "status": "error",
            "message": "No totals picks available"
        }

    for col in [
        "actual_total",
        "home_score",
        "away_score",
        "result",
        "profit_loss"
    ]:
        if col not in df.columns:
            if col == "result":
                df[col] = "Pending"
            elif col == "profit_loss":
                df[col] = 0
            else:
                df[col] = None

    updated_rows = 0

    for idx, row in df.iterrows():

        if str(row.get("result", "Pending")).lower() in ["win", "loss"]:
            continue

        score_data = fetch_final_score(
            row["game_date"],
            row["home_team"],
            row["away_team"]
        )

        if score_data is None:
            continue

        home_score = score_data["home_score"]
        away_score = score_data["away_score"]

        actual_total = home_score + away_score

        try:
            sportsbook_total = float(row["sportsbook_total"])
        except Exception:
            continue

        recommendation = str(
            row.get("recommendation", "")
        ).lower()

        if "over" in recommendation:
            result = "Win" if actual_total > sportsbook_total else "Loss"

        elif "under" in recommendation:
            result = "Win" if actual_total < sportsbook_total else "Loss"

        else:
            continue

        if result == "Win":
            profit_loss = WIN_PROFIT
        else:
            profit_loss = -STAKE

        df.loc[idx, "home_score"] = home_score
        df.loc[idx, "away_score"] = away_score
        df.loc[idx, "actual_total"] = actual_total
        df.loc[idx, "result"] = result
        df.loc[idx, "profit_loss"] = profit_loss

        updated_rows += 1

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    settled = df[
        df["result"].astype(str).str.lower().isin(["win", "loss"])
    ]

    wins = len(
        settled[
            settled["result"].astype(str).str.lower() == "win"
        ]
    )

    losses = len(
        settled[
            settled["result"].astype(str).str.lower() == "loss"
        ]
    )

    total_profit = pd.to_numeric(
        settled["profit_loss"],
        errors="coerce"
    ).fillna(0).sum()

    roi = 0

    if len(settled) > 0:
        roi = round(
            total_profit / (len(settled) * STAKE) * 100,
            2
        )

    return {
        "status": "success",
        "updated_rows": updated_rows,
        "settled": len(settled),
        "wins": wins,
        "losses": losses,
        "profit": round(float(total_profit), 2),
        "roi": roi
    }


if __name__ == "__main__":
    print(grade_totals_results())
