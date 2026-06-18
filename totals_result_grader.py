import os
import pandas as pd
import requests

TOTALS_HISTORY_FILE = "totals_history.csv"

API_URL = "https://oluwa-blazee-new.onrender.com"


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
            "home_score": data.get("home_score", 0),
            "away_score": data.get("away_score", 0)
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
            "message": "No totals picks found"
        }

    updated_rows = 0

    for idx, row in df.iterrows():

        result = str(
            row.get("result", "Pending")
        )

        if result.lower() in ["win", "loss"]:
            continue

        score_data = fetch_final_score(
            row["game_date"],
            row["home_team"],
            row["away_team"]
        )

        if score_data is None:
            continue

        actual_total = (
            score_data["home_score"]
            + score_data["away_score"]
        )

        recommendation = str(
            row["recommendation"]
        ).lower()

        sportsbook_total = float(
            row["sportsbook_total"]
        )

        win = False

        if "over" in recommendation:
            win = actual_total > sportsbook_total

        elif "under" in recommendation:
            win = actual_total < sportsbook_total

        df.loc[idx, "actual_total"] = actual_total

        if win:
            df.loc[idx, "result"] = "Win"
            df.loc[idx, "profit_loss"] = 91
        else:
            df.loc[idx, "result"] = "Loss"
            df.loc[idx, "profit_loss"] = -100

        updated_rows += 1

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    wins = len(
        df[df["result"] == "Win"]
    )

    losses = len(
        df[df["result"] == "Loss"]
    )

    settled = wins + losses

    roi = 0

    if settled > 0:
        roi = (
            df["profit_loss"].sum()
            / (settled * 100)
        ) * 100

    return {
        "status": "success",
        "updated_rows": updated_rows,
        "wins": wins,
        "losses": losses,
        "settled": settled,
        "roi": round(roi, 2)
    }


if __name__ == "__main__":
    print(
        grade_totals_results()
    )
