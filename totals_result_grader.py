import os
import pandas as pd
import requests

from totals_clv import calculate_totals_clv


TOTALS_HISTORY_FILE = "totals_history.csv"
API_URL = "https://oluwa-blazee-new.onrender.com"

STAKE = 100
WIN_PROFIT = 91


def fetch_final_score(game_date, home_team, away_team):

    attempts = [
        {
            "home_team": home_team,
            "away_team": away_team,
            "best_bet": home_team
        },
        {
            "home_team": away_team,
            "away_team": home_team,
            "best_bet": away_team
        }
    ]

    for params in attempts:

        try:
            response = requests.get(
                f"{API_URL}/score_result",
                params={
                    "date": game_date,
                    "home_team": params["home_team"],
                    "away_team": params["away_team"],
                    "best_bet": params["best_bet"]
                },
                timeout=30
            )

            if response.status_code != 200:
                continue

            data = response.json()

            if data.get("status") != "completed":
                continue

            return {
                "home_score": float(data.get("home_score", 0)),
                "away_score": float(data.get("away_score", 0)),
                "matched_home_team": params["home_team"],
                "matched_away_team": params["away_team"]
            }

        except Exception:
            continue

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
        "saved_total",
        "closing_total",
        "clv",
        "actual_total",
        "home_score",
        "away_score",
        "matched_home_team",
        "matched_away_team",
        "result",
        "profit_loss"
    ]:
        if col not in df.columns:
            if col == "result":
                df[col] = "Pending"
            elif col == "profit_loss":
                df[col] = 0
            elif col == "saved_total":
                df[col] = df.get("sportsbook_total", None)
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

        recommendation = str(row.get("recommendation", "")).lower()

        if "over" in recommendation:
            result = "Win" if actual_total > sportsbook_total else "Loss"

        elif "under" in recommendation:
            result = "Win" if actual_total < sportsbook_total else "Loss"

        else:
            continue

        profit_loss = WIN_PROFIT if result == "Win" else -STAKE

        saved_total = row.get("saved_total", sportsbook_total)
        closing_total = row.get("closing_total", sportsbook_total)

        if pd.isna(saved_total):
            saved_total = sportsbook_total

        if pd.isna(closing_total):
            closing_total = sportsbook_total

        clv = calculate_totals_clv(
            saved_total,
            closing_total
        )

        df.loc[idx, "home_score"] = home_score
        df.loc[idx, "away_score"] = away_score
        df.loc[idx, "matched_home_team"] = score_data["matched_home_team"]
        df.loc[idx, "matched_away_team"] = score_data["matched_away_team"]
        df.loc[idx, "actual_total"] = actual_total
        df.loc[idx, "saved_total"] = saved_total
        df.loc[idx, "closing_total"] = closing_total
        df.loc[idx, "clv"] = clv
        df.loc[idx, "result"] = result
        df.loc[idx, "profit_loss"] = profit_loss

        updated_rows += 1

    df.to_csv(TOTALS_HISTORY_FILE, index=False)

    return {
        "status": "success",
        "updated_rows": updated_rows
    }
