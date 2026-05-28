import os
import pandas as pd
from datetime import datetime


HISTORICAL_FILE = "historical_training_data.csv"


def create_historical_training_file():
    columns = [
        "game_date",
        "home_team",
        "away_team",
        "home_odds",
        "away_odds",
        "model_probability",
        "expected_value",
        "kelly",
        "result"
    ]

    if not os.path.isfile(HISTORICAL_FILE):
        df = pd.DataFrame(columns=columns)
        df.to_csv(HISTORICAL_FILE, index=False)

    return {
        "status": "success",
        "message": "Historical training file ready.",
        "file": HISTORICAL_FILE
    }


def add_historical_game(
    game_date,
    home_team,
    away_team,
    selected_team,
    odds,
    model_probability,
    expected_value,
    kelly,
    result
):
    create_historical_training_file()

    row = {
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "selected_team": selected_team,
        "odds": odds,
        "model_probability": model_probability,
        "expected_value": expected_value,
        "kelly": kelly,
        "result": result,
        "created_at": datetime.now().isoformat()
    }

    df = pd.read_csv(HISTORICAL_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORICAL_FILE, index=False)

    return {
        "status": "success",
        "message": "Historical game added.",
        "rows": len(df)
    }


def merge_historical_into_bet_history():
    if not os.path.isfile(HISTORICAL_FILE):
        return {
            "status": "error",
            "message": "historical_training_data.csv not found."
        }

    historical_df = pd.read_csv(HISTORICAL_FILE)

    if historical_df.empty:
        return {
            "status": "error",
            "message": "No historical rows available."
        }

    required_cols = [
        "game_date",
        "home_team",
        "away_team",
        "selected_team",
        "odds",
        "model_probability",
        "expected_value",
        "kelly",
        "result"
    ]

    for col in required_cols:
        if col not in historical_df.columns:
            return {
                "status": "error",
                "message": f"Missing column: {col}"
            }

    rows = []

    for _, row in historical_df.iterrows():
        rows.append({
            "timestamp": datetime.now().isoformat(),
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "best_bet": row["selected_team"],
            "odds": row["odds"],
            "model_probability": row["model_probability"],
            "expected_value": row["expected_value"],
            "kelly": row["kelly"],
            "stake": 100,
            "result": row["result"],
            "profit_loss": (
                (float(row["odds"]) - 1) * 100
                if row["result"] == "Win"
                else -100
            ),
            "closing_odds": "",
            "clv": ""
        })

    new_df = pd.DataFrame(rows)

    if os.path.isfile("bet_history.csv"):
        old_df = pd.read_csv("bet_history.csv")
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv("bet_history.csv", index=False)

    return {
        "status": "success",
        "message": "Historical rows merged into bet_history.csv.",
        "merged_rows": len(new_df),
        "total_bet_history_rows": len(final_df)
    }
