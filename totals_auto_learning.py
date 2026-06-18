import os
import pandas as pd


TOTALS_HISTORY_FILE = "totals_history.csv"
TOTALS_LEARNING_FILE = "totals_learning_dataset.csv"


def build_totals_learning_dataset():

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

    df = df[
        df["result"].astype(str).str.lower().isin(["win", "loss"])
    ].copy()

    if df.empty:
        return {
            "status": "error",
            "message": "No graded totals picks found"
        }

    for col in [
        "projected_total",
        "sportsbook_total",
        "edge",
        "actual_total",
        "profit_loss"
    ]:
        df[col] = pd.to_numeric(
            df.get(col, 0),
            errors="coerce"
        ).fillna(0)

    df["is_under"] = (
        df["pick_type"].astype(str).str.lower() == "under"
    ).astype(int)

    df["is_over"] = (
        df["pick_type"].astype(str).str.lower() == "over"
    ).astype(int)

    df["target"] = (
        df["result"].astype(str).str.lower() == "win"
    ).astype(int)

    learning_df = df[
        [
            "game_date",
            "home_team",
            "away_team",
            "pick_type",
            "projected_total",
            "sportsbook_total",
            "edge",
            "actual_total",
            "is_under",
            "is_over",
            "profit_loss",
            "target"
        ]
    ].copy()

    learning_df.to_csv(
        TOTALS_LEARNING_FILE,
        index=False
    )

    return {
        "status": "success",
        "rows": len(learning_df),
        "file": TOTALS_LEARNING_FILE
    }
