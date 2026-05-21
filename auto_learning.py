import os
import pandas as pd

PREDICTION_HISTORY_PATH = "prediction_history.csv"
BET_HISTORY_PATH = "bet_history.csv"
OUTPUT_PATH = "learning_dataset.csv"


def safe_read_csv(path):
    if not os.path.isfile(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_learning_dataset():
    predictions = safe_read_csv(PREDICTION_HISTORY_PATH)
    bets = safe_read_csv(BET_HISTORY_PATH)

    if predictions.empty:
        return pd.DataFrame()

    learning_df = predictions.copy()
    if not bets.empty:

    merge_cols = [
        "game_date",
        "home_team",
        "away_team"
    ]

    learning_df = learning_df.merge(
        bets,
        on=merge_cols,
        how="left",
        suffixes=("_prediction", "_bet")
    )
    if "result" in learning_df.columns:

        learning_df["target_win"] = learning_df["result"].apply(
            lambda x: 1 if str(x).lower() == "win" else 0
        )
    
    else:
    
        learning_df["target_win"] = 0

    if not bets.empty:
        merge_cols = ["game_date", "home_team", "away_team"]

        learning_df = learning_df.merge(
            bets,
            on=merge_cols,
            how="left",
            suffixes=("_prediction", "_bet")
        )

    numeric_cols = [
        "home_probability",
        "away_probability",
        "odds",
        "model_probability",
        "expected_value",
        "kelly",
        "stake",
        "profit_loss",
        "clv"
    ]

    for col in numeric_cols:
        if col in learning_df.columns:
            learning_df[col] = pd.to_numeric(learning_df[col], errors="coerce")

    if "home_probability" in learning_df.columns and "away_probability" in learning_df.columns:
        learning_df["model_confidence"] = learning_df[
            ["home_probability", "away_probability"]
        ].max(axis=1)

    if "result" in learning_df.columns:
        learning_df["target_win"] = learning_df["result"].apply(
            lambda x: 1 if str(x).lower() == "win" else 0
        )
    else:
        learning_df["target_win"] = 0

    learning_df.to_csv(OUTPUT_PATH, index=False)

    return learning_df


def summarize_learning():
    df = build_learning_dataset()

    if df.empty:
        return {
            "status": "empty",
            "message": "No learning data available yet."
        }

    return {
        "status": "success",
        "rows": len(df),
        "output_file": OUTPUT_PATH,
        "has_target_win": "target_win" in df.columns
    }


if __name__ == "__main__":
    print(summarize_learning())
