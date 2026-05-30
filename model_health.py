import os
import pandas as pd
from datetime import datetime


def get_model_health():

    result = {
        "training_rows": 0,
        "win_rate": 0,
        "roi": 0,
        "last_model_update": "N/A"
    }

    try:

        if os.path.exists("bet_history.csv"):

            df = pd.read_csv("bet_history.csv")

            settled = df[
                df["result"].isin(["Win", "Loss"])
            ]

            if len(settled) > 0:

                wins = len(
                    settled[
                        settled["result"] == "Win"
                    ]
                )

                result["win_rate"] = round(
                    wins / len(settled) * 100,
                    2
                )

                if "profit_loss" in settled.columns:

                    total_profit = settled[
                        "profit_loss"
                    ].sum()

                    stake = len(settled) * 100

                    result["roi"] = round(
                        (total_profit / stake) * 100,
                        2
                    )

        if os.path.exists("bet_history.csv"):

            bet_df = pd.read_csv("bet_history.csv")
        
            settled = bet_df[
                bet_df["result"].astype(str).isin(["Win", "Loss"])
            ]
        
            result["training_rows"] = len(settled)

        if os.path.exists(
            "models/ensemble_model.joblib"
        ):

            ts = os.path.getmtime(
                "models/ensemble_model.joblib"
            )

            result["last_model_update"] = (
                datetime.fromtimestamp(ts)
                .strftime("%Y-%m-%d %H:%M")
            )

    except Exception as e:

        result["error"] = str(e)

    return result
