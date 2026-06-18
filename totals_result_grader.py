import os
import pandas as pd

TOTALS_HISTORY_FILE = "totals_history.csv"


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

    updated_rows = 0

    for idx, row in df.iterrows():

        if str(row["result"]).lower() != "pending":
            continue

        try:

            actual_total = row.get("actual_total")

            if pd.isna(actual_total):
                continue

            projected = float(row["projected_total"])
            sportsbook = float(row["sportsbook_total"])

            recommendation = str(
                row["recommendation"]
            ).lower()

            actual_total = float(actual_total)

            if "over" in recommendation:

                if actual_total > sportsbook:
                    result = "Win"
                    profit = 110

                else:
                    result = "Loss"
                    profit = -100

            else:

                if actual_total < sportsbook:
                    result = "Win"
                    profit = 110

                else:
                    result = "Loss"
                    profit = -100

            df.loc[idx, "result"] = result
            df.loc[idx, "profit_loss"] = profit

            updated_rows += 1

        except Exception:
            continue

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    return {
        "status": "success",
        "updated_rows": updated_rows
    }
