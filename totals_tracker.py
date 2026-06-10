import os
import pandas as pd


TOTALS_HISTORY_FILE = "totals_history.csv"


def save_totals_pick(
    game_date,
    home_team,
    away_team,
    projected_total,
    sportsbook_total,
    recommendation
):

    row = {
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "projected_total": projected_total,
        "sportsbook_total": sportsbook_total,
        "recommendation": recommendation,
        "actual_total": None,
        "result": "Pending",
        "profit_loss": 0
    }

    if os.path.exists(TOTALS_HISTORY_FILE):
        df = pd.read_csv(TOTALS_HISTORY_FILE)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True
    )

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    return True


def load_totals_history():

    if not os.path.exists(TOTALS_HISTORY_FILE):
        return pd.DataFrame()

    return pd.read_csv(TOTALS_HISTORY_FILE)
