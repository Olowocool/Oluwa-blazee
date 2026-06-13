import os
import pandas as pd
from datetime import datetime

ODDS_FILE = "odds_snapshots.csv"


def save_odds_snapshot(games):

    rows = []

    for game in games:

        rows.append({
            "timestamp": datetime.now().isoformat(),

            "game_date": game.get("game_date"),

            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),

            # Winner Market
            "home_odds": game.get("home_odds"),
            "away_odds": game.get("away_odds"),

            # Totals Market
            "total_line": game.get("total_line"),
            "over_odds": game.get("over_odds"),
            "under_odds": game.get("under_odds")
        })

    snapshot_df = pd.DataFrame(rows)

    if os.path.exists(ODDS_FILE):

        existing = pd.read_csv(ODDS_FILE)

        snapshot_df = pd.concat(
            [existing, snapshot_df],
            ignore_index=True
        )

    snapshot_df.to_csv(
        ODDS_FILE,
        index=False
    )

    return {
        "status": "success",
        "saved_rows": len(rows),
        "file": ODDS_FILE
    }
