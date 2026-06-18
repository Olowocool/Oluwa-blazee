import os
import random
import pandas as pd

TOTALS_HISTORY_FILE = "totals_history.csv"


def add_totals_test_data():

    rows = []

    teams = [
        "Boston Celtics",
        "Denver Nuggets",
        "Milwaukee Bucks",
        "Phoenix Suns",
        "Miami Heat",
        "Dallas Mavericks",
        "Golden State Warriors",
        "Cleveland Cavaliers",
        "Minnesota Timberwolves",
        "Oklahoma City Thunder"
    ]

    for i in range(10):

        sportsbook_total = random.randint(210, 235)

        actual_total = sportsbook_total - random.randint(5, 20)

        rows.append({
            "game_date": "06/14/2026",
            "home_team": random.choice(teams),
            "away_team": random.choice(teams),
            "market": "Totals",
            "pick_type": "Under",
            "projected_total": sportsbook_total - 10,
            "sportsbook_total": sportsbook_total,
            "edge": -10,
            "recommendation": "Strong Under",
            "stake": 100,
            "actual_total": actual_total,
            "result": "Win",
            "profit_loss": 91
        })

    for i in range(10):

        sportsbook_total = random.randint(210, 235)

        actual_total = sportsbook_total + random.randint(5, 20)

        rows.append({
            "game_date": "06/14/2026",
            "home_team": random.choice(teams),
            "away_team": random.choice(teams),
            "market": "Totals",
            "pick_type": "Under",
            "projected_total": sportsbook_total - 10,
            "sportsbook_total": sportsbook_total,
            "edge": -10,
            "recommendation": "Strong Under",
            "stake": 100,
            "actual_total": actual_total,
            "result": "Loss",
            "profit_loss": -100
        })

    if os.path.exists(TOTALS_HISTORY_FILE):
        df = pd.read_csv(TOTALS_HISTORY_FILE)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame(rows)],
        ignore_index=True
    )

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    return {
        "status": "success",
        "rows_added": len(rows)
    }


if __name__ == "__main__":
    print(add_totals_test_data())
