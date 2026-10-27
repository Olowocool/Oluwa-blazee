import os
import random
import pandas as pd


# Synthetic rows must never be written into the real bet ledger.
TEST_DATA_FILE = "totals_test_data.csv"


def add_totals_test_data(seed=42):
    rng = random.Random(seed)
    rows = []

    teams = [
        "Boston Celtics", "Denver Nuggets", "Milwaukee Bucks",
        "Phoenix Suns", "Miami Heat", "Dallas Mavericks",
        "Golden State Warriors", "Cleveland Cavaliers",
        "Minnesota Timberwolves", "Oklahoma City Thunder",
    ]

    for result in ["Win", "Loss"]:
        for _ in range(10):
            line = rng.randint(210, 235)
            actual = line - rng.randint(5, 20) if result == "Win" else line + rng.randint(5, 20)
            rows.append({
                "game_date": "2026-06-14",
                "home_team": rng.choice(teams),
                "away_team": rng.choice(teams),
                "market": "Totals",
                "pick_type": "Under",
                "projected_total": line - 10,
                "sportsbook_total": line,
                "edge": -10,
                "recommendation": "Strong Under",
                "stake": 100,
                "actual_total": actual,
                "result": result,
                "profit_loss": 91 if result == "Win" else -100,
                "source": "synthetic_test",
                "is_synthetic": True,
            })

    df = pd.DataFrame(rows)
    df.to_csv(TEST_DATA_FILE, index=False)

    return {
        "status": "success",
        "rows_added": len(df),
        "file": TEST_DATA_FILE,
        "note": "Synthetic rows are isolated from totals_history.csv and real learning data.",
    }


if __name__ == "__main__":
    print(add_totals_test_data())
