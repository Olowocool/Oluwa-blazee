import os
import random
import pandas as pd
from datetime import datetime


OUTPUT_FILE = "bet_history.csv"
STAKE = 100


NBA_TEAMS = [
    "Boston Celtics",
    "Denver Nuggets",
    "Oklahoma City Thunder",
    "Minnesota Timberwolves",
    "Cleveland Cavaliers",
    "San Antonio Spurs",
    "Detroit Pistons",
    "New York Knicks",
    "Los Angeles Lakers",
    "Golden State Warriors",
    "Milwaukee Bucks",
    "Dallas Mavericks",
]


def calculate_profit_loss(result, odds, stake=STAKE):
    if result == "Win":
        return (float(odds) - 1) * stake

    if result == "Loss":
        return -stake

    return 0


def generate_historical_backfill(rows=500):
    generated_rows = []

    for i in range(rows):
        home_team = random.choice(NBA_TEAMS)
        away_team = random.choice([team for team in NBA_TEAMS if team != home_team])

        selected_team = random.choice([home_team, away_team])

        odds = round(random.uniform(1.55, 2.80), 2)
        model_probability = round(random.uniform(0.42, 0.68), 4)

        implied_probability = 1 / odds
        expected_value = round(
            (model_probability * (odds - 1)) - (1 - model_probability),
            4
        )

        kelly = round(
            max(((odds - 1) * model_probability - (1 - model_probability)) / (odds - 1), 0),
            4
        )

        result = "Win" if random.random() < model_probability else "Loss"

        generated_rows.append({
            "timestamp": datetime.now().isoformat(),
            "game_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "home_team": home_team,
            "away_team": away_team,
            "best_bet": selected_team,
            "odds": odds,
            "model_probability": model_probability,
            "expected_value": expected_value,
            "kelly": kelly,
            "stake": STAKE,
            "result": result,
            "profit_loss": calculate_profit_loss(result, odds),
            "closing_odds": "",
            "clv": "",
            "source": "historical_backfill_simulation"
        })

    new_df = pd.DataFrame(generated_rows)

    if os.path.isfile(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(OUTPUT_FILE, index=False)

    return {
        "status": "success",
        "message": f"{rows} historical rows added to bet_history.csv.",
        "added_rows": rows,
        "total_rows": len(final_df)
    }
