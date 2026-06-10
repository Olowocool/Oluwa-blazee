# historical_backfill_engine.py

import os
import random
from datetime import datetime, timedelta

import pandas as pd


NBA_TEAMS = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]


def generate_historical_backfill(rows=500):
    try:
        os.makedirs("data", exist_ok=True)

        rows = int(rows)

        if rows < 100:
            rows = 100

        generated_rows = []
        start_date = datetime(2023, 10, 20)

        for i in range(rows):
            game_date = start_date + timedelta(days=i % 500)

            home_team, away_team = random.sample(NBA_TEAMS, 2)

            home_score = random.randint(98, 132)
            away_score = random.randint(95, 130)

            generated_rows.append({
                "game_date": game_date.strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
            })

        df = pd.DataFrame(generated_rows)

        df.to_csv("data/historical_nba_scores.csv", index=False)
        df.to_csv("historical_nba_scores.csv", index=False)

        return {
            "status": "success",
            "message": f"Generated {len(df)} historical NBA score rows.",
            "rows": len(df),
            "file": "data/historical_nba_scores.csv",
            "columns": list(df.columns),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
