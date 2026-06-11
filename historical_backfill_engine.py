# historical_backfill_engine.py

import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def generate_historical_backfill(rows=500):
    try:
        os.makedirs("data", exist_ok=True)

        rows = int(rows)

        games_df = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00"
        ).get_data_frames()[0]

        if games_df.empty:
            return {
                "status": "error",
                "message": "NBA API returned no games.",
            }

        required_cols = [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_NAME",
            "MATCHUP",
            "PTS",
        ]

        for col in required_cols:
            if col not in games_df.columns:
                return {
                    "status": "error",
                    "message": f"NBA API data missing column: {col}",
                    "columns": list(games_df.columns),
                }

        games_df = games_df[required_cols].copy()

        games_df["GAME_DATE"] = pd.to_datetime(
            games_df["GAME_DATE"],
            errors="coerce"
        )

        games_df = games_df.dropna(subset=["GAME_DATE"])

        completed_games = []

        for game_id, group in games_df.groupby("GAME_ID"):
            if len(group) != 2:
                continue

            row1 = group.iloc[0]
            row2 = group.iloc[1]

            matchup1 = str(row1["MATCHUP"])

            if " vs. " in matchup1:
                home_row = row1
                away_row = row2
            elif " @ " in matchup1:
                away_row = row1
                home_row = row2
            else:
                continue

            try:
                home_score = int(home_row["PTS"])
                away_score = int(away_row["PTS"])
            except Exception:
                continue

            completed_games.append({
                "game_date": home_row["GAME_DATE"].strftime("%Y-%m-%d"),
                "home_team": home_row["TEAM_NAME"],
                "away_team": away_row["TEAM_NAME"],
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
                "game_id": str(game_id),
            })

        historical_df = pd.DataFrame(completed_games)

        if historical_df.empty:
            return {
                "status": "error",
                "message": "No completed games could be converted.",
            }

        historical_df = historical_df.sort_values(
            "game_date",
            ascending=False
        )

        historical_df = historical_df.head(rows)

        historical_df.to_csv(
            "data/historical_nba_scores.csv",
            index=False
        )

        historical_df.to_csv(
            "historical_nba_scores.csv",
            index=False
        )

        return {
            "status": "success",
            "message": f"Generated {len(historical_df)} real NBA historical score rows.",
            "rows": len(historical_df),
            "file": "data/historical_nba_scores.csv",
            "columns": list(historical_df.columns),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
