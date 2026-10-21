
import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

TRAINING_DATA_PATH = "outputs/training_dataset.parquet"
OUTPUT_PATH = "data/historical_nba_scores.csv"

REQUIRED_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]


def load_existing_history():
    """Load and normalize the existing historical training data."""
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(
            f"Existing training data not found: {TRAINING_DATA_PATH}"
        )

    df = pd.read_parquet(TRAINING_DATA_PATH)

    rename_map = {
        "date": "game_date",
        "home_team_name": "home_team",
        "away_team_name": "away_team",
        "home_team_score": "home_score",
        "away_team_score": "away_score",
        "home_points": "home_score",
        "away_points": "away_score",
    }

    for old, new in rename_map.items():
        if new not in df.columns and old in df.columns:
            df = df.rename(columns={old: new})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Training dataset is missing required columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["home_score"] != df["away_score"]]

    return df


def download_real_nba_games():
    """Fetch completed NBA games and reconstruct home/away scores."""
    frames = []

    # Query each season individually so older seasons are not missed
    # because an endpoint returns only a limited recent result set.
    for year in range(1947, 2027):
        season = f"{year}-{str(year + 1)[-2:]}"

        try:
            games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
                timeout=30,
            ).get_data_frames()[0]

            if games.empty:
                continue

            required = [
                "GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"
            ]
            if not all(col in games.columns for col in required):
                continue

            games = games[required].copy()
            games["PTS"] = pd.to_numeric(games["PTS"], errors="coerce")
            games["game_date"] = pd.to_datetime(
                games["GAME_DATE"], errors="coerce"
            )

            for game_id, group in games.groupby("GAME_ID"):
                if len(group) != 2:
                    continue

                home = group[
                    group["MATCHUP"].astype(str).str.contains(
                        "vs.", regex=False
                    )
                ]
                away = group[
                    group["MATCHUP"].astype(str).str.contains(
                        "@", regex=False
                    )
                ]

                if len(home) != 1 or len(away) != 1:
                    continue

                home_row = home.iloc[0]
                away_row = away.iloc[0]

                if pd.isna(home_row["PTS"]) or pd.isna(away_row["PTS"]):
                    continue

                if home_row["PTS"] == away_row["PTS"]:
                    continue

                frames.append({
                    "game_date": home_row["game_date"],
                    "home_team": home_row["TEAM_NAME"],
                    "away_team": away_row["TEAM_NAME"],
                    "home_score": float(home_row["PTS"]),
                    "away_score": float(away_row["PTS"]),
                })

        except Exception as exc:
            print(f"Season {season} could not be downloaded: {exc}")

    if not frames:
        raise RuntimeError(
            "No games were downloaded. Existing history was not changed."
        )

    return pd.DataFrame(frames)


def main():
    os.makedirs("data", exist_ok=True)

    existing = load_existing_history()
    print(f"Existing valid history rows: {len(existing)}")

    downloaded = download_real_nba_games()
    print(f"Real NBA games downloaded: {len(downloaded)}")

    combined = pd.concat([existing, downloaded], ignore_index=True)

    combined["home_team"] = combined["home_team"].astype(str).str.strip()
    combined["away_team"] = combined["away_team"].astype(str).str.strip()
    combined["game_date"] = pd.to_datetime(
        combined["game_date"], errors="coerce"
    )

    combined = combined.dropna(subset=REQUIRED_COLUMNS)

    # Deduplicate using date and the two teams, independent of source.
    combined["_dedupe_key"] = (
        combined["game_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + combined["home_team"].str.casefold()
        + "|"
        + combined["away_team"].str.casefold()
    )

    combined = (
        combined.sort_values("game_date")
        .drop_duplicates("_dedupe_key", keep="last")
        .drop(columns="_dedupe_key")
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    # Write to a temporary file first; replace the output only after
    # the combined dataset has been built successfully.
    temp_path = OUTPUT_PATH + ".tmp"
    combined.to_csv(temp_path, index=False)
    os.replace(temp_path, OUTPUT_PATH)

    print(f"Combined history saved: {OUTPUT_PATH}")
    print(f"Total unique games: {len(combined)}")
    print(f"Earliest game: {combined['game_date'].min()}")
    print(f"Latest game: {combined['game_date'].max()}")
    print("Existing Parquet training data was not modified.")


if __name__ == "__main__":
    main()
