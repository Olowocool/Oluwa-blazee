
import os
import time
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

# Existing history ends in April 2023.
# Download the remaining 2022-23 season and subsequent seasons.
SEASONS_TO_UPDATE = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def load_existing_history():
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_DATA_PATH}"
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
        raise ValueError(f"Missing columns in training dataset: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["home_score"] != df["away_score"]]

    if df.empty:
        raise ValueError("Existing training history contains no valid games.")

    return df


def download_season(season):
    print(f"Downloading NBA regular-season games: {season}")

    games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        timeout=45,
    ).get_data_frames()[0]

    if games.empty:
        raise RuntimeError(
            f"No data returned for {season}. Update cancelled; "
            "existing historical data will not be overwritten."
        )

    required = ["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"]
    missing = [col for col in required if col not in games.columns]

    if missing:
        raise RuntimeError(
            f"NBA API response for {season} is missing: {missing}"
        )

    games = games[required].copy()
    games["PTS"] = pd.to_numeric(games["PTS"], errors="coerce")
    games["game_date"] = pd.to_datetime(
        games["GAME_DATE"], errors="coerce"
    )

    season_rows = []

    for game_id, group in games.groupby("GAME_ID"):
        if len(group) != 2:
            continue

        matchup = group["MATCHUP"].astype(str)
        home = group[matchup.str.contains("vs.", regex=False)]
        away = group[matchup.str.contains("@", regex=False)]

        if len(home) != 1 or len(away) != 1:
            continue

        home_row = home.iloc[0]
        away_row = away.iloc[0]

        if (
            pd.isna(home_row["game_date"])
            or pd.isna(home_row["PTS"])
            or pd.isna(away_row["PTS"])
        ):
            continue

        # Exclude games without a final, non-tied score.
        if home_row["PTS"] == away_row["PTS"]:
            continue

        season_rows.append({
            "game_date": home_row["game_date"],
            "home_team": str(home_row["TEAM_NAME"]).strip(),
            "away_team": str(away_row["TEAM_NAME"]).strip(),
            "home_score": float(home_row["PTS"]),
            "away_score": float(away_row["PTS"]),
        })

    result = pd.DataFrame(season_rows, columns=REQUIRED_COLUMNS)

    if result.empty:
        raise RuntimeError(
            f"No valid completed games reconstructed for {season}. "
            "Update cancelled."
        )

    print(f"  Valid completed games: {len(result)}")
    return result


def main():
    existing = load_existing_history()
    print(f"Existing historical rows: {len(existing)}")
    print(f"Existing latest date: {existing['game_date'].max()}")

    downloaded_seasons = []

    # If any request fails, execution stops before writing the output.
    for season in SEASONS_TO_UPDATE:
        season_data = download_season(season)
        downloaded_seasons.append(season_data)
        time.sleep(1)

    downloaded = pd.concat(downloaded_seasons, ignore_index=True)

    combined = pd.concat([existing, downloaded], ignore_index=True)
    combined["game_date"] = pd.to_datetime(
        combined["game_date"], errors="coerce"
    )

    for col in ["home_team", "away_team"]:
        combined[col] = combined[col].astype(str).str.strip()

    for col in ["home_score", "away_score"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna(subset=REQUIRED_COLUMNS)
    combined = combined[
        combined["home_team"].ne("")
        & combined["away_team"].ne("")
        & combined["home_team"].ne("nan")
        & combined["away_team"].ne("nan")
        & combined["home_score"].ne(combined["away_score"])
    ].copy()

    # Normalize team-name case for duplicate detection.
    combined["_key"] = (
        combined["game_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + combined["home_team"].str.casefold()
        + "|"
        + combined["away_team"].str.casefold()
    )

    combined = (
        combined.sort_values("game_date")
        .drop_duplicates("_key", keep="last")
        .drop(columns="_key")
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    if len(combined) < len(existing):
        raise RuntimeError(
            "Safety check failed: combined history is smaller than "
            "the existing dataset. No output was written."
        )

    latest_date = combined["game_date"].max()

    # Confirm that the current season's known opening games were included.
    current_season = downloaded[
        downloaded["game_date"] >= pd.Timestamp("2026-10-01")
    ]

    if current_season.empty:
        raise RuntimeError(
            "Safety check failed: no October 2026 games found. "
            "No output was written."
        )

    os.makedirs("data", exist_ok=True)
    temp_path = OUTPUT_PATH + ".tmp"

    combined[REQUIRED_COLUMNS].to_csv(temp_path, index=False)
    os.replace(temp_path, OUTPUT_PATH)

    print("\nUPDATE SUCCESSFUL")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Previous valid rows: {len(existing)}")
    print(f"New combined rows: {len(combined)}")
    print(f"Earliest game: {combined['game_date'].min()}")
    print(f"Latest game: {latest_date}")
    print("Original Parquet dataset was not modified.")


if __name__ == "__main__":
    main()
