import pandas as pd

df = pd.read_csv("data/game.csv")

# Use only regular season games
if "season_type" in df.columns:
    df = df[df["season_type"] == "Regular Season"].copy()

# Clean dates
df["game_date"] = pd.to_datetime(df["game_date"])

# Build model-ready dataset
out = pd.DataFrame({
    "date": df["game_date"],
    "season": df["season_id"],
    "home_team_id": df["team_id_home"],
    "away_team_id": df["team_id_away"],
    "home_team_name": df["team_name_home"],
    "away_team_name": df["team_name_away"],
    "home_score": df["pts_home"],
    "away_score": df["pts_away"],
})

# Remove bad rows
out = out.dropna()
out = out[out["home_score"] != out["away_score"]]

# Save for model training
out.to_csv("data/nba_games.csv", index=False)

print("Saved real NBA training file to data/nba_games.csv")
print(out.head())
print("Rows:", len(out))