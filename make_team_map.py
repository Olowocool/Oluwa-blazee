import pandas as pd
import json

df = pd.read_parquet("outputs/training_dataset.parquet")

team_map = {}

for _, row in df.iterrows():
    team_map[int(row["home_team_id"])] = row["home_team_name"]
    team_map[int(row["away_team_id"])] = row["away_team_name"]

with open("team_map.json", "w") as f:
    json.dump(team_map, f)

print("Saved team_map.json")