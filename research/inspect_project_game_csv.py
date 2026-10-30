"""Verify the repository's own data/game.csv schema for V4 Phase 1."""
from pathlib import Path
import json
import pandas as pd

path = Path("data/game.csv")
if not path.is_file():
    raise FileNotFoundError(path)
header = pd.read_csv(path, nrows=0)
columns = list(header.columns)
lookup = {str(c).strip().lower(): c for c in columns}
required_aliases = {
    "fga": ["fga", "field_goals_attempted"],
    "fta": ["fta", "free_throws_attempted"],
    "oreb": ["oreb", "offensive_rebounds"],
    "tov": ["tov", "turnovers"],
    "pts": ["pts", "points"],
    "game_id": ["game_id", "gameid"],
    "game_date": ["game_date", "game_date_est", "date"],
    "team_id": ["team_id", "teamid"],
}
resolved = {}
missing = []
for key, aliases in required_aliases.items():
    value = next((lookup[a] for a in aliases if a in lookup), None)
    if value is None:
        missing.append(key)
    else:
        resolved[key] = value
if missing:
    raise RuntimeError(f"data/game.csv is missing required V4 fields: {missing}; columns={columns}")
sample = pd.read_csv(path, usecols=list(dict.fromkeys(resolved.values())), nrows=100)
result = {
    "file": str(path),
    "columns_count": len(columns),
    "columns": columns,
    "resolved_columns": resolved,
    "sample_rows_read": int(len(sample)),
    "file_bytes": path.stat().st_size,
    "schema_check_passed": True,
}
print(json.dumps(result, indent=2, default=str))
