"""Inspect the repository's game.csv schema and season/date conventions."""
from pathlib import Path
import json
import pandas as pd

path = Path("data/game.csv")
if not path.is_file():
    raise FileNotFoundError(path)
header = pd.read_csv(path, nrows=0)
columns = list(header.columns)
sample = pd.read_csv(path, nrows=5)
meta_cols = [c for c in ("season_id", "game_date", "season_type", "game_id") if c in sample.columns]
usecols = [c for c in ("season_id", "game_date", "season_type") if c in columns]
meta = pd.read_csv(path, usecols=usecols, low_memory=False) if usecols else pd.DataFrame()
result = {
    "file": str(path),
    "columns_count": len(columns),
    "columns": columns,
    "sample_rows": sample[meta_cols + [c for c in ("team_id_home", "team_name_home", "team_id_away", "team_name_away", "fga_home", "fta_home", "oreb_home", "tov_home", "pts_home", "fga_away", "fta_away", "oreb_away", "tov_away", "pts_away") if c in sample.columns]].to_dict(orient="records"),
    "file_bytes": path.stat().st_size,
    "row_count": int(len(meta)),
    "season_id_values": sorted(meta.season_id.dropna().astype(str).unique().tolist()) if "season_id" in meta else [],
    "season_type_counts": meta.season_type.value_counts(dropna=False).to_dict() if "season_type" in meta else {},
    "first_game_date": str(pd.to_datetime(meta.game_date, errors="coerce").min()) if "game_date" in meta else None,
    "last_game_date": str(pd.to_datetime(meta.game_date, errors="coerce").max()) if "game_date" in meta else None,
}
print(json.dumps(result, indent=2, default=str))
