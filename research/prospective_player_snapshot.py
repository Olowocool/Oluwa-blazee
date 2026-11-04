#!/usr/bin/env python3
"""Fail-closed point-in-time input adapter for prospective NBA totals research.

This module deliberately does not emit a player-adjusted total. It prepares and
freezes validated fixture/market/player inputs; a calibrated residual model is
required before a player-aware number can be published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SEASON = 2027
REQUIRED_ODDS = {
    "game_key", "game_date", "commence_time_utc", "away_team", "home_team",
    "bookmaker", "total_line", "over_odds_american", "under_odds_american",
    "published_as_of_utc", "observed_at_utc",
}
TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "ny knicks": "new york knicks",
    "gs warriors": "golden state warriors",
    "okc thunder": "oklahoma city thunder",
    "portland trail blazers": "portland trail blazers",
    "phoenix suns": "phoenix suns",
}


def utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone: {value!r}")
    return ts.tz_convert("UTC")


def norm_team(value: str) -> str:
    key = " ".join(str(value).strip().casefold().replace(".", "").split())
    return TEAM_ALIASES.get(key, key)


def map_odds_teams(away: str, home: str, team_seasons: pd.DataFrame) -> tuple[str, str]:
    required = {"team_id", "full_name"}
    if not required.issubset(team_seasons.columns):
        raise ValueError(f"team_seasons missing columns: {sorted(required-set(team_seasons.columns))}")
    lookup = {}
    for row in team_seasons.itertuples(index=False):
        key = norm_team(row.full_name)
        if key in lookup and lookup[key] != str(row.team_id):
            raise ValueError(f"Ambiguous NBA team name mapping: {row.full_name}")
        lookup[key] = str(row.team_id)
    a, h = norm_team(away), norm_team(home)
    if a not in lookup or h not in lookup or a == h:
        raise ValueError(f"Unmapped or duplicate matchup teams: away={away!r}, home={home!r}")
    return lookup[a], lookup[h]


def latest_draftkings_quotes(archive: pd.DataFrame, now_utc, max_observation_age_minutes=20) -> pd.DataFrame:
    missing = REQUIRED_ODDS - set(archive.columns)
    if missing:
        raise ValueError(f"Odds archive missing required fields: {sorted(missing)}")
    now = utc_timestamp(now_utc)
    d = archive.copy()
    d = d[d.bookmaker.astype(str).str.casefold().eq("draftkings")].copy()
    if d.empty:
        raise ValueError("No DraftKings rows in odds archive")
    for c in ("published_as_of_utc", "observed_at_utc", "commence_time_utc"):
        d[c] = d[c].map(utc_timestamp)
    for c in ("total_line", "over_odds_american", "under_odds_american"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d[["total_line", "over_odds_american", "under_odds_american"]].isna().any().any():
        raise ValueError("DraftKings market quote contains missing/non-numeric total or prices")
    if (d.total_line <= 0).any() or (d.over_odds_american == 0).any() or (d.under_odds_american == 0).any():
        raise ValueError("DraftKings quote contains invalid total or American price")
    d["observation_age_minutes"] = (now - d.observed_at_utc).dt.total_seconds() / 60
    publication_lag = (d.observed_at_utc - d.published_as_of_utc).dt.total_seconds() / 60
    if (publication_lag < -5).any():
        raise ValueError("Odds publication timestamp is implausibly later than its observation")
    d = d[d.observation_age_minutes.between(0, max_observation_age_minutes)].copy()
    d = d[d.commence_time_utc > now].copy()
    if d.empty:
        raise ValueError("No fresh DraftKings observations within configured age limit")
    # Keep the latest actual observation per fixture; never combine prices from rows.
    d = d.sort_values(["game_key", "observed_at_utc", "published_as_of_utc"])
    d = d.groupby("game_key", as_index=False, sort=False).tail(1)
    if d.game_key.duplicated().any():
        raise RuntimeError("Latest quote selection did not produce unique game keys")
    return d.reset_index(drop=True)


def validate_pregame_history(history: pd.DataFrame, target_date) -> pd.DataFrame:
    if "game_date" not in history:
        raise ValueError("Score history lacks game_date")
    d = history.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    target = pd.Timestamp(target_date).normalize()
    if d.game_date.isna().any():
        raise ValueError("Score history contains unparseable game dates")
    d = d[d.game_date < target].copy()
    if d.empty:
        raise ValueError(f"No completed score history strictly before {target.date()}")
    if not (d.game_date < target).all():
        raise RuntimeError("Same-day or future score history leaked into pregame history")
    return d


def validate_player_sources(games, team_logs, player_logs, rotations, events, target_date):
    target = pd.Timestamp(target_date).normalize()
    for label, frame in (("games", games), ("team_logs", team_logs),
                         ("player_logs", player_logs), ("rotations", rotations),
                         ("events", events)):
        if "game_date" not in frame:
            raise ValueError(f"{label} lacks game_date")
        dates = pd.to_datetime(frame.game_date, errors="coerce").dt.normalize()
        if dates.isna().any() or (dates >= target).any():
            raise ValueError(f"{label} contains same-day/future rows for target {target.date()}")
    if games.game_id.astype(str).duplicated().any():
        raise ValueError("Duplicate completed game IDs in source games")
    ids = set(games.game_id.astype(str))
    for label, frame in (("team_logs", team_logs), ("player_logs", player_logs),
                         ("rotations", rotations), ("events", events)):
        unknown = set(frame.game_id.astype(str)) - ids
        if unknown:
            raise ValueError(f"{label} contains rows for unknown games ({len(unknown)})")
    return True


def require_complete_injury_report(manifest: pd.DataFrame, target_date) -> dict:
    if manifest.empty or not {"game_date", "status", "snapshot_time_utc", "sha256"}.issubset(manifest.columns):
        raise ValueError("Official injury-report manifest is absent or incomplete")
    target = pd.Timestamp(target_date).date()
    rows = manifest[pd.to_datetime(manifest.game_date).dt.date.eq(target)]
    if len(rows) != 1 or str(rows.iloc[0].status).casefold() != "complete":
        raise ValueError(f"No single complete official injury report for {target}")
    row = rows.iloc[0]
    if not str(row.sha256).strip():
        raise ValueError("Injury report has no content hash")
    ts = utc_timestamp(row.snapshot_time_utc)
    if ts.date() > target:
        raise ValueError("Injury report timestamp is after target game date")
    return {"snapshot_time_utc": ts.isoformat(), "sha256": str(row.sha256)}


def immutable_snapshot(record: dict, output_dir: Path) -> Path:
    """Create one content-addressed JSON record; never overwrite an existing record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = output_dir / f"{digest}.json"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise RuntimeError("Content-address collision; refusing overwrite")
        return path
    # Exclusive create makes the append-only guarantee explicit.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds-archive", type=Path, default=Path("data/private/dunkapi_public_odds_snapshots.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/prospective_input_snapshots"))
    args = parser.parse_args()
    if not args.odds_archive.is_file():
        raise FileNotFoundError(f"Private odds archive not found: {args.odds_archive}")
    archive = pd.read_csv(args.odds_archive)
    now = datetime.now(timezone.utc)
    quotes = latest_draftkings_quotes(archive, now)
    records = []
    for row in quotes.to_dict("records"):
        record = {
            "schema_version": "prospective-input-v1",
            "created_at_utc": now.isoformat(),
            "fixture": {k: row[k] for k in ("game_key","game_date","commence_time_utc","away_team","home_team")},
            "market": {k: row[k] for k in ("bookmaker","total_line","over_odds_american","under_odds_american","published_as_of_utc","observed_at_utc","observation_age_minutes")},
            "team_ids": None,
            "forecast_status": "market_snapshot_only_player_inputs_not_yet_attached",
            "player_adjusted_total": None,
            "calibration_status": "not_fitted_no_player_adjustment_emitted",
        }
        path = immutable_snapshot(record, args.output_dir)
        records.append({"game_key": row["game_key"], "snapshot": str(path)})
    print(json.dumps({
        "status": "market_snapshots_saved",
        "snapshots": records,
        "player_adjusted_predictions_emitted": 0,
        "raw_odds_committed_to_git": False,
        "production_modified": False,
        "paper_ledger_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
