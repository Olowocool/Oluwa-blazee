from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/"
    "research/player-state-snapshots/latest.json.gz"
)
SNAPSHOT_MAX_AGE_HOURS = 36


def _history_rows(history):
    return [
        [pd.Timestamp(item[0]).date().isoformat(), str(item[1]), str(item[2]), float(item[3])]
        for item in history
    ]


def write_player_state_snapshot(state: dict, target_date, output: Path) -> dict:
    target = pd.Timestamp(target_date).normalize()
    team_history = [
        [str(team_id), str(player_id), _history_rows(rows)]
        for (team_id, player_id), rows in state["team_history"].items()
    ]
    player_history = {
        str(player_id): _history_rows(rows)
        for player_id, rows in state["player_history"].items()
    }
    first_seen = [
        [str(team_id), str(player_id), pd.Timestamp(date).date().isoformat()]
        for (team_id, player_id), date in state["first_seen"].items()
    ]
    payload = {
        "schema_version": "player-state-snapshot-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_target_date": str(target.date()),
        "diagnostics": state["diagnostics"],
        "games": state["games"].to_dict(orient="records"),
        "player_names": state["player_names"].to_dict(orient="records"),
        "team_names": state["team_names"].to_dict(orient="records"),
        "rapm": state["rapm"],
        "team_history": team_history,
        "player_history": player_history,
        "first_seen": first_seen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    output.write_bytes(gzip.compress(raw, compresslevel=6))
    return {
        "schema_version": payload["schema_version"],
        "generated_at_utc": payload["generated_at_utc"],
        "snapshot_target_date": payload["snapshot_target_date"],
        "compressed_bytes": output.stat().st_size,
        "diagnostics": state["diagnostics"],
    }


def load_player_state_snapshot(target_date, *, timeout: int = 60) -> dict:
    target = pd.Timestamp(target_date).normalize()
    response = requests.get(
        SNAPSHOT_URL,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = json.loads(gzip.decompress(response.content).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Player-state snapshot is unreadable: {type(exc).__name__}: {exc}") from exc

    if payload.get("schema_version") != "player-state-snapshot-v1":
        raise RuntimeError("Player-state snapshot schema is missing or unsupported.")
    generated = pd.Timestamp(payload["generated_at_utc"])
    if generated.tzinfo is None:
        raise RuntimeError("Player-state snapshot timestamp has no timezone.")
    age_hours = (pd.Timestamp.now(tz="UTC") - generated.tz_convert("UTC")).total_seconds() / 3600
    if age_hours < -1 or age_hours > SNAPSHOT_MAX_AGE_HOURS:
        raise RuntimeError(
            f"Player-state snapshot is stale ({age_hours:.1f} hours; "
            f"maximum {SNAPSHOT_MAX_AGE_HOURS}). Refresh is required."
        )
    snapshot_target = pd.Timestamp(payload["snapshot_target_date"]).normalize()
    if snapshot_target > target:
        raise RuntimeError(
            f"Player-state snapshot starts at {snapshot_target.date()}, "
            f"after selected date {target.date()}; a point-in-time snapshot is unavailable."
        )
    max_history = pd.Timestamp(payload["diagnostics"]["rapm_max_history_date"]).normalize()
    if max_history >= target:
        raise RuntimeError(
            f"Snapshot history reaches {max_history.date()}, not strictly before "
            f"selected date {target.date()}; refusing future-data leakage."
        )

    team_history = defaultdict(list)
    for team_id, player_id, rows in payload["team_history"]:
        team_history[(str(team_id), str(player_id))] = [
            (pd.Timestamp(row[0]), str(row[1]), str(row[2]), float(row[3]))
            for row in rows
        ]
    player_history = defaultdict(list)
    for player_id, rows in payload["player_history"].items():
        player_history[str(player_id)] = [
            (pd.Timestamp(row[0]), str(row[1]), str(row[2]), float(row[3]))
            for row in rows
        ]
    first_seen = {
        (str(team_id), str(player_id)): pd.Timestamp(date)
        for team_id, player_id, date in payload["first_seen"]
    }
    return {
        "games": pd.DataFrame(payload["games"], columns=["game_id"]),
        "player_names": pd.DataFrame(payload["player_names"]),
        "team_names": pd.DataFrame(payload["team_names"]),
        "rapm": payload["rapm"],
        "team_history": team_history,
        "player_history": player_history,
        "first_seen": first_seen,
        "diagnostics": {
            **payload["diagnostics"],
            "snapshot_generated_at_utc": payload["generated_at_utc"],
            "snapshot_age_hours": round(age_hours, 2),
            "snapshot_target_date": str(snapshot_target.date()),
            "snapshot_source": "GitHub Actions precomputed point-in-time state",
        },
    }
