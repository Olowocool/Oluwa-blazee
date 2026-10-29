"""Append-only, timestamped live NBA totals market snapshots.

The ordinary historical_odds.csv file is a latest-value UI cache and overwrites
older rows for a game/day. This ledger preserves each genuine API observation
so later paper-trade evaluation can reconstruct the line and prices seen before
tip-off. It never calls the provider or fabricates an observation timestamp.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from durable_totals_storage import (
    MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
    fetch_records,
    persist_records,
    sync_csv_mirror,
)

DEFAULT_OUTPUT_FILE = "data/live_totals_market_snapshots.csv"
SNAPSHOT_FIELDS = [
    "game_date",
    "display_game_date",
    "home_team",
    "away_team",
    "bookmaker",
    "snapshot_timestamp_utc",
    "market_last_update_utc",
    "commence_time_utc",
    "total_line",
    "over_odds",
    "under_odds",
    "snapshot_source",
    "source_event_id",
]


def _parse_aware_timestamp(value: Any, field: str, *, optional: bool = False):
    if value is None or not str(value).strip():
        if optional:
            return None
        raise ValueError(f"Missing {field}.")
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_positive(value: Any, *, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Expected a finite number greater than {minimum}.") from exc
    if not math.isfinite(number) or number <= minimum:
        raise ValueError(f"Expected a finite number greater than {minimum}.")
    return number


def _snapshot_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    teams = sorted([
        str(row.get("home_team", "")).strip().casefold(),
        str(row.get("away_team", "")).strip().casefold(),
    ])
    return (
        str(row.get("game_date", "")).strip(),
        "|".join(teams),
        str(row.get("bookmaker", "")).strip().casefold(),
        str(row.get("snapshot_timestamp_utc", "")).strip(),
    )


def _row_from_odds(game_date: str, home_team: str, away_team: str, odds: dict[str, Any]):
    if not home_team.strip() or not away_team.strip() or home_team.casefold() == away_team.casefold():
        raise ValueError("Missing or duplicated team names.")

    snapshot = _parse_aware_timestamp(odds.get("snapshot_timestamp_utc"), "snapshot_timestamp_utc")
    commence = _parse_aware_timestamp(odds.get("commence_time_utc"), "commence_time_utc")
    market_update = _parse_aware_timestamp(
        odds.get("market_last_update_utc"), "market_last_update_utc", optional=True
    )
    if snapshot >= commence:
        raise ValueError("Post-tipoff market snapshots are not eligible.")
    if market_update is not None and (market_update > snapshot or market_update >= commence):
        raise ValueError("Market last-update timestamp is inconsistent with snapshot/tip-off.")

    line = _finite_positive(odds.get("total_line"), minimum=0.0)
    over_price = _finite_positive(odds.get("over_odds"), minimum=1.0)
    under_price = _finite_positive(odds.get("under_odds"), minimum=1.0)
    if not str(odds.get("totals_bookmaker") or "").strip():
        raise ValueError("A named bookmaker is required for a market snapshot.")

    over_point = odds.get("over_total_line", odds.get("total_line"))
    under_point = odds.get("under_total_line", odds.get("total_line"))
    if _finite_positive(over_point, minimum=0.0) != line or _finite_positive(under_point, minimum=0.0) != line:
        raise ValueError("Over and Under prices do not share the recorded total line.")

    # Match the historical odds archive's Eastern calendar date while also
    # preserving the app's selected/display date (Nigeria WAT) for auditing.
    eastern_game_date = commence.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    return {
        "game_date": eastern_game_date,
        "display_game_date": str(game_date).strip(),
        "home_team": str(odds.get("market_home_team") or home_team).strip(),
        "away_team": str(odds.get("market_away_team") or away_team).strip(),
        "bookmaker": str(odds["totals_bookmaker"]).strip(),
        "snapshot_timestamp_utc": _utc_iso(snapshot),
        "market_last_update_utc": _utc_iso(market_update) if market_update is not None else "",
        "commence_time_utc": _utc_iso(commence),
        "total_line": line,
        "over_odds": over_price,
        "under_odds": under_price,
        "snapshot_source": str(odds.get("snapshot_source") or "the-odds-api-v4-current-endpoint"),
        "source_event_id": str(odds.get("source_event_id") or ""),
    }


def append_live_totals_market_snapshots(
    game_date: str,
    odds_map: dict,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, Any]:
    """Append valid pre-tipoff snapshots without erasing earlier observations.

    Duplicate calls with the same game/book/snapshot timestamp are idempotent.
    Conflicting prices at the same identity fail closed instead of rewriting history.
    """
    if not isinstance(odds_map, dict):
        return {
            "status": "error",
            "message": "odds_map must be a dictionary.",
            "rows_seen": 0,
            "rows_added": 0,
            "invalid_rows_skipped": 0,
            "post_tipoff_rows_skipped": 0,
        }

    path = Path(output_file)
    database_active = sync_csv_mirror(
        "market_snapshot",
        path,
        fieldnames=SNAPSHOT_FIELDS,
    )

    seen = len(odds_map)
    candidates = []
    invalid = post_tipoff = 0
    for key, odds in odds_map.items():
        if not isinstance(key, tuple) or len(key) != 2 or not isinstance(odds, dict):
            invalid += 1
            continue
        home_team, away_team = str(key[0]), str(key[1])
        try:
            candidates.append(_row_from_odds(game_date, home_team, away_team, odds))
        except ValueError as exc:
            if "Post-tipoff" in str(exc):
                post_tipoff += 1
            else:
                invalid += 1

    existing: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(SNAPSHOT_FIELDS).difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Existing market snapshot ledger is missing columns: {sorted(missing)}")
            existing = list(reader)

    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in existing:
        merged[_snapshot_identity(row)] = row

    added = 0
    for row in candidates:
        identity = _snapshot_identity(row)
        old = merged.get(identity)
        if old is not None:
            for field in ("total_line", "over_odds", "under_odds"):
                try:
                    equal = math.isclose(float(old[field]), float(row[field]), rel_tol=0, abs_tol=1e-9)
                except (TypeError, ValueError):
                    equal = str(old.get(field, "")).strip() == str(row.get(field, "")).strip()
                if not equal:
                    raise ValueError(
                        "Conflicting market data for the same game/bookmaker/snapshot timestamp; "
                        "the existing ledger was not modified."
                    )
            continue
        merged[identity] = row
        added += 1

    # Do not create or rewrite a file when this poll produced no new valid
    # snapshots; in particular, an in-play-only poll must not create an empty ledger.
    if added == 0:
        return {
            "status": "success",
            "path": str(path),
            "rows_seen": seen,
            "rows_added": 0,
            "invalid_rows_skipped": invalid,
            "post_tipoff_rows_skipped": post_tipoff,
            "total_rows": len(merged),
        }

    if database_active:
        # PostgreSQL is authoritative when configured; persist first so a failed
        # local mirror write cannot silently lose an observed market snapshot.
        db_result = persist_records(
            "market_snapshot",
            candidates,
            immutable=True,
            conflict_fields=MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
        )
        database_rows = fetch_records("market_snapshot")
        merged = {_snapshot_identity(row): row for row in database_rows}
        added = db_result["inserted"]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            tmp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(
                merged.values(),
                key=lambda r: (
                    str(r.get("game_date", "")),
                    str(r.get("snapshot_timestamp_utc", "")),
                    str(r.get("home_team", "")),
                    str(r.get("away_team", "")),
                    str(r.get("bookmaker", "")),
                ),
            ):
                writer.writerow({field: row.get(field, "") for field in SNAPSHOT_FIELDS})
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "status": "success",
        "path": str(path),
        "rows_seen": seen,
        "rows_added": added,
        "invalid_rows_skipped": invalid,
        "post_tipoff_rows_skipped": post_tipoff,
        "total_rows": len(merged),
    }
