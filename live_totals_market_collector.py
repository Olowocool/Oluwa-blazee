"""Scheduled DraftKings NBA totals market collector.

This process captures provider observations only. It does not generate picks,
change the Points Engine, or settle wagers. Durable PostgreSQL is mandatory so
a scheduled run can never silently write to ephemeral local CSV storage.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from durable_totals_storage import durable_storage_required, test_database_connection
from live_totals_market_snapshots import append_live_totals_market_snapshots

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ODDS_API_MARKET = "totals"
ODDS_API_REGION = "us"
TARGET_BOOKMAKER_KEY = "draftkings"
SNAPSHOT_SOURCE = "the-odds-api-v4-current-endpoint"
REQUEST_TIMEOUT_SECONDS = 30


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def build_market_snapshot_map(
    events: list[dict[str, Any]],
    snapshot_timestamp_utc: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep only complete, single-line DraftKings Over/Under pregame markets."""
    if not _is_aware_timestamp(snapshot_timestamp_utc):
        raise ValueError("Snapshot timestamp must be timezone-aware.")
    if not isinstance(events, list):
        raise ValueError("Odds API response must be a list of events.")

    snapshot_time = datetime.fromisoformat(
        snapshot_timestamp_utc.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    odds_map: dict[tuple[str, str], dict[str, Any]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id") or "").strip()
        home = str(event.get("home_team") or "").strip()
        away = str(event.get("away_team") or "").strip()
        commence = str(event.get("commence_time") or "").strip()
        if not event_id or not home or not away or home.casefold() == away.casefold():
            continue
        if not _is_aware_timestamp(commence):
            continue
        commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        if commence_dt.astimezone(timezone.utc) <= snapshot_time:
            continue

        selected: dict[str, Any] | None = None
        for bookmaker in event.get("bookmakers", []):
            if not isinstance(bookmaker, dict):
                continue
            if str(bookmaker.get("key") or "").strip().casefold() != TARGET_BOOKMAKER_KEY:
                continue
            for market in bookmaker.get("markets", []):
                if not isinstance(market, dict) or market.get("key") != "totals":
                    continue
                outcomes = market.get("outcomes", [])
                if not isinstance(outcomes, list):
                    continue
                overs = [
                    x for x in outcomes
                    if isinstance(x, dict)
                    and str(x.get("name") or "").strip().casefold() == "over"
                ]
                unders = [
                    x for x in outcomes
                    if isinstance(x, dict)
                    and str(x.get("name") or "").strip().casefold() == "under"
                ]
                if len(overs) != 1 or len(unders) != 1:
                    continue

                over_line = _finite_float(overs[0].get("point"))
                under_line = _finite_float(unders[0].get("point"))
                over_price = _finite_float(overs[0].get("price"))
                under_price = _finite_float(unders[0].get("price"))
                if (
                    over_line is None
                    or under_line is None
                    or over_price is None
                    or under_price is None
                    or over_line <= 0
                    or over_line != under_line
                    or over_price <= 1.0
                    or under_price <= 1.0
                ):
                    continue

                market_update = str(
                    market.get("last_update") or bookmaker.get("last_update") or ""
                ).strip()
                if market_update and not _is_aware_timestamp(market_update):
                    continue

                selected = {
                    "snapshot_timestamp_utc": snapshot_timestamp_utc,
                    "market_last_update_utc": market_update,
                    "commence_time_utc": commence,
                    "total_line": over_line,
                    "over_total_line": over_line,
                    "under_total_line": under_line,
                    "over_odds": over_price,
                    "under_odds": under_price,
                    "totals_bookmaker": str(bookmaker.get("title") or "DraftKings").strip(),
                    "snapshot_source": SNAPSHOT_SOURCE,
                    "source_event_id": event_id,
                    "market_home_team": home,
                    "market_away_team": away,
                }
                break
            if selected is not None:
                break

        if selected is not None:
            odds_map[(home.lower(), away.lower())] = selected

    return odds_map


def collect_once(
    *,
    session: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch one live totals response and persist pre-tipoff DraftKings snapshots."""
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not configured for the collector.")

    if not durable_storage_required():
        raise RuntimeError("Durable storage is not required; refusing a CSV-only collector run.")
    database_check = test_database_connection()
    if not database_check.get("ok"):
        raise RuntimeError(str(database_check.get("message") or "PostgreSQL connection failed."))

    client = session or requests
    response = client.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key,
            "regions": ODDS_API_REGION,
            "markets": ODDS_API_MARKET,
            "oddsFormat": "decimal",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    events = response.json()
    captured_at = now or datetime.now(timezone.utc)
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("Collector clock must be timezone-aware.")
    captured_text = captured_at.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")

    odds_map = build_market_snapshot_map(events, captured_text)
    display_date = captured_at.astimezone(ZoneInfo("Africa/Lagos")).strftime("%m/%d/%Y")
    saved = append_live_totals_market_snapshots(display_date, odds_map)

    return {
        "status": saved.get("status", "error"),
        "captured_at_utc": captured_text,
        "events_returned": len(events) if isinstance(events, list) else 0,
        "draftkings_totals_events": len(odds_map),
        "rows_added": int(saved.get("rows_added", 0)),
        "invalid_rows_skipped": int(saved.get("invalid_rows_skipped", 0)),
        "post_tipoff_rows_skipped": int(saved.get("post_tipoff_rows_skipped", 0)),
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "requests_used": response.headers.get("x-requests-used"),
        "requests_last": response.headers.get("x-requests-last"),
    }


def main() -> int:
    try:
        result = collect_once()
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "success" else 1
    except Exception as exc:
        # Keep API credentials and database connection strings out of logs.
        print(f"NBA totals collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
