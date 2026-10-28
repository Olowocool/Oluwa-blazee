"""Collect real historical NBA totals snapshots from The Odds API.

Historical odds require a compatible paid provider plan. This script never
fabricates prices and never calls the provider unless the caller explicitly
confirms a per-run API-credit budget.

Example (preview only):
  python collect_historical_totals_odds.py --start-date 2025-10-21 \
    --end-date 2025-10-25 --snapshot-times 16:00,18:00,20:00,22:00 --plan-only

Example (real collection; review the estimate first):
  python collect_historical_totals_odds.py --start-date 2025-10-21 \
    --end-date 2025-10-25 --snapshot-times 16:00,18:00,20:00,22:00 \
    --max-api-credits 2000 --confirm-api-cost

Times are America/New_York wall-clock times. The saved game_date is also
America/New_York's calendar date, while timestamps are UTC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
DEFAULT_OUTPUT_FILE = "data/historical_totals_market_snapshots.csv"
LOCAL_TIMEZONE = "America/New_York"
EXPECTED_CREDITS_PER_REQUEST = 10  # US region x featured totals market per provider docs.
CSV_FIELDS = [
    "game_date",
    "home_team",
    "away_team",
    "bookmaker",
    "snapshot_timestamp_utc",
    "commence_time_utc",
    "total_line",
    "over_odds",
    "under_odds",
    "snapshot_source",
    "source_event_id",
    "bookmaker_key",
    "requested_snapshot_utc",
    "market_last_update_utc",
]


def _parse_aware_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Historical odds response is missing {field}.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_snapshot_times(value: str) -> list[time]:
    """Parse a comma-separated list of unique HH:MM Eastern local times."""
    if not value or not value.strip():
        raise ValueError("Provide at least one --snapshot-times value, such as 16:00,18:00,20:00.")
    times: list[time] = []
    for part in value.split(","):
        token = part.strip()
        try:
            hour_text, minute_text = token.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
            parsed = time(hour=hour, minute=minute)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid snapshot time {token!r}; expected HH:MM.") from exc
        if parsed in times:
            raise ValueError(f"Duplicate snapshot time: {token}")
        times.append(parsed)
    return sorted(times)


def build_requested_snapshots(start_date: str, end_date: str, snapshot_times: list[time]) -> list[datetime]:
    """Create timestamped request targets for inclusive Eastern calendar dates."""
    try:
        first = date.fromisoformat(start_date)
        last = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format.") from exc
    if last < first:
        raise ValueError("--end-date must be on or after --start-date.")
    if not snapshot_times:
        raise ValueError("At least one snapshot time is required.")
    zone = ZoneInfo(LOCAL_TIMEZONE)
    requested: list[datetime] = []
    day = first
    while day <= last:
        for clock in snapshot_times:
            local = datetime.combine(day, clock).replace(tzinfo=zone)
            requested.append(local.astimezone(timezone.utc))
        day += timedelta(days=1)
    return requested


def estimate_api_credits(request_count: int) -> int:
    if request_count < 0:
        raise ValueError("request_count cannot be negative.")
    return request_count * EXPECTED_CREDITS_PER_REQUEST


def _valid_decimal_price(value: Any) -> bool:
    try:
        return float(value) > 1.0
    except (TypeError, ValueError):
        return False


def parse_snapshot_response(
    payload: dict[str, Any],
    requested_snapshot_utc: datetime,
    bookmaker_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert an actual historical API response to the backtest CSV schema."""
    snapshot_timestamp = _parse_aware_datetime(payload.get("timestamp"), "timestamp")
    events = payload.get("data")
    if not isinstance(events, list):
        raise ValueError("Historical odds response field 'data' must be a list.")

    zone = ZoneInfo(LOCAL_TIMEZONE)
    rows: list[dict[str, Any]] = []
    events_seen = 0
    post_tipoff_events = 0
    events_without_totals = 0
    invalid_market_pairs = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        events_seen += 1
        try:
            commence = _parse_aware_datetime(event.get("commence_time"), "commence_time")
        except ValueError:
            continue
        # A current/in-play price is never valid for a pregame backtest.
        if snapshot_timestamp >= commence:
            post_tipoff_events += 1
            continue

        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        if not home_team or not away_team or home_team.casefold() == away_team.casefold():
            continue

        team_date = commence.astimezone(zone).date().isoformat()
        event_id = str(event.get("id") or "")
        for book in event.get("bookmakers") or []:
            if not isinstance(book, dict):
                continue
            book_key = str(book.get("key") or "").strip()
            book_title = str(book.get("title") or book_key).strip()
            if bookmaker_key and book_key.casefold() != bookmaker_key.strip().casefold():
                continue

            for market in book.get("markets") or []:
                if not isinstance(market, dict) or market.get("key") != "totals":
                    continue
                outcomes = market.get("outcomes") or []
                overs = [o for o in outcomes if isinstance(o, dict) and str(o.get("name", "")).casefold() == "over"]
                unders = [o for o in outcomes if isinstance(o, dict) and str(o.get("name", "")).casefold() == "under"]
                # The main totals market should contain exactly one matching pair.
                # Skip ambiguous market structures rather than selecting a price opportunistically.
                if len(overs) != 1 or len(unders) != 1:
                    invalid_market_pairs += 1
                    continue
                over, under = overs[0], unders[0]
                try:
                    over_point = float(over.get("point"))
                    under_point = float(under.get("point"))
                    over_price = float(over.get("price"))
                    under_price = float(under.get("price"))
                except (TypeError, ValueError):
                    invalid_market_pairs += 1
                    continue
                if (
                    over_point != under_point
                    or over_point <= 0
                    or not _valid_decimal_price(over_price)
                    or not _valid_decimal_price(under_price)
                ):
                    invalid_market_pairs += 1
                    continue

                last_update = market.get("last_update") or book.get("last_update")
                if last_update:
                    try:
                        last_update_value = _utc_iso(_parse_aware_datetime(last_update, "market last_update"))
                    except ValueError:
                        last_update_value = ""
                else:
                    last_update_value = ""

                rows.append({
                    "game_date": team_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": book_title,
                    "snapshot_timestamp_utc": _utc_iso(snapshot_timestamp),
                    "commence_time_utc": _utc_iso(commence),
                    "total_line": over_point,
                    "over_odds": over_price,
                    "under_odds": under_price,
                    "snapshot_source": "the-odds-api-v4-historical",
                    "source_event_id": event_id,
                    "bookmaker_key": book_key,
                    "requested_snapshot_utc": _utc_iso(requested_snapshot_utc),
                    "market_last_update_utc": last_update_value,
                })

    return rows, {
        "events_seen": events_seen,
        "post_tipoff_events_excluded": post_tipoff_events,
        "events_without_totals": max(0, events_seen - post_tipoff_events - len({r["source_event_id"] for r in rows})),
        "invalid_market_pairs_skipped": invalid_market_pairs,
        "pregame_market_rows": len(rows),
    }


def _read_archive(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        required = {"game_date", "home_team", "away_team", "bookmaker", "snapshot_timestamp_utc", "commence_time_utc", "total_line", "over_odds", "under_odds"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Existing archive is missing required fields: {sorted(missing)}")
        return list(reader)


def _snapshot_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    # Match the same game/bookmaker/snapshot regardless of team display order.
    teams = sorted([str(row.get("home_team", "")).strip().casefold(), str(row.get("away_team", "")).strip().casefold()])
    return (
        str(row.get("game_date", "")),
        "|".join(teams),
        str(row.get("bookmaker_key") or row.get("bookmaker") or "").strip().casefold(),
        str(row.get("snapshot_timestamp_utc", "")),
    )


def merge_archive(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge snapshots idempotently; conflicting prices at the same timestamp fail closed."""
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in [*existing, *new_rows]:
        key = _snapshot_key(row)
        if key not in merged:
            merged[key] = row
            continue
        old = merged[key]
        comparable = ["total_line", "over_odds", "under_odds"]
        if any(str(old.get(col, "")) != str(row.get(col, "")) for col in comparable):
            raise ValueError(
                "Conflicting totals prices were returned for the same game/bookmaker/snapshot. "
                "Keep the prior archive and investigate the source response."
            )
        # Preserve extra metadata that is available in a later run.
        for field, value in row.items():
            if (field not in old or old.get(field) in (None, "")) and value not in (None, ""):
                old[field] = value
    return sorted(merged.values(), key=lambda r: (str(r.get("game_date", "")), str(r.get("snapshot_timestamp_utc", "")), str(r.get("bookmaker", ""))))


def write_archive(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically persist the archive to avoid partial CSV writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temp_name = handle.name
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(temp_name, path)


def load_api_key() -> str:
    """Read ODDS_API_KEY from the environment only; never log or persist its value."""
    return os.getenv("ODDS_API_KEY", "").strip()


def fetch_snapshot(api_key: str, requested_snapshot_utc: datetime, timeout: int = 30) -> tuple[dict[str, Any], dict[str, str]]:
    query = urlencode({
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": _utc_iso(requested_snapshot_utc),
    })
    request = Request(API_URL + "?" + query, headers={"User-Agent": "OluwaBlazeeHistoricalBacktest/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            headers = {
                "requests_remaining": str(response.headers.get("x-requests-remaining", "")),
                "requests_used": str(response.headers.get("x-requests-used", "")),
            }
    except HTTPError as exc:
        # Do not print the request URL because it contains the API key.
        hint = " Historical odds may require a paid plan." if exc.code in (401, 403, 422) else ""
        raise RuntimeError(f"Historical Odds API returned HTTP {exc.code}.{hint}") from None
    except URLError as exc:
        raise RuntimeError(f"Could not connect to The Odds API ({exc.reason}).") from None
    except (TimeoutError, OSError):
        raise RuntimeError("The Odds API request timed out or failed at the network layer.") from None
    if not isinstance(payload, dict):
        raise ValueError("The Odds API response was not a JSON object.")
    return payload, headers


def collect(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_times = parse_snapshot_times(args.snapshot_times)
    requested = build_requested_snapshots(args.start_date, args.end_date, snapshot_times)
    expected_credits = estimate_api_credits(len(requested))
    print(f"Requested snapshots: {len(requested)}")
    print(f"Estimated API credits: {expected_credits} (US region, totals market only; verify provider plan/quota).")
    print(f"Date range (Eastern): {args.start_date} through {args.end_date}")
    print(f"Snapshot times (Eastern): {', '.join(t.strftime('%H:%M') for t in snapshot_times)}")
    if args.plan_only:
        return {"status": "plan_only", "requests": len(requested), "estimated_api_credits": expected_credits}
    if not args.confirm_api_cost:
        raise ValueError("No API calls were made. Review the estimate, then pass --confirm-api-cost.")
    if args.max_api_credits is None or args.max_api_credits < 0:
        raise ValueError("A non-negative --max-api-credits budget is required for live collection.")
    if expected_credits > args.max_api_credits:
        raise ValueError(
            f"Estimated cost {expected_credits} credits exceeds --max-api-credits {args.max_api_credits}; no API calls were made."
        )
    api_key = load_api_key()
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set in the environment. No API calls were made.")

    output_path = Path(args.output_file)
    archive = _read_archive(output_path)
    diagnostics = {
        "requests_attempted": 0,
        "requests_remaining_header": "",
        "requests_used_header": "",
        "rows_parsed": 0,
        "post_tipoff_events_excluded": 0,
        "invalid_market_pairs_skipped": 0,
    }
    for snapshot_time in requested:
        payload, headers = fetch_snapshot(api_key, snapshot_time)
        rows, row_diag = parse_snapshot_response(payload, snapshot_time, args.bookmaker_key)
        diagnostics["requests_attempted"] += 1
        diagnostics["requests_remaining_header"] = headers.get("requests_remaining", "")
        diagnostics["requests_used_header"] = headers.get("requests_used", "")
        diagnostics["rows_parsed"] += len(rows)
        diagnostics["post_tipoff_events_excluded"] += row_diag["post_tipoff_events_excluded"]
        diagnostics["invalid_market_pairs_skipped"] += row_diag["invalid_market_pairs_skipped"]
        if rows:
            archive = merge_archive(archive, rows)
            write_archive(output_path, archive)
    if not diagnostics["rows_parsed"]:
        raise RuntimeError("No valid pregame totals prices were returned. Existing archive was left unchanged.")
    diagnostics.update({
        "status": "success",
        "archive_rows": len(archive),
        "output_file": str(output_path),
        "provider": "The Odds API v4 historical endpoint",
    })
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect genuine historical NBA totals prices. A paid historical-data plan may be required."
    )
    parser.add_argument("--start-date", required=True, help="First Eastern calendar date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Last Eastern calendar date, YYYY-MM-DD inclusive.")
    parser.add_argument("--snapshot-times", required=True, help="Comma-separated Eastern local times, e.g. 16:00,18:00,20:00,22:00.")
    parser.add_argument("--bookmaker-key", default=None, help="Optional exact provider bookmaker key, e.g. draftkings; blank saves all books.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--plan-only", action="store_true", help="Estimate request count and credits without making network calls.")
    parser.add_argument("--confirm-api-cost", action="store_true", help="Required to make historical API calls.")
    parser.add_argument("--max-api-credits", type=int, default=None, help="Hard spending/quota cap for this run; required unless --plan-only.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        collect(arguments)
    except Exception as error:
        parser.error(str(error))
