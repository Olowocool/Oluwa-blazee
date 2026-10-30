#!/usr/bin/env python3
"""Collect DunkAPI's public hourly NBA odds-board observations for private research.

This reads https://dunkapi.com/odds and does not call the credit-metered API.
Each row stores the page's published-as-of time and the time this collector
retrieved the page. The retrieval time is the conservative availability time
used for prediction cutoffs.

Keep raw odds local/private. Do not commit the archive or redistribute it as
a raw/bulk feed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://dunkapi.com/odds"
EASTERN = ZoneInfo("America/New_York")
DEFAULT_OUTPUT = Path("data/private/dunkapi_public_odds_snapshots.csv")
DEFAULT_MAX_PAGE_AGE_MINUTES = 180
USER_AGENT = "OluwaBlazee-NBA-Research/1.0"

FIELDS = [
    "game_key", "game_date", "commence_time_utc", "away_team", "home_team",
    "bookmaker", "home_ml_american", "away_ml_american", "home_spread",
    "home_spread_odds_american", "total_line", "over_odds_american",
    "under_odds_american", "published_as_of_utc", "observed_at_utc",
    "source_page_url",
]
NUMERIC_FIELDS = [
    "home_ml_american", "away_ml_american", "home_spread",
    "home_spread_odds_american", "total_line", "over_odds_american",
    "under_odds_american",
]


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_page_asof(page_text: str, now: datetime) -> datetime:
    match = re.search(
        r"Lines\s+as\s+of\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{1,2}:\d{2}\s+[AP]M)\s+ET",
        page_text, flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Odds page has no recognizable 'Lines as of ... ET' timestamp.")
    local_now = now.astimezone(EASTERN)
    token = match.group(1)
    candidates = []
    for year in (local_now.year - 1, local_now.year, local_now.year + 1):
        try:
            candidates.append(
                datetime.strptime(f"{token} {year}", "%b %d, %I:%M %p %Y").replace(tzinfo=EASTERN)
            )
        except ValueError:
            pass
    if not candidates:
        raise ValueError(f"Could not parse page timestamp: {token!r}")
    return min(candidates, key=lambda dt: abs((dt - local_now).total_seconds()))


def parse_game_start(token: str, asof: datetime) -> datetime:
    token = " ".join(token.replace("ET", "").split())
    match = re.search(
        r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*([A-Za-z]{3}\s+\d{1,2}),\s*(\d{1,2}:\d{2}\s+[AP]M)",
        token, flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized game start label: {token!r}")
    date_part, time_part = match.groups()
    candidates = []
    for year in (asof.year, asof.year + 1, asof.year - 1):
        try:
            candidates.append(
                datetime.strptime(
                    f"{date_part}, {time_part} {year}", "%b %d, %I:%M %p %Y"
                ).replace(tzinfo=EASTERN)
            )
        except ValueError:
            pass
    future = [dt for dt in candidates if dt >= asof - timedelta(hours=3)]
    if not future:
        raise ValueError(f"Game start is outside the upcoming schedule window: {token!r}")
    return min(future)


def parse_american(cell: str, field: str) -> int:
    value = cell.strip().replace("−", "-").replace("＋", "+")
    if not re.fullmatch(r"[+-]?\d{2,4}", value):
        raise ValueError(f"Invalid {field} American price: {cell!r}")
    price = int(value)
    if price == 0:
        raise ValueError(f"Zero is not a valid {field} American price.")
    return price


def parse_spread(cell: str) -> tuple[float, int]:
    value = " ".join(cell.replace("−", "-").split())
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d{2,4})", value)
    if not match:
        raise ValueError(f"Invalid home spread/price cell: {cell!r}")
    return float(match.group(1)), parse_american(match.group(2), "home spread")


def parse_total(cell: str) -> tuple[float, int, int]:
    value = " ".join(cell.replace("−", "-").split())
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s+([+-]?\d{2,4})\s*/\s*([+-]?\d{2,4})", value)
    if not match:
        raise ValueError(f"Invalid total / Over / Under cell: {cell!r}")
    total = float(match.group(1))
    if total <= 0:
        raise ValueError("Total line must be positive.")
    return total, parse_american(match.group(2), "Over"), parse_american(match.group(3), "Under")


def table_for_heading(heading):
    for node in heading.find_all_next():
        if node.name in {"h1", "h2", "h3"} and node is not heading:
            break
        if node.name == "table":
            return node
    return None


def start_label_for_heading(heading) -> str:
    pattern = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*[A-Za-z]{3}\s+\d{1,2},\s*\d{1,2}:\d{2}\s+[AP]M\s+ET"
    for node in heading.find_all_next():
        if node.name in {"h1", "h2", "h3"} and node is not heading:
            break
        if node.name in {"p", "time", "div", "span"}:
            text = " ".join(node.stripped_strings)
            if re.search(pattern, text, re.IGNORECASE):
                return text
    raise ValueError(f"No Eastern game start label found after {heading.get_text(' ', strip=True)!r}.")


def parse_odds_page(
    html: str,
    retrieved_at: datetime | None = None,
    max_page_age_minutes: int = DEFAULT_MAX_PAGE_AGE_MINUTES,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Parse bookmaker observations; missing/ambiguous prices are never inferred."""
    now = retrieved_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")
    soup = BeautifulSoup(html, "html.parser")
    page_text = " ".join(soup.stripped_strings)
    if "Upcoming NBA odds" not in page_text:
        raise ValueError("Fetched page is not the Upcoming NBA odds board.")
    asof = parse_page_asof(page_text, now)
    age = (now.astimezone(EASTERN) - asof).total_seconds() / 60.0
    if age < -5:
        raise ValueError(f"Odds page timestamp is {abs(age):.1f} minutes in the future.")
    if age > max_page_age_minutes:
        raise ValueError(f"Odds page is stale ({age:.1f} minutes; limit {max_page_age_minutes}).")

    rows: list[dict[str, str]] = []
    skipped = 0
    games_seen = 0
    games_without_odds_table = 0
    for heading in soup.find_all(["h2", "h3"]):
        matchup = " ".join(heading.stripped_strings)
        if " at " not in matchup:
            continue
        away_team, home_team = [part.strip() for part in matchup.split(" at ", 1)]
        if not away_team or not home_team or away_team.casefold() == home_team.casefold():
            raise ValueError(f"Invalid matchup heading: {matchup!r}")
        commence = parse_game_start(start_label_for_heading(heading), asof)
        game_date = commence.astimezone(EASTERN).date().isoformat()
        games_seen += 1
        table = table_for_heading(heading)
        if table is None:
            games_without_odds_table += 1
            continue
        table_rows = table.find_all("tr")
        if len(table_rows) < 2:
            continue
        headers = [" ".join(c.stripped_strings).strip() for c in table_rows[0].find_all(["th", "td"])]
        normalized = [h.casefold() for h in headers]
        book_idx = next((i for i, h in enumerate(normalized) if h in {"book", "sportsbook"}), 0)
        spread_idx = next((i for i, h in enumerate(normalized) if "spread" in h), None)
        total_idx = next((i for i, h in enumerate(normalized) if h == "total" or h.startswith("total ")), None)
        ml_indices = [i for i, h in enumerate(normalized) if h.endswith(" ml") or h == "ml"]
        if spread_idx is None or total_idx is None or len(ml_indices) < 2:
            raise ValueError(f"Unexpected odds headers for {matchup}: {headers!r}")
        home_abbrev = normalized[spread_idx].split()[0]
        home_ml_idx = next((i for i in ml_indices if normalized[i].split()[0] == home_abbrev), ml_indices[-1])
        away_ml_idx = next(i for i in ml_indices if i != home_ml_idx)

        for tr in table_rows[1:]:
            cells = [" ".join(c.stripped_strings).strip() for c in tr.find_all(["td", "th"])]
            if len(cells) <= max(book_idx, spread_idx, total_idx, home_ml_idx, away_ml_idx):
                skipped += 1
                continue
            bookmaker = cells[book_idx].strip()
            if not bookmaker:
                skipped += 1
                continue
            try:
                home_ml = parse_american(cells[home_ml_idx], "home moneyline")
                away_ml = parse_american(cells[away_ml_idx], "away moneyline")
                home_spread, spread_price = parse_spread(cells[spread_idx])
                total, over_price, under_price = parse_total(cells[total_idx])
            except ValueError:
                skipped += 1
                continue
            key = "|".join([game_date, away_team.casefold(), home_team.casefold()])
            rows.append({
                "game_key": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
                "game_date": game_date,
                "commence_time_utc": utc_iso(commence),
                "away_team": away_team,
                "home_team": home_team,
                "bookmaker": bookmaker,
                "home_ml_american": str(home_ml),
                "away_ml_american": str(away_ml),
                "home_spread": str(home_spread),
                "home_spread_odds_american": str(spread_price),
                "total_line": str(total),
                "over_odds_american": str(over_price),
                "under_odds_american": str(under_price),
                "published_as_of_utc": utc_iso(asof),
                "observed_at_utc": utc_iso(now),
                "source_page_url": PAGE_URL,
            })
    if not rows:
        raise ValueError("No complete bookmaker rows could be parsed from the odds board.")
    return rows, {
        "source_page": PAGE_URL,
        "published_as_of_utc": utc_iso(asof),
        "observed_at_utc": utc_iso(now),
        "page_age_minutes": round(age, 2),
        "games_seen": games_seen,
        "games_without_odds_table": games_without_odds_table,
        "bookmaker_rows_parsed": len(rows),
        "bookmaker_rows_skipped_incomplete": skipped,
        "unique_games": len({r["game_key"] for r in rows}),
        "unique_bookmakers": sorted({r["bookmaker"] for r in rows}),
    }


def fetch_odds_page(timeout: int = 30) -> tuple[str, datetime]:
    try:
        response = requests.get(
            PAGE_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=timeout,
        )
        response.raise_for_status()
        retrieved_at = datetime.now(timezone.utc)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not retrieve public odds page ({type(exc).__name__}).") from None
    content_type = response.headers.get("content-type", "").casefold()
    if "html" not in content_type:
        raise RuntimeError(f"Expected HTML, received content-type {content_type!r}.")
    return response.text, retrieved_at


def merge_snapshot_rows(
    existing: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    """Idempotent merge by game/book/page publication; fail on timestamp conflicts."""
    merged: dict[tuple[str, str, str], dict[str, str]] = {}

    def add(row: dict[str, str], is_new: bool) -> int:
        key = (row["game_key"], row["bookmaker"].casefold(), row["published_as_of_utc"])
        if key not in merged:
            merged[key] = row
            return int(is_new)
        old = merged[key]
        for col in NUMERIC_FIELDS + ["commence_time_utc", "home_team", "away_team"]:
            if str(old.get(col, "")) != str(row.get(col, "")):
                raise ValueError(
                    "Conflicting odds values for the same game, bookmaker and page timestamp; "
                    "the prior archive was not overwritten."
                )
        return 0  # retain first observed_at when the same page publication is polled again

    for row in existing:
        add(row, False)
    inserted = sum(add(row, True) for row in new_rows)
    return sorted(
        merged.values(),
        key=lambda r: (r["game_date"], r["commence_time_utc"], r["bookmaker"], r["published_as_of_utc"]),
    ), inserted


def read_archive(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Private market archive is missing columns: {sorted(missing)}")
        return list(reader)


def write_archive(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in rows])
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def run_once(args: argparse.Namespace) -> dict[str, object]:
    html, retrieved_at = fetch_odds_page(timeout=args.timeout)
    rows, diagnostics = parse_odds_page(
        html, retrieved_at=retrieved_at, max_page_age_minutes=args.max_page_age_minutes
    )
    if args.check_only:
        diagnostics["status"] = "check_only_no_data_saved"
        print(json.dumps(diagnostics, indent=2))
        return diagnostics
    path = Path(args.output_file)
    merged, inserted = merge_snapshot_rows(read_archive(path), rows)
    write_archive(path, merged)
    diagnostics.update({
        "status": "saved",
        "rows_inserted": inserted,
        "archive_rows": len(merged),
        "output_file": str(path),
        "raw_prices_committed_to_git": False,
    })
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-page-age-minutes", type=int, default=DEFAULT_MAX_PAGE_AGE_MINUTES)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Fetch and validate the current page without saving odds data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_page_age_minutes < 1 or args.timeout < 1:
        raise SystemExit("--max-page-age-minutes and --timeout must be positive.")
    try:
        run_once(args)
    except Exception as exc:
        raise SystemExit(f"Market snapshot collection failed: {type(exc).__name__}: {exc}") from None


if __name__ == "__main__":
    main()
