#!/usr/bin/env python3
"""Read-only, privacy-preserving summary of the local public odds snapshot archive.

This script never prints or uploads raw prices, never calls DunkAPI, and never
modifies the archive. Run from the repository root on the Windows machine that
contains data/private/dunkapi_public_odds_snapshots.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("data/private/dunkapi_public_odds_snapshots.csv")
TASK_NAME = "NBA Research Odds Snapshot Capture"
REQUIRED = {
    "game_key", "game_date", "commence_time_utc", "away_team", "home_team",
    "bookmaker", "total_line", "over_odds_american", "under_odds_american",
    "published_as_of_utc", "observed_at_utc",
}


def parse_utc(value: str) -> datetime:
    token = (value or "").strip()
    if not token:
        raise ValueError("empty timestamp")
    dt = datetime.fromisoformat(token.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return dt.astimezone(timezone.utc)


def percentile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return round(ordered[lo], 2)
    return round(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo), 2)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": round(max(values), 2),
    }


def task_scheduler_status() -> dict:
    if shutil.which("schtasks") is None:
        return {"available": False, "reason": "schtasks command is unavailable on this OS"}
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"],
            capture_output=True, text=True, timeout=8, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"available": True, "status": "query_timed_out"}
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return {"available": True, "found": False, "return_code": result.returncode}
    selected = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().casefold()
        if k in {
            "taskname", "next run time", "status", "last run time",
            "last result", "schedule type", "start time", "repeat: every",
        }:
            selected[key.strip()] = value.strip()
    return {"available": True, "found": True, "details": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    args = parser.parse_args()
    path = Path(args.path)
    result = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(path),
        "raw_prices_printed": False,
        "raw_archive_modified": False,
        "paid_api_calls": 0,
        "scheduler_task": task_scheduler_status(),
    }
    if not path.exists():
        result["file_exists"] = False
        print(json.dumps(result, indent=2))
        return

    result["file_exists"] = True
    result["file_size_bytes"] = path.stat().st_size
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        result["columns"] = columns
        missing = sorted(REQUIRED - set(columns))
        if missing:
            result["schema_error"] = {"missing_required_columns": missing}
            print(json.dumps(result, indent=2))
            return
        rows = list(reader)

    ages: list[float] = []
    tip_offsets: list[float] = []
    books = Counter()
    dates = Counter()
    games = set()
    keys = Counter()
    missing_timestamps = 0
    invalid_prices = 0
    negative_age = 0
    after_tip = 0
    near_t120_15 = 0
    near_t120_30 = 0
    candidate_t120_30 = 0
    candidate_t120_30_dk = 0

    for row in rows:
        book = (row.get("bookmaker") or "").strip()
        books[book or "<blank>"] += 1
        dates[(row.get("game_date") or "").strip() or "<blank>"] += 1
        games.add((row.get("game_key") or "").strip())
        key = (
            (row.get("game_key") or "").strip(),
            book.casefold(),
            (row.get("published_as_of_utc") or "").strip(),
        )
        keys[key] += 1

        try:
            published = parse_utc(row.get("published_as_of_utc", ""))
            observed = parse_utc(row.get("observed_at_utc", ""))
            commence = parse_utc(row.get("commence_time_utc", ""))
        except Exception:
            missing_timestamps += 1
            continue

        age = (observed - published).total_seconds() / 60.0
        before_tip = (commence - observed).total_seconds() / 60.0
        ages.append(age)
        tip_offsets.append(before_tip)
        if age < 0:
            negative_age += 1
        if before_tip <= 0:
            after_tip += 1
        close_15 = abs(before_tip - 120.0) <= 15.0
        close_30 = abs(before_tip - 120.0) <= 30.0
        if close_15:
            near_t120_15 += 1
        if close_30:
            near_t120_30 += 1
            age_ok = 0 <= age <= 240
            prices_ok = all(
                (row.get(k) or "").strip() not in {"", "0", "0.0"}
                for k in ("total_line", "over_odds_american", "under_odds_american")
            )
            if age_ok and prices_ok:
                candidate_t120_30 += 1
                if book.casefold() == "draftkings":
                    candidate_t120_30_dk += 1
        try:
            if float(row.get("total_line", "")) <= 0:
                invalid_prices += 1
            if int(row.get("over_odds_american", "0")) == 0 or int(row.get("under_odds_american", "0")) == 0:
                invalid_prices += 1
        except Exception:
            invalid_prices += 1

    duplicate_rows = sum(n - 1 for n in keys.values() if n > 1)
    result.update({
        "rows": len(rows),
        "unique_game_keys": len(games - {""}),
        "date_start": min((d for d in dates if d != "<blank>"), default=None),
        "date_end": max((d for d in dates if d != "<blank>"), default=None),
        "rows_by_game_date": dict(sorted(dates.items())),
        "rows_by_bookmaker": dict(sorted(books.items())),
        "duplicate_game_book_published_keys_extra_rows": duplicate_rows,
        "rows_with_missing_or_invalid_timestamps": missing_timestamps,
        "rows_with_invalid_total_or_prices": invalid_prices,
        "published_page_age_minutes": {
            **distribution(ages),
            "age_le_240_minutes": sum(0 <= x <= 240 for x in ages),
            "age_gt_240_minutes": sum(x > 240 for x in ages),
            "future_published_timestamp": negative_age,
        },
        "minutes_from_observation_to_tip": {
            **distribution(tip_offsets),
            "observed_at_or_after_tip": after_tip,
            "within_15_minutes_of_T_minus_120": near_t120_15,
            "within_30_minutes_of_T_minus_120": near_t120_30,
        },
        "provisional_T120_diagnostic_only": {
            "within_30_minutes_of_T_minus_120_and_page_age_le_240_and_prices_present": candidate_t120_30,
            "same_filter_draftkings_rows": candidate_t120_30_dk,
            "note": "Diagnostic count only; not a finalized eligibility rule or betting record."
        },
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
