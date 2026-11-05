"""Pure-Python retrieval and parsing of the official NBA injury-report PDFs.

The research pipeline uses nbainjuries/tabula (Java) for historical bulk work.
The deployed testbench uses this PyPDF2 parser so its native Python Render service
does not need a Java runtime. It preserves the official report timestamp and hash.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, time, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from PyPDF2 import PdfReader

NBA_INJURY_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_*.pdf"
ET = ZoneInfo("America/New_York")
STATUS_PATTERN = re.compile(
    r"^(.*?)\s+(Available|Questionable|Out|Probable|Doubtful|Suspended|Not With Team)\b(?:\s+(.*))?$",
    re.IGNORECASE,
)
GAME_HEADER_PATTERN = re.compile(
    r"^(?:(\d{2}/\d{2}/\d{4})\s+)?(\d{2}:\d{2}\s+\(ET\))\s+"
    r"([A-Z0-9]{2,4}@[A-Z0-9]{2,4})\s+(.*)$",
    re.IGNORECASE,
)


def _report_url(timestamp: datetime) -> str:
    if timestamp >= datetime(2025, 12, 22, 9, 0):
        suffix = timestamp.strftime("%I_%M%p")
    elif timestamp <= datetime(2025, 12, 19, 15, 30) or timestamp >= datetime(2025, 12, 19, 16, 45):
        suffix = timestamp.strftime("%I%p")
    else:
        raise ValueError(f"Official injury report archive timestamp gap: {timestamp}")
    return NBA_INJURY_URL.replace("*", timestamp.strftime("%Y-%m-%d") + "_" + suffix)


def _team_prefix(value: str, team_names: list[str]) -> tuple[str | None, str]:
    lowered = value.casefold()
    for name in sorted(team_names, key=len, reverse=True):
        if lowered.startswith(name.casefold()):
            return name, value[len(name):].strip()
    return None, value.strip()


def parse_official_injury_report_text(
    text: str, target_date, team_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Parse the tabular text emitted by PyPDF2, preserving the nbainjuries row schema."""
    target = pd.Timestamp(target_date).normalize()
    names = list(dict.fromkeys(str(x).strip() for x in team_names if str(x).strip()))
    if not names:
        raise ValueError("Current-season NBA team names are required to parse injury reports")
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None
    records: list[dict] = []
    unsubmitted: list[str] = []
    last_record: dict | None = None

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", str(raw_line)).strip()
        if not line:
            continue
        line = re.sub(r"Injury Report:\s*\d{2}/\d{2}/\d{2,4}.*$", "", line, flags=re.I).strip()
        if not line or re.match(r"Page\s+\d+\s+of\s+\d+", line, flags=re.I):
            continue
        if "Game Date Game Time Matchup Team Player Name Current Status" in line:
            continue
        if line.casefold().startswith("injury report:"):
            continue

        header = GAME_HEADER_PATTERN.match(line)
        remainder = line
        if header:
            if header.group(1):
                current_date = pd.to_datetime(header.group(1), format="%m/%d/%Y").normalize()
            current_time = header.group(2)
            current_matchup = header.group(3).upper()
            remainder = header.group(4).strip()
        if current_date is None or current_time is None or current_matchup is None:
            # The first game line must establish the report's date/time/matchup.
            continue

        team_name, remainder = _team_prefix(remainder, names)
        if team_name:
            current_team = team_name
            last_record = None

        if "NOT YET SUBMITTED" in remainder.upper():
            unsubmitted.append(current_team or "unknown team")
            last_record = None
            continue

        status_match = STATUS_PATTERN.match(remainder)
        if status_match:
            player_name = status_match.group(1).strip()
            status = status_match.group(2).strip().upper()
            reason = (status_match.group(3) or "").strip()
            if not player_name:
                continue
            if current_team is None:
                raise ValueError(f"Injury player row has no team context: {line}")
            record = {
                "Game Date": current_date.strftime("%m/%d/%Y"),
                "Game Time": current_time,
                "Matchup": current_matchup,
                "Team": current_team,
                "Player Name": player_name,
                "Current Status": status,
                "Reason": reason,
            }
            if current_date == target:
                records.append(record)
                last_record = record
            else:
                last_record = None
            continue

        # Continuation lines belong to the preceding player's reason field.
        if last_record is not None and current_date == target:
            continuation = line.strip()
            if continuation and not continuation.casefold().startswith("page "):
                last_record["Reason"] = (
                    f'{last_record["Reason"]} {continuation}'.strip()
                )

    if unsubmitted:
        return pd.DataFrame(records), sorted(set(unsubmitted))
    columns = [
        "Game Date", "Game Time", "Matchup", "Team", "Player Name",
        "Current Status", "Reason",
    ]
    return pd.DataFrame(records, columns=columns), []


def fetch_official_injury_reports(game_dates, team_names: pd.DataFrame):
    """Fetch one complete official report per date; fail closed on incomplete reports."""
    names = team_names.full_name.astype(str).dropna().unique().tolist()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    rows: list[dict] = []
    manifest: list[dict] = []

    for game_date in sorted(pd.Timestamp(x).normalize() for x in game_dates):
        if game_date.date() in {datetime(2025, 12, 20).date(), datetime(2025, 12, 21).date()}:
            manifest.append({"game_date": str(game_date.date()), "status": "archive_timestamp_gap"})
            continue
        candidates = [time(10, 0), time(9, 0), time(8, 0)]
        if game_date >= pd.Timestamp("2025-12-22"):
            candidates = [
                time(10, 0), time(9, 45), time(9, 30), time(9, 15),
                time(9, 0), time(8, 45), time(8, 30), time(8, 15), time(8, 0),
            ]
        attempts: list[str] = []
        found = False
        for clock in candidates:
            timestamp = datetime.combine(game_date.date(), clock)
            timestamp_et = timestamp.replace(tzinfo=ET)
            timestamp_utc = timestamp_et.astimezone(timezone.utc).isoformat()
            url = _report_url(timestamp)
            try:
                response = session.get(url, timeout=45)
                if response.status_code in (403, 404):
                    attempts.append(f"{clock}:HTTP{response.status_code}")
                    continue
                response.raise_for_status()
                pdf_bytes = response.content
                reader = PdfReader(BytesIO(pdf_bytes))
                page_texts = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(page_texts)
                target_label = game_date.strftime("%m/%d/%Y")
                if not text.strip() or target_label not in text:
                    attempts.append(f"{clock}:target_date_or_text_missing")
                    continue
                parsed, unsubmitted = parse_official_injury_report_text(
                    text, game_date, names
                )
                if unsubmitted:
                    attempts.append(f"{clock}:unsubmitted:{','.join(unsubmitted)}")
                    continue
                digest = hashlib.sha256(pdf_bytes).hexdigest()
                if not parsed.empty:
                    parsed["game_date"] = pd.to_datetime(
                        parsed["Game Date"], format="%m/%d/%Y"
                    ).dt.normalize()
                    parsed = parsed[parsed.game_date.eq(game_date)].copy()
                    parsed["snapshot_time"] = timestamp.isoformat()
                    parsed["snapshot_timezone"] = "America/New_York"
                    parsed["snapshot_time_utc"] = timestamp_utc
                    parsed["snapshot_url"] = url
                    parsed["snapshot_sha256"] = digest
                    rows.extend(parsed.to_dict("records"))
                manifest.append({
                    "game_date": str(game_date.date()),
                    "status": "complete",
                    "snapshot_time": timestamp.isoformat(),
                    "snapshot_timezone": "America/New_York",
                    "snapshot_time_utc": timestamp_utc,
                    "url": url,
                    "sha256": digest,
                    "rows": int(len(parsed)),
                    "pdf_pages": len(reader.pages),
                })
                found = True
                break
            except Exception as exc:
                attempts.append(f"{clock}:{type(exc).__name__}:{str(exc)[:140]}")
        if not found:
            manifest.append({
                "game_date": str(game_date.date()),
                "status": "missing_or_incomplete",
                "attempts": attempts,
            })

    columns = [
        "Game Date", "Game Time", "Matchup", "Team", "Player Name",
        "Current Status", "Reason", "game_date", "snapshot_time",
        "snapshot_timezone", "snapshot_time_utc", "snapshot_url", "snapshot_sha256",
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(manifest)
