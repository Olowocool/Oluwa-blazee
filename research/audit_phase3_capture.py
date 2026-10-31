#!/usr/bin/env python3
"""Read-only, privacy-limited audit of the local Phase 3 odds archive."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/private/dunkapi_public_odds_snapshots.csv"
DEFAULT_OUTPUT = ROOT / "outputs/research/phase3_capture_audit"
REQUIRED = {
    "game_key", "away_team", "home_team", "bookmaker",
    "commence_time_utc", "published_as_of_utc", "observed_at_utc",
}


def build_capture_audit(
    raw: pd.DataFrame,
    audit_time: pd.Timestamp | None = None,
    bookmaker: str = "DraftKings",
    entry_minutes: int = 120,
    max_age_minutes: int = 240,
) -> dict[str, Any]:
    if raw is None or raw.empty:
        raise ValueError("Private odds snapshot archive is empty.")
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Archive is missing required fields: {sorted(missing)}")
    if entry_minutes < 0 or max_age_minutes < entry_minutes:
        raise ValueError("Require 0 <= entry cutoff <= maximum entry age.")
    d = raw.copy()
    for col in ("commence_time_utc", "published_as_of_utc", "observed_at_utc"):
        d[col] = pd.to_datetime(d[col], errors="coerce", utc=True)
    blank = d[["game_key", "away_team", "home_team", "bookmaker"]].fillna("").astype(str).apply(
        lambda col: col.str.strip().eq("")
    ).any(axis=1)
    invalid = d[["commence_time_utc", "published_as_of_utc", "observed_at_utc"]].isna().any(axis=1) | blank
    d = d.loc[~invalid].copy()
    d["_book"] = d["bookmaker"].astype(str).str.strip().str.casefold()
    dk = d.loc[d["_book"].eq(bookmaker.casefold())].copy()
    if dk.empty:
        raise ValueError(f"No valid rows for {bookmaker}.")
    post = dk["observed_at_utc"] >= dk["commence_time_utc"]
    pre = dk.loc[~post].copy()
    key = ["game_key", "_book", "published_as_of_utc"]
    # Retain the first local retrieval for repeated provider publications.
    pre = pre.sort_values("observed_at_utc", kind="stable")
    duplicates = int(pre.duplicated(key).sum())
    pre = pre.drop_duplicates(key, keep="first")
    now = pd.Timestamp.now(tz="UTC") if audit_time is None else pd.Timestamp(audit_time)
    if now.tzinfo is None:
        raise ValueError("audit_time must be timezone-aware.")
    now = now.tz_convert("UTC")
    games = []
    for (game_key, _), g in pre.groupby(["game_key", "_book"], sort=True):
        g = g.sort_values("observed_at_utc").reset_index(drop=True)
        starts = g["commence_time_utc"].unique()
        if len(starts) != 1:
            raise ValueError(f"Inconsistent scheduled starts for game {game_key}.")
        start = pd.Timestamp(starts[0]).tz_convert("UTC")
        cutoff = start - pd.Timedelta(minutes=entry_minutes)
        eligible = g.loc[g["observed_at_utc"] <= cutoff]
        entry = None if eligible.empty else eligible.iloc[-1]
        latest = g.iloc[-1]
        age = None if entry is None else (start - entry["observed_at_utc"]).total_seconds() / 60
        status = "no_entry" if entry is None else ("fresh_entry" if age <= max_age_minutes else "stale_entry")
        games.append({
            "matchup": f"{g.iloc[0]['away_team']} at {g.iloc[0]['home_team']}",
            "start_utc": start.isoformat().replace("+00:00", "Z"),
            "pregame_rows": int(len(g)),
            "board_publications": int(g["published_as_of_utc"].nunique()),
            "rows_by_t120": int(len(eligible)),
            "entry_observed_utc": None if entry is None else entry["observed_at_utc"].isoformat().replace("+00:00", "Z"),
            "entry_age_minutes": None if age is None else round(float(age), 2),
            "later_pregame_rows": 0 if entry is None else int(
                ((g["observed_at_utc"] > entry["observed_at_utc"]) & (g["observed_at_utc"] < start)).sum()
            ),
            "latest_pregame_observed_utc": latest["observed_at_utc"].isoformat().replace("+00:00", "Z"),
            "minutes_since_latest_observation": round((now - latest["observed_at_utc"]).total_seconds() / 60, 2),
            "status": status,
        })
    games.sort(key=lambda x: (x["start_utc"], x["matchup"]))
    pubs = dk["published_as_of_utc"].drop_duplicates().sort_values()
    gaps = pubs.diff().dropna().dt.total_seconds().div(60).to_numpy(dtype=float)
    page_age = (dk["observed_at_utc"] - dk["published_as_of_utc"]).dt.total_seconds().div(60)
    last_obs = dk["observed_at_utc"].max()
    statuses = {s: sum(g["status"] == s for g in games) for s in ("fresh_entry", "stale_entry", "no_entry")}
    return {
        "archive_rows": int(len(raw)),
        "invalid_rows_excluded": int(invalid.sum()),
        "bookmaker": bookmaker,
        "bookmaker_rows_including_post_start": int(len(dk)),
        "post_start_rows_excluded": int(post.sum()),
        "pregame_rows_after_dedup": int(len(pre)),
        "duplicate_pregame_rows_removed_for_audit": duplicates,
        "unique_games": int(pre.game_key.nunique()),
        "distinct_board_publications": int(pubs.nunique()),
        "publication_gap_median_minutes": None if not len(gaps) else round(float(pd.Series(gaps).median()), 2),
        "publication_gap_min_minutes": None if not len(gaps) else round(float(gaps.min()), 2),
        "publication_gap_max_minutes": None if not len(gaps) else round(float(gaps.max()), 2),
        "page_age_median_minutes": round(float(page_age.median()), 2),
        "page_age_max_minutes": round(float(page_age.max()), 2),
        "first_observation_utc": dk.observed_at_utc.min().isoformat().replace("+00:00", "Z"),
        "last_observation_utc": last_obs.isoformat().replace("+00:00", "Z"),
        "minutes_since_last_observation": round((now - last_obs).total_seconds() / 60, 2),
        "audit_time_utc": now.isoformat().replace("+00:00", "Z"),
        "entry_cutoff_minutes": entry_minutes,
        "max_entry_age_minutes": max_age_minutes,
        "entry_status_counts": statuses,
        "games": games,
        "privacy_note": "Timestamps, matchup labels and counts only. No market-price fields are emitted.",
        "archive_caveats": [
            "Same-publication polls are deduplicated, so the archive cannot show every collector invocation.",
            "The previous batch launcher did not log invocations; historical poll frequency cannot be reconstructed from this archive alone.",
            "A scheduled-task last-run status describes only the latest run, not every prior trigger.",
        ],
    }


def scheduled_task_status() -> dict[str, Any]:
    if platform.system().casefold() != "windows":
        return {"status": "not_windows", "tasks": []}
    ps = r"""
    $ErrorActionPreference='Stop'; Import-Module ScheduledTasks -ErrorAction Stop
    $found=@(Get-ScheduledTask | Where-Object { (($_.Actions | ForEach-Object { $_.Execute+' '+$_.Arguments }) -join ' ') -match '(?i)run_market_snapshot_capture\.bat' } | ForEach-Object {
      $task=$_
      $i=Get-ScheduledTaskInfo -TaskName $task.TaskName -TaskPath $task.TaskPath
      $triggers=@($task.Triggers | ForEach-Object { [pscustomobject]@{type=$_.CimClass.CimClassName;enabled=$_.Enabled;start_boundary=$_.StartBoundary;repetition_interval=$_.Repetition.Interval;repetition_duration=$_.Repetition.Duration} })
      [pscustomobject]@{
        name=$task.TaskName;path=$task.TaskPath;state=[string]$task.State
        last_run=[string]$i.LastRunTime;last_result=[string]$i.LastTaskResult;next_run=[string]$i.NextRunTime
        action=(($task.Actions | ForEach-Object {$_.Execute+' '+$_.Arguments}) -join ' ')
        triggers=$triggers
        settings=[pscustomobject]@{
          start_when_available=$task.Settings.StartWhenAvailable
          wake_to_run=$task.Settings.WakeToRun
          execution_time_limit=$task.Settings.ExecutionTimeLimit
          multiple_instances=$task.Settings.MultipleInstances
        }
      }
    })
    if($found.Count){@{status='matched';tasks=$found}|ConvertTo-Json -Compress -Depth 7}else{@{status='not_found';tasks=@()}|ConvertTo-Json -Compress}
    """
    try:
        p = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=30)
        return json.loads((p.stdout or "{}").strip().splitlines()[-1])
    except Exception as exc:
        return {"status": "query_failed", "error": type(exc).__name__, "tasks": []}


def write_report(audit: dict[str, Any], task: dict[str, Any], out_dir: Path, log_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    audit["scheduled_task_status"] = task
    audit["capture_run_log"] = str(log_path)
    lines = [
        "# Phase 3B — local market-capture audit", "", "## Archive summary", "",
        f"- Archive rows: {audit['archive_rows']:,}; DraftKings rows incl. post-start: {audit['bookmaker_rows_including_post_start']:,}.",
        f"- Post-start rows excluded: {audit['post_start_rows_excluded']:,}; retained pregame rows: {audit['pregame_rows_after_dedup']:,}.",
        f"- Games: {audit['unique_games']}; distinct provider publications: {audit['distinct_board_publications']}.",
        f"- Publication gap median/min/max: {audit['publication_gap_median_minutes']} / {audit['publication_gap_min_minutes']} / {audit['publication_gap_max_minutes']} minutes.",
        f"- Retrieval/page-age median/max: {audit['page_age_median_minutes']} / {audit['page_age_max_minutes']} minutes.",
        f"- Archive observation range: {audit['first_observation_utc']} to {audit['last_observation_utc']} UTC.",
        f"- Minutes since latest archived observation: {audit['minutes_since_last_observation']}.", "",
        "## T−120 coverage", "",
        f"- Fresh entries: {audit['entry_status_counts']['fresh_entry']}; stale: {audit['entry_status_counts']['stale_entry']}; no entry by cutoff: {audit['entry_status_counts']['no_entry']}.", "",
        "| Matchup | Pregame rows | Publications | Rows by T−120 | Entry UTC | Age (min) | Later rows | Latest pregame UTC | Status |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for g in audit["games"]:
        age = "—" if g["entry_age_minutes"] is None else g["entry_age_minutes"]
        lines.append(
            f"| {g['matchup']} | {g['pregame_rows']} | {g['board_publications']} | {g['rows_by_t120']} | "
            f"{g['entry_observed_utc'] or '—'} | {age} | {g['later_pregame_rows']} | "
            f"{g['latest_pregame_observed_utc']} | {g['status']} |"
        )
    lines += ["", "## Windows Task Scheduler", "", f"- Query status: {task.get('status')}."]
    if task.get("status") == "matched":
        for item in task.get("tasks", []):
            lines += [
                f"- Task: {item.get('path','')}{item.get('name','')}; state {item.get('state')}; "
                f"last run {item.get('last_run')}; result {item.get('last_result')}; next run {item.get('next_run')}.",
                f"- Action: {item.get('action')}.",
            ]
            for trigger in item.get("triggers", []):
                lines.append(
                    f"- Trigger: type {trigger.get('type')}; enabled {trigger.get('enabled')}; "
                    f"start {trigger.get('start_boundary')}; repeat interval {trigger.get('repetition_interval')}; "
                    f"repeat duration {trigger.get('repetition_duration')}."
                )
            settings = item.get("settings", {})
            lines.append(
                f"- Settings: start when available {settings.get('start_when_available')}; "
                f"wake to run {settings.get('wake_to_run')}; execution limit {settings.get('execution_time_limit')}; "
                f"multiple instances {settings.get('multiple_instances')}."
            )
    elif task.get("status") == "not_found":
        lines.append("- No task action referencing run_market_snapshot_capture.bat was found. A differently named launcher/task is not ruled out.")
    else:
        lines.append(f"- Scheduler query unavailable: {task.get('error', task.get('status'))}.")
    lines += [
        "", "## Collector run log", "",
        f"- Path: {log_path}; exists: {log_path.is_file()}.",
        "- This log begins only after the updated capture launcher is first run. It cannot recreate past invocation history.", "",
        "## Interpretation", "",
        "- DunkAPI documents hourly odds captures for games in the next four days. A 15-minute local poll helps avoid missing a new hourly publication; it does not create new prices between provider updates.",
        "- Repeated polls of one publication are intentionally deduplicated.",
        "- Keep the T−120 cutoff and 240-minute maximum age unchanged.",
        "- Timestamps can identify stale observations but cannot, by themselves, prove whether the computer slept, a task failed, the page omitted a game, or a provider update was missed.",
        "- No odds or market prices are written to this report. The input archive is read-only.", "",
    ]
    md = out_dir / "phase3_capture_audit.md"
    js = out_dir / "phase3_capture_audit.json"
    md.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return md, js


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-file", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    src = Path(args.snapshots_file)
    src = src if src.is_absolute() else ROOT / src
    if not src.is_file():
        raise SystemExit(f"Private archive not found: {src}")
    audit = build_capture_audit(pd.read_csv(src))
    task = scheduled_task_status()
    out = Path(args.output_dir)
    out = out if out.is_absolute() else ROOT / out
    log = ROOT / "outputs/research/market_capture/capture_runs.log"
    md, js = write_report(audit, task, out, log)
    print(json.dumps({
        "status": "success", "archive_rows": audit["archive_rows"], "games": audit["unique_games"],
        "entry_status_counts": audit["entry_status_counts"], "task_status": task.get("status"),
        "report": str(md), "details": str(js), "market_prices_emitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
