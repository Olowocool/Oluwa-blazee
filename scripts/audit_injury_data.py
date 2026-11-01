#!/usr/bin/env python3
"""Audit SportsDataverse historical NBA injury reports against game rosters/boxscores.

Downloads small public Parquet release assets at run time. Nothing is committed to
the application repository; only a compact JSON/Markdown audit report is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_ROOT = "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags"
RELEASES = {
    "injury": "nba_stats_injury_report",
    "box": "nba_stats_player_boxscores",
    "schedule": "nba_stats_schedules",
    "roster": "nba_stats_game_rosters",
}
REQUIRED_INJURY = {
    "season", "report_datetime", "game_date", "game_time", "report_matchup",
    "team", "report_player_name", "current_status", "reason", "game_id",
    "team_id", "person_id", "match_method",
}


def release_assets(session: requests.Session, tag: str) -> dict[str, dict[str, Any]]:
    response = session.get(f"{API_ROOT}/{tag}", timeout=45)
    response.raise_for_status()
    payload = response.json()
    return {asset["name"]: asset for asset in payload.get("assets", [])}


def download_asset(
    session: requests.Session,
    asset: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(asset["browser_download_url"], timeout=90)
    response.raise_for_status()
    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    expected = str(asset.get("digest") or "")
    if expected.startswith("sha256:") and digest != expected.split(":", 1)[1]:
        raise ValueError(f"SHA-256 mismatch for {asset['name']}")
    destination.write_bytes(content)
    return {
        "name": asset["name"],
        "size_bytes": len(content),
        "sha256": digest,
        "published_asset_updated_at": asset.get("updated_at"),
    }


def download_season_assets(
    session: requests.Session,
    data_dir: Path,
    kind: str,
    seasons: list[int],
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    tag = RELEASES[kind]
    assets = release_assets(session, tag)
    paths: list[Path] = []
    downloaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    prefixes = {
        "injury": "injury_report",
        "box": "player_boxscores",
        "schedule": "nba_schedule",
        "roster": "game_rosters",
    }
    for season in seasons:
        name = f"{prefixes[kind]}_{season}.parquet"
        asset = assets.get(name)
        if asset is None:
            missing.append({"dataset": kind, "season": season, "asset": name})
            continue
        path = data_dir / tag / name
        downloaded.append(download_asset(session, asset, path))
        paths.append(path)
    return paths, downloaded, missing


def read_frames(paths: list[Path], kind: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["_asset"] = path.name
        frame["_dataset_kind"] = kind
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def parse_minutes(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "dnp", "inactive"}:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) + float(parts[1]) / 60.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _null_rate(series: pd.Series) -> float:
    return round(float(series.isna().mean() * 100.0), 2) if len(series) else 100.0


def _normal_status(series: pd.Series) -> pd.Series:
    return series.fillna("<NULL>").astype(str).str.strip().str.title()


def audit_frames(
    injury: pd.DataFrame,
    box: pd.DataFrame,
    schedule: pd.DataFrame,
    roster: pd.DataFrame,
) -> dict[str, Any]:
    missing_columns = sorted(REQUIRED_INJURY.difference(injury.columns))
    if missing_columns:
        raise ValueError(f"Injury report is missing required columns: {missing_columns}")
    for frame, column in [(injury, "game_id"), (box, "game_id"), (schedule, "game_id")]:
        if column in frame:
            frame[column] = frame[column].astype("string").str.strip()
    for frame, column in [(injury, "person_id"), (box, "person_id"), (roster, "player_id")]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    injury["game_date"] = pd.to_datetime(injury["game_date"], errors="coerce")
    injury["report_datetime"] = pd.to_datetime(injury["report_datetime"], errors="coerce")
    injury["status_normalized"] = _normal_status(injury["current_status"])
    injury["is_submission_placeholder"] = (
        injury["reason"].fillna("").astype(str).str.upper().str.contains("NOT YET SUBMITTED")
        | injury["report_player_name"].isna()
    )

    # Only actual player listings are used in player-match and status checks.
    player_listings = injury.loc[
        ~injury["is_submission_placeholder"]
    ].copy()
    players = player_listings.loc[
        player_listings["person_id"].notna()
        & player_listings["game_id"].notna()
    ].copy()
    report_games = set(injury["game_id"].dropna().astype(str))
    box_games = set(box["game_id"].dropna().astype(str)) if "game_id" in box else set()
    schedule_games = set(schedule["game_id"].dropna().astype(str)) if "game_id" in schedule else set()

    box_player_keys = set()
    if {"game_id", "person_id"}.issubset(box.columns):
        valid_box_keys = box.dropna(subset=["game_id", "person_id"])
        box_player_keys = set(zip(
            valid_box_keys["game_id"].astype(str),
            valid_box_keys["person_id"].astype("int64"),
        ))
    roster_player_keys = set()
    if {"game_id", "player_id"}.issubset(roster.columns):
        valid_roster_keys = roster.dropna(subset=["game_id", "player_id"])
        roster_player_keys = set(zip(
            valid_roster_keys["game_id"].astype(str),
            valid_roster_keys["player_id"].astype("int64"),
        ))

    def key_coverage(frame: pd.DataFrame, key_set: set[tuple[str, int]], id_col: str) -> dict[str, Any]:
        eligible = frame.dropna(subset=["game_id", id_col])
        if eligible.empty or not key_set:
            return {"eligible_rows": int(len(eligible)), "matched_rows": 0, "match_rate_pct": None}
        pairs = list(zip(eligible["game_id"].astype(str), eligible[id_col].astype("int64")))
        matched = sum(pair in key_set for pair in pairs)
        return {
            "eligible_rows": len(pairs),
            "matched_rows": int(matched),
            "match_rate_pct": round(matched / len(pairs) * 100.0, 2),
        }

    match_method = (
        players["match_method"].fillna("<NULL>").astype(str).str.casefold().value_counts()
        .to_dict()
        if "match_method" in players else {}
    )
    status_counts = injury["status_normalized"].value_counts().to_dict()
    report_version_counts = (
        injury.dropna(subset=["game_id", "report_datetime"])
        .groupby("game_id")["report_datetime"].nunique()
    )
    per_season = []
    for season, part in injury.groupby("season", dropna=False):
        p = part.loc[~part["is_submission_placeholder"]]
        per_season.append({
            "season_end_year": int(season) if pd.notna(season) else None,
            "report_rows": int(len(part)),
            "player_listing_rows": int(len(p)),
            "unique_report_games": int(part["game_id"].nunique(dropna=True)),
            "unique_report_versions": int(
                part[["game_id", "report_datetime"]].dropna().drop_duplicates().shape[0]
            ),
            "matched_person_id_pct": round(float(p["person_id"].notna().mean() * 100), 2) if len(p) else None,
            "game_id_pct": round(float(part["game_id"].notna().mean() * 100), 2),
            "min_game_date": str(part["game_date"].min().date()) if part["game_date"].notna().any() else None,
            "max_game_date": str(part["game_date"].max().date()) if part["game_date"].notna().any() else None,
        })

    # Status-to-actual-appearance check using the final archived report version
    # for each game/player. This is a data consistency diagnostic, not a
    # pre-tip 60-minute validation; tip-time semantics are audited separately.
    latest = players.sort_values("report_datetime").drop_duplicates(
        ["game_id", "person_id"], keep="last"
    )
    if {"game_id", "person_id", "minutes"}.issubset(box.columns):
        box_check = box[["game_id", "person_id", "minutes"]].copy()
        box_check["minutes_numeric"] = box_check["minutes"].map(parse_minutes)
        box_check["appeared"] = box_check["minutes_numeric"].fillna(0).gt(0)
        box_check = box_check.drop_duplicates(["game_id", "person_id"], keep="last")
        latest = latest.merge(
            box_check[["game_id", "person_id", "appeared", "minutes_numeric"]],
            on=["game_id", "person_id"], how="left", validate="many_to_one",
        )
        latest["boxscore_player_row_found"] = latest["appeared"].notna()
        status_check = (
            latest.groupby("status_normalized", dropna=False)
            .agg(
                latest_report_players=("person_id", "size"),
                boxscore_rows_found=("boxscore_player_row_found", "sum"),
                appeared=("appeared", "sum"),
                appeared_pct=("appeared", "mean"),
            )
            .reset_index()
        )
        status_check["appeared_pct"] = (status_check["appeared_pct"] * 100).round(2)
        status_check = status_check.replace({pd.NA: None}).to_dict("records")
    else:
        status_check = []
        latest = latest.copy()

    # Inspect available schedule time columns; this tells us whether a strict
    # 60-minute report cutoff can be reconstructed without guessing AM/PM.
    time_columns = [
        column for column in schedule.columns
        if re.search(r"(time|datetime|date_time|commence)", column, re.I)
    ]
    report_time_range = {
        "earliest_report_datetime_et_naive": (
            str(injury["report_datetime"].min()) if injury["report_datetime"].notna().any() else None
        ),
        "latest_report_datetime_et_naive": (
            str(injury["report_datetime"].max()) if injury["report_datetime"].notna().any() else None
        ),
    }

    match_methods_summary = {
        str(key): int(value) for key, value in match_method.items()
    }
    return {
        "data_date_range": {
            "first_game_date": str(injury["game_date"].min().date()) if injury["game_date"].notna().any() else None,
            "last_game_date": str(injury["game_date"].max().date()) if injury["game_date"].notna().any() else None,
            **report_time_range,
        },
        "totals": {
            "injury_report_rows_including_placeholders": int(len(injury)),
            "player_listing_rows": int(len(player_listings)),
            "unique_games_in_reports": int(injury["game_id"].nunique(dropna=True)),
            "unique_report_versions_game_level": int(report_version_counts.sum()),
            "median_versions_per_game": round(float(report_version_counts.median()), 2) if len(report_version_counts) else None,
            "mean_versions_per_game": round(float(report_version_counts.mean()), 2) if len(report_version_counts) else None,
            "rows_with_game_id_pct": round(float(injury["game_id"].notna().mean() * 100), 2),
            "rows_with_person_id_pct": round(float(injury["person_id"].notna().mean() * 100), 2),
            "player_listings_with_person_id_pct": round(float(player_listings["person_id"].notna().mean() * 100), 2) if len(player_listings) else None,
            "player_listings_with_game_id_pct": round(float(player_listings["game_id"].notna().mean() * 100), 2) if len(player_listings) else None,
            "not_yet_submitted_or_placeholder_rows": int(injury["is_submission_placeholder"].sum()),
            "distinct_report_dates": int(injury["game_date"].nunique(dropna=True)),
        },
        "join_coverage": {
            "report_game_ids": len(report_games),
            "report_game_ids_found_in_player_boxscores": len(report_games & box_games),
            "report_game_id_boxscore_coverage_pct": round(len(report_games & box_games) / len(report_games) * 100, 2) if report_games else None,
            "report_game_ids_found_in_schedule": len(report_games & schedule_games),
            "report_game_id_schedule_coverage_pct": round(len(report_games & schedule_games) / len(report_games) * 100, 2) if report_games else None,
            "player_id_matches_boxscores": key_coverage(players, box_player_keys, "person_id"),
            "player_id_matches_game_rosters": key_coverage(players, roster_player_keys, "person_id"),
        },
        "match_method_counts": match_methods_summary,
        "status_counts_all_rows": {str(k): int(v) for k, v in status_counts.items()},
        "status_vs_boxscore_latest_archived_listing": status_check,
        "schedule_time_columns_for_cutoff_audit": time_columns,
        "schedule_columns": list(schedule.columns),
        "injury_columns": list(injury.columns),
        "player_boxscore_columns": list(box.columns),
        "game_roster_columns": list(roster.columns),
        "per_season": per_season,
        "limitations": [
            "Status-to-boxscore check above uses the last archived status per game/player, not a validated 60-minute pre-tip snapshot.",
            "No missing player was classified as healthy; only listed report rows were audited.",
            "A non-null person_id is not automatically proof of a correct match; the game/player key joins are measured separately.",
            "Schedule time columns must be interpreted and validated before enforcing the fixed pre-tip cutoff.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2019, help="Season end year.")
    parser.add_argument("--end-season", type=int, default=2027, help="Season end year.")
    parser.add_argument("--output-dir", default="outputs/injury_data_audit")
    args = parser.parse_args()
    if args.start_season > args.end_season:
        raise SystemExit("start-season cannot exceed end-season")
    seasons = list(range(args.start_season, args.end_season + 1))
    output_dir = Path(args.output_dir)
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.github+json", "User-Agent": "nba-injury-audit"})
    downloads: dict[str, Any] = {}
    missing_assets: list[dict[str, Any]] = []
    datasets: dict[str, pd.DataFrame] = {}
    for kind in RELEASES:
        paths, downloaded, missing = download_season_assets(session, download_dir, kind, seasons)
        downloads[kind] = downloaded
        missing_assets.extend(missing)
        datasets[kind] = read_frames(paths, kind)

    report = audit_frames(
        datasets["injury"], datasets["box"], datasets["schedule"], datasets["roster"]
    )
    report["audit_metadata"] = {
        "source_repository": "sportsdataverse/sportsdataverse-data",
        "source_release_tags": RELEASES,
        "seasons_requested": seasons,
        "seasons_missing_any_asset": missing_assets,
        "downloaded_assets": downloads,
        "total_downloaded_bytes": sum(
            item["size_bytes"]
            for group in downloads.values()
            for item in group
        ),
        "audit_scope": "historical source coverage, identifiers, player boxscore joins and status consistency",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "injury_data_audit.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path = output_dir / "injury_data_audit.md"
    md_lines = [
        "# Historical NBA Injury Data Audit",
        "",
        f"Game dates: {report['data_date_range']['first_game_date']} to {report['data_date_range']['last_game_date']}",
        f"Report rows: {report['totals']['injury_report_rows_including_placeholders']:,}",
        f"Player listing rows: {report['totals']['player_listing_rows']:,}",
        f"Unique report games: {report['totals']['unique_games_in_reports']:,}",
        f"Player ID match coverage to boxscores: {report['join_coverage']['player_id_matches_boxscores'].get('match_rate_pct')}%",
        f"Game ID coverage to boxscores: {report['join_coverage']['report_game_id_boxscore_coverage_pct']}%",
        f"Game ID coverage to schedule: {report['join_coverage']['report_game_id_schedule_coverage_pct']}%",
        "",
        "## Per-season coverage",
        "",
        "| Season end year | Rows | Player listings | Games | Versions | Person ID % | Game ID % | Date range |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["per_season"]:
        md_lines.append(
            f"| {row['season_end_year']} | {row['report_rows']:,} | {row['player_listing_rows']:,} | "
            f"{row['unique_report_games']:,} | {row['unique_report_versions']:,} | "
            f"{row['matched_person_id_pct']} | {row['game_id_pct']} | "
            f"{row['min_game_date']} to {row['max_game_date']} |"
        )
    md_lines += [
        "",
        "## Status versus appearance check",
        "",
        "| Report status | Players | Boxscore row found | Appeared | Appeared % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["status_vs_boxscore_latest_archived_listing"]:
        md_lines.append(
            f"| {row['status_normalized']} | {row['latest_report_players']} | "
            f"{row['boxscore_rows_found']} | {row['appeared']} | {row['appeared_pct']} |"
        )
    md_lines += ["", "## Cutoff limitations", ""]
    md_lines.extend(f"- {item}" for item in report["limitations"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_json": str(json_path),
        "audit_markdown": str(md_path),
        "data_date_range": report["data_date_range"],
        "totals": report["totals"],
        "join_coverage": report["join_coverage"],
        "match_method_counts": report["match_method_counts"],
        "status_vs_boxscore_latest_archived_listing": report["status_vs_boxscore_latest_archived_listing"],
        "schedule_time_columns_for_cutoff_audit": report["schedule_time_columns_for_cutoff_audit"],
        "per_season": report["per_season"],
        "missing_assets": missing_assets,
        "downloaded_bytes": report["audit_metadata"]["total_downloaded_bytes"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
