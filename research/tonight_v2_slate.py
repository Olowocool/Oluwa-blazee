"""One-night, research-only Points Engine V2 totals screen for 2026-10-31.

Uses the repository's historical score file with a strict pregame cutoff.
Does not apply the experimental pace/injury adjustment layer and does not
write to production systems, picks, or the betting ledger.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

# This script lives in research/, while the engines and data live at repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

from team_name_utils import normalize_history_team_columns

from points_engine import (
    calculate_matchup_points,
    history_strictly_before_target_date,
    load_game_history,
)

def fetch_recent_public_scores() -> pd.DataFrame:
    """Read recent completed NBA score rows from the audited public game-log Parquet source."""
    import duckdb

    logs_url = "https://basketball-data.billmill.org/nba/stats/team_game_logs/season=2027/data.parquet"
    teams_url = "https://basketball-data.billmill.org/nba/stats/team_seasons/season=2027/data.parquet"

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        query = """
            WITH logs AS (
                SELECT
                    CAST(game_date AS DATE) AS game_date,
                    season,
                    game_id,
                    team_id,
                    home,
                    pts
                FROM read_parquet(?, hive_partitioning=true)
                WHERE CAST(game_date AS DATE)
                    BETWEEN DATE '2026-10-21' AND DATE '2026-10-30'
            ),
            teams AS (
                SELECT season, team_id, full_name
                FROM read_parquet(?, hive_partitioning=true)
            )
            SELECT
                h.game_date,
                ht.full_name AS home_team,
                away_lookup.full_name AS away_team,
                h.pts AS home_score,
                a.pts AS away_score,
                h.pts + a.pts AS total_score
            FROM logs h
            JOIN logs a
              ON a.season = h.season
             AND a.game_id = h.game_id
             AND a.team_id <> h.team_id
            JOIN teams ht
              ON ht.season = h.season AND ht.team_id = h.team_id
            JOIN teams away_lookup
              ON away_lookup.season = a.season AND away_lookup.team_id = a.team_id
            WHERE h.home IS TRUE AND a.home IS FALSE
            ORDER BY h.game_date, h.game_id
        """
        recent = con.execute(query, [logs_url, teams_url]).fetchdf()
    finally:
        con.close()

    if recent.empty:
        raise RuntimeError("Audited public NBA game-log source returned no recent completed games.")
    recent["game_date"] = pd.to_datetime(recent["game_date"], errors="coerce")
    recent = recent.dropna(subset=["game_date", "home_team", "away_team", "home_score", "away_score"])
    recent = normalize_history_team_columns(recent)
    recent = recent.drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )
    latest = recent["game_date"].max()
    if latest.date().isoformat() != "2026-10-30" or len(recent) < 30:
        raise RuntimeError(
            f"Recent score source incomplete: games={len(recent)}, latest={latest}"
        )
    recent["source"] = "Audited public NBA team_game_logs Parquet"
    return recent


TARGET_DATE = "2026-10-31"
REFERENCE_SOURCE = "FantasyAlarm NBA odds board; 2026-10-31 05:21 ET"
GAMES = [
    ("Utah Jazz", "Portland Trail Blazers", 232.5),
    ("Charlotte Hornets", "Philadelphia 76ers", 219.5),
    ("New Orleans Pelicans", "Atlanta Hawks", 238.5),
    ("Milwaukee Bucks", "Cleveland Cavaliers", 236.5),
    ("Memphis Grizzlies", "Sacramento Kings", 234.5),
    ("San Antonio Spurs", "Minnesota Timberwolves", 226.5),
    ("Houston Rockets", "Oklahoma City Thunder", 218.5),
]


def main() -> None:
    history = load_game_history()
    if history is None or history.empty:
        raise RuntimeError("Historical score history did not load.")

    # The tracked historical file is behind the latest slate. Retrieve
    # completed score rows from the same public NBA Parquet source already
    # used by the repository's advanced-data audit. Keep this patch in-memory.
    recent = fetch_recent_public_scores()
    history = pd.concat([history, recent], ignore_index=True)
    history = normalize_history_team_columns(history)
    history = history.drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )
    history = history_strictly_before_target_date(history, TARGET_DATE)
    if history.empty:
        raise RuntimeError("No strictly pregame score history is available.")

    latest = pd.to_datetime(history["game_date"], errors="coerce").max()
    if pd.isna(latest) or latest.date().isoformat() < "2026-10-30":
        raise RuntimeError(
            f"Recent-score patch failed: latest usable date={latest}"
        )
    if (pd.to_datetime(history["game_date"], errors="coerce") >= pd.Timestamp(TARGET_DATE)).any():
        raise RuntimeError("Leakage guard failed: target-date or future results found.")

    results = []
    for home, away, line in GAMES:
        result = calculate_matchup_points(
            home_team=home,
            away_team=away,
            history_df=history,
            bookmaker_total=line,
            game_date=TARGET_DATE,
        )
        if result["home_games_used"] < 5 or result["away_games_used"] < 5:
            raise RuntimeError(
                f"Insufficient team history for {away} at {home}: "
                f"{result['away_games_used']} / {result['home_games_used']}"
            )
        results.append({
            "away": away,
            "home": home,
            "reference_total": line,
            "projected_away_points": result["projected_away_points"],
            "projected_home_points": result["projected_home_points"],
            "projected_total": result["projected_total"],
            "edge": result["edge"],
            "recommendation": result["recommendation"],
            "away_last5_ppg": result["away_last_5_ppg"],
            "home_last5_ppg": result["home_last_5_ppg"],
            "away_last10_ppg": result["away_last_10_ppg"],
            "home_last10_ppg": result["home_last_10_ppg"],
            "away_opponent_allowed": result["away_opponent_allowed"],
            "home_opponent_allowed": result["home_opponent_allowed"],
            "away_venue_points": result["away_venue_points"],
            "home_venue_points": result["home_venue_points"],
            "away_rest_days": result["away_rest_days"],
            "home_rest_days": result["home_rest_days"],
            "h2h_average_total": result["h2h_average_total"],
            "h2h_games_used": result["h2h_games_used"],
            "history_rows": result["history_rows"],
        })

    output = {
        "target_date": TARGET_DATE,
        "model": "Points Engine V2 (unaltered baseline)",
        "experimental_adjustments_applied": False,
        "historical_rows_after_strict_cutoff": int(len(history)),
        "recent_public_final_games_added": int(len(recent)),
        "recent_public_patch_range": "2026-10-21 through 2026-10-30",
        "latest_historical_game_date": latest.date().isoformat(),
        "reference_line_source": REFERENCE_SOURCE,
        "important_note": (
            "Reference market totals are not guaranteed to match the user's "
            "sportsbook. Recheck the actual line before considering any wager."
        ),
        "games": results,
    }

    out_dir = ROOT / "outputs/research/tonight_v2_slate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tonight_v2_slate.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    print("TONIGHT V2 TOTALS SCREEN")
    print(f"Target date: {TARGET_DATE}")
    print(f"History rows: {len(history):,}; latest usable date: {latest.date()}")
    print(f"Model: {output['model']}; experimental adjustments: OFF")
    print(f"Market reference: {REFERENCE_SOURCE}")
    print()
    print("| Away at Home | V2 projection | Ref line | Edge | V2 signal |")
    print("|---|---:|---:|---:|---|")
    for row in results:
        print(
            f"| {row['away']} at {row['home']} "
            f"| {row['projected_total']:.2f} "
            f"| {row['reference_total']:.1f} "
            f"| {row['edge']:+.2f} "
            f"| {row['recommendation']} |"
        )
    print()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
