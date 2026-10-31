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

# This script lives in research/, while the engines and data live at repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

from points_engine import (
    calculate_matchup_points,
    history_strictly_before_target_date,
    load_game_history,
)

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

    history = history_strictly_before_target_date(history, TARGET_DATE)
    if history.empty:
        raise RuntimeError("No strictly pregame score history is available.")

    latest = pd.to_datetime(history["game_date"], errors="coerce").max()
    if pd.isna(latest) or latest.date().isoformat() < "2026-10-30":
        raise RuntimeError(
            f"Score history is stale for tonight: latest usable date={latest}"
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
