from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import pandas as pd

from durable_totals_storage import persist_records, sync_csv_mirror

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
HISTORY_FILE = OUTPUT_DIR / "prediction_history.csv"
INJURY_HISTORY_DIR = BASE_DIR / "data" / "injury_history"


def latest_injury_snapshot():
    if not INJURY_HISTORY_DIR.exists():
        return ""

    files = sorted(INJURY_HISTORY_DIR.glob("injuries_*.csv"))
    return str(files[-1].resolve()) if files else ""


def _normalize_aware_utc_timestamp(value):
    if value is None or not str(value).strip():
        return ""
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log_prediction(
    prediction,
    game_date=None,
    commence_time_utc=None,
    market_snapshot_time_utc=None,
    market_last_update_utc=None,
    market_snapshot_total_line=None,
    over_odds=None,
    under_odds=None,
    totals_bookmaker=None,
    snapshot_source=None,
    source_event_id=None,
):
    required = [
        "home_team",
        "away_team",
        "bookmaker_total",
        "projected_total",
    ]
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"Missing prediction fields: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "prediction_id": uuid.uuid4().hex,
        "prediction_time": datetime.now().astimezone().isoformat(),
        "prediction_time_utc": _normalize_aware_utc_timestamp(
            prediction.get("prediction_time_utc")
        ),
        "commence_time_utc": _normalize_aware_utc_timestamp(commence_time_utc),
        "market_snapshot_time_utc": _normalize_aware_utc_timestamp(
            market_snapshot_time_utc or prediction.get("market_snapshot_time_utc")
        ),
        "market_last_update_utc": _normalize_aware_utc_timestamp(
            market_last_update_utc or prediction.get("market_last_update_utc")
        ),
        "market_snapshot_total_line": (
            market_snapshot_total_line
            if market_snapshot_total_line is not None
            else prediction.get("market_snapshot_total_line")
        ),
        "over_odds": over_odds if over_odds is not None else prediction.get("over_odds"),
        "under_odds": under_odds if under_odds is not None else prediction.get("under_odds"),
        "totals_bookmaker": totals_bookmaker or prediction.get("totals_bookmaker", ""),
        "snapshot_source": snapshot_source or prediction.get("snapshot_source", ""),
        "source_event_id": source_event_id or prediction.get("source_event_id", ""),
        "game_date": game_date or "",
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "bookmaker_total": prediction["bookmaker_total"],
        "projected_home_points": prediction.get("projected_home_points"),
        "projected_away_points": prediction.get("projected_away_points"),
        "projected_total": prediction["projected_total"],
        "edge": prediction.get("edge"),
        "recommendation": prediction.get("recommendation"),
        "home_injury_penalty": prediction.get("home_injury_penalty", 0),
        "away_injury_penalty": prediction.get("away_injury_penalty", 0),
        "injury_adjustment": prediction.get("injury_adjustment", 0),
        "home_missing_players": json.dumps(
            prediction.get("home_missing_players", []),
            ensure_ascii=False,
        ),
        "away_missing_players": json.dumps(
            prediction.get("away_missing_players", []),
            ensure_ascii=False,
        ),
        "injury_snapshot_path": latest_injury_snapshot(),
        "actual_total": None,
        "total_error": None,
    }

        # PostgreSQL becomes authoritative when configured. In strict durable mode,
    # a missing database setting fails closed before any new audit row is saved.
    database_active = sync_csv_mirror("prediction_audit", HISTORY_FILE)

    # Prevent duplicate records for the same game, line, and recommendation.
    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE)

        required_columns = {
            "game_date",
            "home_team",
            "away_team",
            "bookmaker_total",
            "recommendation",
        }

        if not existing.empty and required_columns.issubset(existing.columns):
            same_game = (
                existing["game_date"].astype(str).eq(str(row["game_date"]))
                & existing["home_team"].astype(str).eq(str(row["home_team"]))
                & existing["away_team"].astype(str).eq(str(row["away_team"]))
            )

            same_line = pd.to_numeric(
                existing["bookmaker_total"], errors="coerce"
            ).eq(float(row["bookmaker_total"]))

            same_recommendation = existing["recommendation"].astype(str).eq(
                str(row["recommendation"])
            )

            if (same_game & same_line & same_recommendation).any():
                print("Duplicate prediction detected; not logged.")
                return row

    # Persist to the durable ledger first. A database failure must not be
    # hidden by a successful write to an ephemeral hosted filesystem.
    persist_records("prediction_audit", [row])

    if database_active:
        sync_csv_mirror("prediction_audit", HISTORY_FILE)
    else:
        if HISTORY_FILE.exists():
            existing = pd.read_csv(HISTORY_FILE)
            df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
        else:
            df = pd.DataFrame([row])
        df.to_csv(HISTORY_FILE, index=False)

    print(f"Prediction logged: {HISTORY_FILE}")
    print(f"Matchup: {row['away_team']} at {row['home_team']}")
    print(f"Projected total: {row['projected_total']}")
    return row

if __name__ == "__main__":
    print("Prediction history logger is ready.")
    print(f"Output file: {HISTORY_FILE}")
    print("No prediction was logged. Import log_prediction() to record real predictions.")

