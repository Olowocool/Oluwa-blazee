import os
import uuid
import math
from datetime import datetime, timezone

import pandas as pd

from durable_totals_storage import persist_records, sync_csv_mirror


TOTALS_HISTORY_FILE = "totals_history.csv"
STAKE = 100


def _normalize_aware_utc_timestamp(value):
    """Normalize an explicit timezone-aware timestamp; reject naive timestamps."""
    if value is None or not str(value).strip():
        return ""
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_totals_pick(
    game_date,
    home_team,
    away_team,
    projected_total,
    sportsbook_total,
    recommendation,
    prediction_time_utc=None,
    commence_time_utc=None,
    market_snapshot_time_utc=None,
    market_last_update_utc=None,
    market_snapshot_total_line=None,
    over_odds=None,
    under_odds=None,
    totals_bookmaker=None,
    snapshot_source=None,
    source_event_id=None,
    experimental_adjustments_applied=False,
):
    if isinstance(sportsbook_total, bool):
        raise ValueError("A valid sportsbook total line is required to save a totals pick.")
    try:
        line = float(sportsbook_total)
        projection = float(projected_total)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("A valid sportsbook total line and projection are required to save a totals pick.") from exc
    if not math.isfinite(line) or line <= 0 or not math.isfinite(projection):
        raise ValueError("A finite, positive sportsbook total line and projection are required.")
    edge = projection - line

    if "Under" in recommendation:
        pick_type = "Under"
    elif "Over" in recommendation:
        pick_type = "Over"
    else:
        pick_type = "No Bet"

    row = {
        "pick_id": uuid.uuid4().hex,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        # Historical rows without these timestamps remain auditable but cannot
        # be used for model training because pre-tipoff provenance is unverified.
        "prediction_time_utc": _normalize_aware_utc_timestamp(prediction_time_utc),
        "commence_time_utc": _normalize_aware_utc_timestamp(commence_time_utc),
        "market_snapshot_time_utc": _normalize_aware_utc_timestamp(market_snapshot_time_utc),
        "market_last_update_utc": _normalize_aware_utc_timestamp(market_last_update_utc),
        "market_snapshot_total_line": market_snapshot_total_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "totals_bookmaker": totals_bookmaker or "",
        "market_snapshot_source": snapshot_source or "",
        "source_event_id": source_event_id or "",
        "saved_at_utc": _utc_now_iso(),
        "experimental_adjustments_applied": bool(experimental_adjustments_applied),
        "market": "Totals",
        "pick_type": pick_type,
        "projected_total": projected_total,
        "sportsbook_total": sportsbook_total,
        "saved_total": sportsbook_total,
        "closing_total": None,
        "clv": None,
        "edge": round(edge, 2),
        "recommendation": recommendation,
        "stake": STAKE,
        "actual_total": None,
        "result": "Pending",
        "profit_loss": 0
    }

    sync_csv_mirror("totals_pick", TOTALS_HISTORY_FILE)

    if os.path.exists(TOTALS_HISTORY_FILE):
        df = pd.read_csv(TOTALS_HISTORY_FILE)
    else:
        df = pd.DataFrame()

    # The prospective paper sample records only the first qualifying forecast
    # for each NBA event. Streamlit reruns and repeated button clicks must not
    # turn one game into several apparently independent bets.
    event_id = str(row.get("source_event_id") or "").strip()
    if event_id and not df.empty and "source_event_id" in df.columns:
        existing_event_ids = df["source_event_id"].fillna("").astype(str).str.strip()
        same_event = existing_event_ids.eq(event_id)
        for column in ("home_team", "away_team"):
            if column in df.columns:
                same_event &= (
                    df[column].fillna("").astype(str).str.strip().str.casefold()
                    == str(row[column]).strip().casefold()
                )
        if "game_date" in df.columns:
            same_event &= (
                df["game_date"].fillna("").astype(str).str.strip()
                == str(game_date).strip()
            )
        if same_event.any():
            return False

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True
    )

    save_totals_history(df)
    return True


def load_totals_history():
    sync_csv_mirror("totals_pick", TOTALS_HISTORY_FILE)
    if not os.path.exists(TOTALS_HISTORY_FILE):
        return pd.DataFrame()

    df = pd.read_csv(TOTALS_HISTORY_FILE)

    for col in [
        "market",
        "pick_type",
        "edge",
        "stake",
        "saved_total",
        "closing_total",
        "clv",
        "actual_total",
        "result",
        "profit_loss"
    ]:
        if col not in df.columns:
            df[col] = None

    return df


def save_totals_history(df):
    df = df.copy()
    if "pick_id" not in df.columns:
        df["pick_id"] = [uuid.uuid4().hex for _ in range(len(df))]
    else:
        missing_ids = df["pick_id"].isna() | df["pick_id"].astype(str).str.strip().eq("")
        for index in df.index[missing_ids]:
            df.loc[index, "pick_id"] = uuid.uuid4().hex

    records = df.to_dict(orient="records")
    persist_records("totals_pick", records)

    os.makedirs(os.path.dirname(TOTALS_HISTORY_FILE) or ".", exist_ok=True)
    df.to_csv(TOTALS_HISTORY_FILE, index=False)
