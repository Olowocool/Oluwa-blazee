import os
from datetime import datetime, timezone

import pandas as pd


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
    experimental_adjustments_applied=False,
):
    edge = float(projected_total) - float(sportsbook_total)

    if "Under" in recommendation:
        pick_type = "Under"
    elif "Over" in recommendation:
        pick_type = "Over"
    else:
        pick_type = "No Bet"

    row = {
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        # Historical rows without these timestamps remain auditable but cannot
        # be used for model training because pre-tipoff provenance is unverified.
        "prediction_time_utc": _normalize_aware_utc_timestamp(prediction_time_utc),
        "commence_time_utc": _normalize_aware_utc_timestamp(commence_time_utc),
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

    if os.path.exists(TOTALS_HISTORY_FILE):
        df = pd.read_csv(TOTALS_HISTORY_FILE)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True
    )

    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )

    return True


def load_totals_history():
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
    df.to_csv(
        TOTALS_HISTORY_FILE,
        index=False
    )
