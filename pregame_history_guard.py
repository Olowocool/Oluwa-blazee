"""Point-in-time history guards for NBA inference.

All team-form features used for a forecast must be built strictly from games
whose Lagos calendar date precedes the target game's Lagos calendar date.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

LAGOS_TIMEZONE = ZoneInfo("Africa/Lagos")


def parse_target_calendar_date(value=None) -> date:
    """Parse a selected date or default to today's Lagos calendar date."""
    if value is None or not str(value).strip():
        return datetime.now(LAGOS_TIMEZONE).date()

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(LAGOS_TIMEZONE).date()
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid game date. Use MM/DD/YYYY or YYYY-MM-DD."
        ) from exc

    if pd.isna(parsed):
        raise ValueError("Invalid game date. Use MM/DD/YYYY or YYYY-MM-DD.")
    if parsed.tzinfo is not None:
        return parsed.tz_convert(LAGOS_TIMEZONE).date()
    return parsed.date()


def history_strictly_before_target_date(history_df, game_date=None):
    """Return historical rows strictly before the target date (Lagos calendar).

    The target calendar day is excluded in full, avoiding accidental leakage
    from same-day completed games into predictions for later games that day.
    Invalid/missing row dates are dropped. A missing date column or invalid
    target date raises rather than passing unbounded history through.
    """
    if history_df is None:
        return pd.DataFrame()
    if history_df.empty:
        return history_df.copy()

    date_col = "date" if "date" in history_df.columns else (
        "game_date" if "game_date" in history_df.columns else None
    )
    if date_col is None:
        raise ValueError(
            "Cannot enforce the pregame history cutoff: no date/game_date column."
        )

    target_date = parse_target_calendar_date(game_date)
    parsed_utc = pd.to_datetime(history_df[date_col], errors="coerce", utc=True)
    local_times = parsed_utc.dt.tz_convert(LAGOS_TIMEZONE)
    local_days = local_times.dt.date
    mask = local_times.notna() & (local_days < target_date)

    eligible = history_df.loc[mask].copy()
    # Ensure downstream chronological sorting compares normalized timestamps
    # rather than a mixture of strings, dates, and timezone-aware values.
    eligible["date"] = local_times.loc[mask].dt.tz_localize(None)
    return eligible
