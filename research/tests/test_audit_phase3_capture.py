import pandas as pd
import pytest
from research.audit_phase3_capture import build_capture_audit


def row(key, start, observed, book="DraftKings"):
    t = pd.Timestamp(observed, tz="UTC")
    return {
        "game_key": key, "away_team": "Away", "home_team": "Home", "bookmaker": book,
        "commence_time_utc": pd.Timestamp(start, tz="UTC"),
        "published_as_of_utc": t, "observed_at_utc": t,
    }


def test_cutoff_uses_local_observation_and_counts_later_only_rows():
    rows = [
        row("fresh", "2026-10-31 20:00", "2026-10-31 15:00"),
        row("fresh", "2026-10-31 20:00", "2026-10-31 17:00"),
        row("fresh", "2026-10-31 20:00", "2026-10-31 19:00"),
        row("stale", "2026-10-31 20:00", "2026-10-31 14:00"),
        row("none", "2026-10-31 20:00", "2026-10-31 18:30"),
    ]
    result = build_capture_audit(pd.DataFrame(rows), audit_time=pd.Timestamp("2026-10-31 20:30", tz="UTC"))
    assert result["entry_status_counts"] == {"fresh_entry": 1, "stale_entry": 1, "no_entry": 1}
    fresh = next(g for g in result["games"] if g["status"] == "fresh_entry")
    assert fresh["entry_age_minutes"] == 180
    assert fresh["later_pregame_rows"] == 1
    stale = next(g for g in result["games"] if g["status"] == "stale_entry")
    assert stale["entry_age_minutes"] == 360


def test_other_books_and_post_start_rows_are_not_counted_as_pregame():
    rows = [
        row("g", "2026-10-31 20:00", "2026-10-31 17:00"),
        row("g", "2026-10-31 20:00", "2026-10-31 20:01"),
        row("other", "2026-10-31 20:00", "2026-10-31 17:00", "Pinnacle"),
    ]
    result = build_capture_audit(pd.DataFrame(rows), audit_time=pd.Timestamp("2026-10-31 21:00", tz="UTC"))
    assert result["unique_games"] == 1
    assert result["post_start_rows_excluded"] == 1
    assert result["pregame_rows_after_dedup"] == 1


def test_rejects_naive_audit_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_capture_audit(
            pd.DataFrame([row("g", "2026-10-31 20:00", "2026-10-31 17:00")]),
            audit_time=pd.Timestamp("2026-10-31 20:30"),
        )


def test_rejects_missing_columns_and_invalid_rules():
    with pytest.raises(ValueError, match="missing required"):
        build_capture_audit(pd.DataFrame([{"game_key": "g"}]))
    with pytest.raises(ValueError, match="Require"):
        build_capture_audit(
            pd.DataFrame([row("g", "2026-10-31 20:00", "2026-10-31 17:00")]),
            max_age_minutes=60,
        )


def test_duplicate_publication_keeps_earliest_observed_time():
    first = row("g", "2026-10-31 20:00", "2026-10-31 17:10")
    second = row("g", "2026-10-31 20:00", "2026-10-31 17:00")
    second["published_as_of_utc"] = first["published_as_of_utc"]
    result = build_capture_audit(
        pd.DataFrame([first, second]), audit_time=pd.Timestamp("2026-10-31 20:30", tz="UTC")
    )
    assert result["duplicate_pregame_rows_removed_for_audit"] == 1
    assert result["games"][0]["entry_observed_utc"] == "2026-10-31T17:00:00Z"
