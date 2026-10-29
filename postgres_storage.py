"""Optional PostgreSQL persistence for live NBA totals audit records.

The existing CSV workflows remain unchanged until the adapter is explicitly
integrated and DATABASE_URL is configured. Importing this module does not
require psycopg unless PostgreSQL is enabled.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping


def database_url() -> str:
    """Return a configured PostgreSQL URL, or an empty string for CSV mode."""
    return (os.getenv("DATABASE_URL") or "").strip()


def postgres_enabled() -> bool:
    return bool(database_url())


def _connect():
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured; CSV mode remains active.")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL is configured but psycopg is not installed. "
            "Install dependencies from requirements.txt."
        ) from exc
    return psycopg.connect(url)


def initialize_schema() -> None:
    """Create versioned audit tables if absent; safe to call repeatedly."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS nba_totals_market_snapshots (
            id BIGSERIAL PRIMARY KEY,
            observed_at_utc TIMESTAMPTZ NOT NULL,
            commence_time_utc TIMESTAMPTZ,
            game_date DATE,
            source_event_id TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            total_line NUMERIC NOT NULL CHECK (total_line > 0),
            over_odds NUMERIC,
            under_odds NUMERIC,
            market_last_update_utc TIMESTAMPTZ,
            source TEXT NOT NULL,
            raw_record JSONB NOT NULL DEFAULT '{}'::jsonb,
            inserted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_nba_totals_snapshots_game_time
        ON nba_totals_market_snapshots
        (game_date, home_team, away_team, observed_at_utc DESC)
        """,
        """
        CREATE TABLE IF NOT EXISTS nba_totals_picks (
            id BIGSERIAL PRIMARY KEY,
            game_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            prediction_time_utc TIMESTAMPTZ,
            commence_time_utc TIMESTAMPTZ,
            market_snapshot_time_utc TIMESTAMPTZ,
            market_last_update_utc TIMESTAMPTZ,
            source_event_id TEXT,
            bookmaker TEXT,
            snapshot_source TEXT,
            projected_total NUMERIC NOT NULL,
            sportsbook_total NUMERIC NOT NULL CHECK (sportsbook_total > 0),
            edge NUMERIC NOT NULL,
            recommendation TEXT NOT NULL,
            pick_type TEXT NOT NULL CHECK (pick_type IN ('Over', 'Under', 'No Bet')),
            over_odds NUMERIC,
            under_odds NUMERIC,
            experimental_adjustments_applied BOOLEAN NOT NULL DEFAULT FALSE,
            stake NUMERIC NOT NULL DEFAULT 100,
            actual_total NUMERIC,
            result TEXT NOT NULL DEFAULT 'Pending',
            profit_loss NUMERIC NOT NULL DEFAULT 0,
            saved_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source_record JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_nba_totals_picks_game_date
        ON nba_totals_picks (game_date, home_team, away_team)
        """,
    ]
    with _connect() as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)


def insert_market_snapshots(records: Iterable[Mapping[str, Any]]) -> int:
    """Append snapshots; never overwrite observations from earlier timestamps."""
    rows = list(records)
    if not rows:
        return 0
    sql = """
        INSERT INTO nba_totals_market_snapshots (
            observed_at_utc, commence_time_utc, game_date, source_event_id,
            home_team, away_team, bookmaker, total_line, over_odds, under_odds,
            market_last_update_utc, source, raw_record
        ) VALUES (
            %(observed_at_utc)s, %(commence_time_utc)s, %(game_date)s,
            %(source_event_id)s, %(home_team)s, %(away_team)s, %(bookmaker)s,
            %(total_line)s, %(over_odds)s, %(under_odds)s,
            %(market_last_update_utc)s, %(source)s, %(raw_record)s
        )
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    return len(rows)


def insert_totals_pick(record: Mapping[str, Any]) -> int:
    """Append one pick with its contemporaneous market/provenance fields."""
    sql = """
        INSERT INTO nba_totals_picks (
            game_date, home_team, away_team, prediction_time_utc,
            commence_time_utc, market_snapshot_time_utc, market_last_update_utc,
            source_event_id, bookmaker, snapshot_source, projected_total,
            sportsbook_total, edge, recommendation, pick_type, over_odds,
            under_odds, experimental_adjustments_applied, stake, source_record
        ) VALUES (
            %(game_date)s, %(home_team)s, %(away_team)s, %(prediction_time_utc)s,
            %(commence_time_utc)s, %(market_snapshot_time_utc)s,
            %(market_last_update_utc)s, %(source_event_id)s, %(bookmaker)s,
            %(snapshot_source)s, %(projected_total)s, %(sportsbook_total)s,
            %(edge)s, %(recommendation)s, %(pick_type)s, %(over_odds)s,
            %(under_odds)s, %(experimental_adjustments_applied)s, %(stake)s,
            %(source_record)s
        ) RETURNING id
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, dict(record))
            row = cur.fetchone()
            return int(row[0])


def load_totals_picks() -> list[dict[str, Any]]:
    """Read persisted picks in insertion order for audit/dashboard integration."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM nba_totals_picks ORDER BY saved_at_utc, id"
            )
            columns = [item.name for item in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
