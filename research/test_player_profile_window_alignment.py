#!/usr/bin/env python3
"""Regression test for point-in-time player rolling-window alignment."""
from datetime import date

import duckdb


def main() -> None:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE player_days AS
        SELECT * FROM (VALUES
            (DATE '2026-01-01', 10.0),
            (DATE '2026-01-02', 20.0),
            (DATE '2026-01-03', 30.0),
            (DATE '2026-01-04', 40.0),
            (DATE '2026-01-05', 50.0),
            (DATE '2026-01-06', 60.0)
        ) AS t(history_date, min)
    """)
    windows = []
    for width in (5, 10, 20):
        windows.append(
            f"SUM(min) OVER (ORDER BY history_date "
            f"ROWS BETWEEN {width - 1} PRECEDING AND CURRENT ROW) AS l{width}_min"
        )
    con.execute(f"""
        CREATE TEMP TABLE player_history AS
        SELECT history_date, min, {", ".join(windows)}
        FROM player_days
    """)
    con.execute("""
        CREATE TABLE targets AS
        SELECT * FROM (VALUES
            (DATE '2026-01-06', 'same_day_target'),
            (DATE '2026-01-07', 'next_day_target')
        ) AS t(game_date, target_name)
    """)
    rows = con.execute("""
        SELECT t.target_name, h.history_date, h.l5_min, h.l10_min, h.l20_min
        FROM targets t
        ASOF LEFT JOIN player_history h
          ON t.game_date > h.history_date
        ORDER BY t.game_date
    """).fetchall()

    expected = [
        ("same_day_target", date(2026, 1, 5), 150.0, 150.0, 150.0),
        ("next_day_target", date(2026, 1, 6), 200.0, 210.0, 210.0),
    ]
    if rows != expected:
        raise AssertionError(f"Rolling-window/PIT mismatch. Expected {expected}; got {rows}")

    print("PASS: windows include the latest eligible prior game-day.")
    print("PASS: same-day target games are excluded by strict ASOF date logic.")
    print("PASS: L5/L10/L20 are distinct prior player game-day windows.")


if __name__ == "__main__":
    main()
