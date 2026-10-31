from datetime import datetime, timezone

import pytest

from research.collect_public_odds_snapshots import merge_snapshot_rows, parse_odds_page


FIXTURE = """<!doctype html>
<html><body>
<h1>Upcoming NBA odds</h1>
<p>Spreads, totals and moneylines for the next week's NBA games.</p>
<p>Lines as of Oct 30, 10:05 AM ET.</p>
<h2>Charlotte Hornets at Washington Wizards</h2>
<p>Fri, Oct 30, 7:00 PM ET</p>
<table>
<thead><tr><th>Book</th><th>CHA ML</th><th>WSH ML</th><th>WSH spread</th><th>Total</th></tr></thead>
<tbody>
<tr><td>DraftKings</td><td>-102</td><td>-118</td><td>-1.5 +102</td><td>219.5 -110/-110</td></tr>
<tr><td>Pinnacle</td><td>-104</td><td>-106</td><td>-1 +100</td><td>220.5 -106/-106</td></tr>
</tbody></table>
<h2>Chicago Bulls at Boston Celtics</h2>
<p>Fri, Oct 30, 7:30 PM ET</p>
<table>
<thead><tr><th>Book</th><th>CHI ML</th><th>BOS ML</th><th>BOS spread</th><th>Total</th></tr></thead>
<tbody><tr><td>DraftKings</td><td>+600</td><td>-900</td><td>-14.5 -115</td><td>234.5 -112/-108</td></tr></tbody>
</table>
</body></html>"""

RETRIEVED = datetime(2026, 10, 30, 15, 10, tzinfo=timezone.utc)


def test_parses_both_side_prices_and_spread_from_same_book_row():
    rows, diagnostics = parse_odds_page(FIXTURE, retrieved_at=RETRIEVED)
    assert len(rows) == 3
    assert diagnostics["games_seen"] == 2
    assert diagnostics["unique_games"] == 2
    assert diagnostics["unique_bookmakers"] == ["DraftKings", "Pinnacle"]

    dk = next(row for row in rows if row["home_team"] == "Washington Wizards")
    assert dk["away_team"] == "Charlotte Hornets"
    assert dk["home_ml_american"] == "-118"
    assert dk["away_ml_american"] == "-102"
    assert dk["home_spread"] == "-1.5"
    assert dk["home_spread_odds_american"] == "102"
    assert dk["total_line"] == "219.5"
    assert dk["over_odds_american"] == "-110"
    assert dk["under_odds_american"] == "-110"
    assert dk["published_as_of_utc"] == "2026-10-30T14:05:00Z"
    assert dk["observed_at_utc"] == "2026-10-30T15:10:00Z"


def test_rejects_stale_page():
    old_retrieval = datetime(2026, 10, 30, 19, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="stale"):
        parse_odds_page(FIXTURE, retrieved_at=old_retrieval, max_page_age_minutes=180)


def test_merge_is_idempotent_for_repeated_page_publication():
    rows, _ = parse_odds_page(FIXTURE, retrieved_at=RETRIEVED)
    merged, inserted = merge_snapshot_rows([], rows)
    assert inserted == len(rows)
    repeated, inserted_again = merge_snapshot_rows(merged, rows)
    assert inserted_again == 0
    assert len(repeated) == len(merged)


def test_merge_rejects_conflicting_prices_at_same_timestamp():
    rows, _ = parse_odds_page(FIXTURE, retrieved_at=RETRIEVED)
    changed = dict(rows[0])
    changed["total_line"] = "999.5"
    with pytest.raises(ValueError, match="Conflicting odds values"):
        merge_snapshot_rows(rows, [changed])


def test_skips_a_scheduled_game_when_no_bookmaker_table_is_posted():
    html = FIXTURE.replace(
        "</body>",
        '<h2>Portland Trail Blazers at Utah Jazz</h2><p>Sat, Oct 31, 9:00 PM ET</p></body>',
    )
    rows, diagnostics = parse_odds_page(html, retrieved_at=RETRIEVED)
    assert len(rows) == 3
    assert diagnostics["games_seen"] == 3
    assert diagnostics["games_without_odds_table"] == 1

def test_rejects_expired_game_instead_of_rolling_it_forward_one_year():
    # The page is fresh, but its Friday game listing is over ten hours old.
    html = FIXTURE.replace(
        "Lines as of Oct 30, 10:05 AM ET.",
        "Lines as of Oct 31, 5:00 AM ET.",
    )
    retrieved = datetime(2026, 10, 31, 9, 28, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="outside the upcoming schedule window"):
        parse_odds_page(html, retrieved_at=retrieved)


def test_resolves_new_year_game_to_the_next_calendar_year():
    from zoneinfo import ZoneInfo
    from research.collect_public_odds_snapshots import parse_game_start

    eastern = ZoneInfo("America/New_York")
    asof = datetime(2026, 12, 31, 18, 0, tzinfo=eastern)
    start = parse_game_start("Fri, Jan 1, 7:00 PM ET", asof)
    assert start.year == 2027
    assert start.month == 1
    assert start.day == 1
