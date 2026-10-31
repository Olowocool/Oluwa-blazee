from datetime import datetime, timezone

import pandas as pd
import pytest

from live_market_testbench_engine import (
    _h2h_features,
    _rest_adjustments,
    append_paper_observations,
    compose_predictions,
    parse_current_totals_markets,
    team_schedule_features,
)


def test_market_parser_accepts_same_book_same_line_and_ignores_mismatched_pair():
    now = datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)
    events = [{
        "id": "event-1",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "commence_time": "2026-10-31T20:00:00Z",
        "bookmakers": [
            {
                "key": "bad-book",
                "title": "Bad Book",
                "markets": [{
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 220.5, "price": 1.91},
                        {"name": "Under", "point": 221.5, "price": 1.91},
                    ],
                }],
            },
            {
                "key": "good-book",
                "title": "Good Book",
                "markets": [{
                    "key": "totals",
                    "last_update": "2026-10-31T09:50:00Z",
                    "outcomes": [
                        {"name": "Over", "point": 221.5, "price": 1.95},
                        {"name": "Under", "point": 221.5, "price": 1.87},
                    ],
                }],
            },
        ],
    }, {
        "id": "event-live",
        "home_team": "Atlanta Hawks",
        "away_team": "Orlando Magic",
        "commence_time": "2026-10-31T09:00:00Z",
        "bookmakers": [],
    }]

    markets = parse_current_totals_markets(events, now=now)
    assert len(markets) == 1
    assert markets[0]["event_id"] == "event-1"
    assert markets[0]["market_median_total"] == pytest.approx(221.5)
    assert markets[0]["bookmaker_count"] == 1
    assert markets[0]["bookmaker_quotes"][0]["bookmaker"] == "Good Book"


def test_rest_and_congestion_features_use_prior_dates_only():
    history = pd.DataFrame([
        ["2026-10-27", "Boston Celtics", "Chicago Bulls", 110, 100],
        ["2026-10-28", "Miami Heat", "Boston Celtics", 108, 104],
        ["2026-10-29", "Boston Celtics", "Brooklyn Nets", 112, 101],
        # Same-day outcome and a future game must not alter the input.
        ["2026-10-31", "Boston Celtics", "New York Knicks", 1, 1],
        ["2026-11-02", "Boston Celtics", "Cleveland Cavaliers", 1, 1],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"])
    history["game_date"] = pd.to_datetime(history.game_date)
    history["total_score"] = history.home_score + history.away_score

    features = team_schedule_features(history, "Boston Celtics", "2026-10-31")
    assert features["last_game_date"] == pd.Timestamp("2026-10-29")
    assert features["rest_days"] == 1
    assert features["games_prev_3d"] == 2
    assert features["games_prev_5d"] == 3
    assert features["games_prev_14d"] == 3


def test_h2h_uses_only_prior_meetings_and_last_ten():
    rows = []
    for index in range(11):
        day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=2 * index)
        rows.append([
            day, "Boston Celtics", "Miami Heat", 110 + index, 100,
        ])
    rows.extend([
        [pd.Timestamp("2026-10-31"), "Boston Celtics", "Miami Heat", 300, 300],
        [pd.Timestamp("2026-11-02"), "Boston Celtics", "Miami Heat", 300, 300],
    ])
    history = pd.DataFrame(
        rows, columns=["game_date", "home_team", "away_team", "home_score", "away_score"]
    )
    history["total_score"] = history.home_score + history.away_score

    result = _h2h_features(
        history, "Boston Celtics", "Miami Heat", "2026-10-31", raw_total=220.0
    )
    expected_totals = [210 + i for i in range(1, 11)]
    assert result["games"] == 10
    assert result["latest_date"] < pd.Timestamp("2026-10-31")
    assert result["average_total_last10"] == pytest.approx(sum(expected_totals) / 10)
    assert result["legacy_adjustment"] == pytest.approx(
        0.15 * (sum(expected_totals) / 10 - 220.0)
    )


def test_rest_adjustments_return_research_variants():
    home = {
        "rest_days": 0,
        "games_prev_3d": 2,
        "games_prev_5d": 3,
    }
    away = {
        "rest_days": 1,
        "games_prev_3d": 0,
        "games_prev_5d": 1,
    }
    result = _rest_adjustments(home, away)
    assert result["legacy_fixed_050"] == pytest.approx(-0.5)
    assert result["learned_shared_0416"] == pytest.approx(-0.416157)
    expected_schedule = (
        0.642787 * -1
        + 0.073308 * 0
        + 0.315415 * 1
        + 0.462948 * 0
        - 0.065809 * 1
        + 0.432959 * 0
    )
    assert result["learned_schedule_density"] == pytest.approx(expected_schedule)


def test_composition_can_disable_each_component():
    pace = {"available": True, "adjustment": 1.25}
    h2h = {"legacy_adjustment": 2.0, "learned_era_adjustment": 0.25}
    rest = {
        "legacy_fixed_050": -0.5,
        "learned_shared_0416": -0.4,
        "learned_schedule_density": 0.7,
    }
    result = compose_predictions(
        220.0, pace, h2h, rest,
        use_pace=False, h2h_mode="off", rest_mode="off",
    )
    assert result["selected_prediction"] == pytest.approx(220.0)
    assert result["pace_adjustment_used"] == 0
    assert result["h2h_adjustment_used"] == 0
    assert result["rest_adjustment_used"] == 0


def test_paper_observation_append_only(tmp_path):
    path = tmp_path / "paper.csv"
    row = {
        "observed_at_utc": "2026-10-31T12:00:00Z",
        "event_id": "event-1",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "market_median_total": 220.5,
        "v5_raw_total": 219.0,
        "paper_status": "unsettled",
    }
    first = append_paper_observations([row], path)
    second = append_paper_observations([row], path)
    assert first["rows_added"] == 1
    assert second["rows_added"] == 1
    saved = pd.read_csv(path)
    assert len(saved) == 2
    assert "stake" not in saved.columns
    assert "wager" not in saved.columns
