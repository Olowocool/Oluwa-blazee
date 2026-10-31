from datetime import datetime

import pandas as pd

import integrated_totals_lab as lab


def _market():
    return {
        "event_id": "test-event",
        "game_date_wat": "2026-11-01",
        "commence_time_utc": "2026-11-01T20:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "market_median_total": 222.5,
        "bookmaker_count": 3,
        "bookmaker_quotes": [],
    }


def test_both_engines_receive_same_target_and_market_line(monkeypatch):
    calls = {}

    def fake_main(**kwargs):
        calls["main"] = kwargs
        return {
            "bookmaker_total": kwargs["bookmaker_total"],
            "projected_total": 224.0,
            "edge": 1.5,
            "recommendation": "No Bet",
            "projected_home_points": 113.0,
            "projected_away_points": 111.0,
            "points_engine_h2h_adjustment": 0.5,
            "points_engine_h2h_games_used": 10,
            "rest_adjustment": 0.0,
            "experimental_projected_total": 226.0,
            "experimental_adjustments_applied": False,
            "history_rows": 123,
        }

    def fake_research(*args, **kwargs):
        calls["research"] = (args, kwargs)
        return {
            "target_date": "2026-11-01",
            "raw_total": 221.0,
            "point_in_time_cutoff": True,
            "composed": {
                "predictions": {
                    "V5 raw": 221.0,
                    "V5 + learned last-10 pace": 222.0,
                    "V5 + legacy H2H 15%": 221.5,
                    "V5 + legacy rest 0.50": 220.5,
                    "V5 + pace + legacy H2H + legacy rest": 222.5,
                    "Selected test-bench configuration": 222.5,
                },
                "selected_prediction": 222.5,
                "pace_adjustment_used": 1.0,
                "h2h_adjustment_used": 0.5,
                "rest_adjustment_used": 0.0,
            },
            "pace": {"available": True, "adjustment": 1.0},
            "h2h": {"legacy_adjustment": 0.5, "learned_era_adjustment": 0.2, "games": 10},
            "rest_adjustments": {
                "legacy_fixed_050": 0.0,
                "learned_shared_0416": 0.0,
                "learned_schedule_density": 0.0,
            },
            "home_rest": {"rest_days": 1},
            "away_rest": {"rest_days": 1},
        }

    monkeypatch.setattr(lab, "predict_game_total", fake_main)
    monkeypatch.setattr(lab, "calculate_game_components", fake_research)

    history = pd.DataFrame({
        "game_date": [pd.Timestamp("2026-10-30")] * 123,
        "home_team": ["Boston Celtics"] * 123,
        "away_team": ["Miami Heat"] * 123,
        "home_score": [110] * 123,
        "away_score": [108] * 123,
    })
    result = lab.evaluate_market_game(
        _market(),
        history,
        {"dummy": True},
        {"boston celtics": "1610612738", "miami heat": "1610612748"},
    )

    assert calls["main"]["bookmaker_total"] == 222.5
    assert calls["main"]["game_date"] == "2026-11-01"
    assert calls["main"]["apply_experimental_adjustments"] is False
    assert calls["research"][0][2] == "2026-11-01"
    assert result["market_line_used_for_both"] == 222.5
    assert result["same_inputs"] is True
    assert result["variant_signals"]["Selected test-bench configuration"]["edge"] == 0.0


def test_integrated_ledger_keeps_main_and_research_outputs_and_no_wager_fields(tmp_path):
    market = _market()
    main = {
        "projected_total": 224.0,
        "edge": 1.5,
        "recommendation": "No Bet",
        "projected_home_points": 113.0,
        "projected_away_points": 111.0,
        "points_engine_h2h_adjustment": 0.5,
        "rest_adjustment": 0.0,
        "experimental_projected_total": 226.0,
    }
    research = {
        "composed": {
            "predictions": {
                "V5 raw": 221.0,
                "V5 + learned last-10 pace": 222.0,
                "V5 + legacy H2H 15%": 221.5,
                "V5 + legacy rest 0.50": 220.5,
                "V5 + pace + legacy H2H + legacy rest": 222.5,
            },
            "h2h_adjustment_used": 0.5,
            "rest_adjustment_used": 0.0,
        },
        "pace": {"adjustment": 1.0},
        "h2h": {"games": 10},
        "home_rest": {"rest_days": 1},
        "away_rest": {"rest_days": 1},
    }
    result = {
        "market": market,
        "market_line_used_for_both": 222.5,
        "main_prediction": main,
        "research_prediction": research,
        "variant_signals": {
            "Selected test-bench configuration": {
                "projected_total": 222.5,
                "edge": 0.0,
                "signal": "No Bet",
            }
        },
        "same_inputs": True,
    }
    row = lab.make_integrated_observation(
        result,
        "2026-10-31T20:00:00Z",
        "pace=True; h2h=Legacy fixed 15%; rest=Legacy fixed 0.50",
    )
    output = tmp_path / "paper.csv"
    saved = lab.append_integrated_observations([row], output)
    assert saved["rows_added"] == 1

    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.loc[0, "main_app_official_total"] == 224.0
    assert frame.loc[0, "v5_raw_total"] == 221.0
    assert frame.loc[0, "market_median_total"] == 222.5
    assert "stake" not in frame.columns
    assert "wager" not in frame.columns
