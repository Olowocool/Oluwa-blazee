from datetime import datetime, timezone

import pandas as pd
import pytest

from live_market_testbench_engine import (
    _h2h_features,
    _rest_adjustments,
    apply_live_market_freshness,
    apply_player_strength_calibration,
    append_paper_observations,
    compose_predictions,
    evaluate_pace_challenger,
    load_score_history,
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



def test_pace_challenger_reports_paired_date_cluster_metrics(monkeypatch):
    import live_market_testbench_engine as engine

    history = pd.DataFrame([
        ["2025-12-30", "Boston Celtics", "Miami Heat", 110, 100],
        ["2026-03-01", "Boston Celtics", "Miami Heat", 112, 99],
        ["2026-03-01", "New York Knicks", "Chicago Bulls", 101, 110],
        ["2026-03-02", "Boston Celtics", "Miami Heat", 108, 105],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"])
    history["game_date"] = pd.to_datetime(history.game_date)

    def fake_v5(home, away, prior, *, game_date, _prepared_cache):
        assert (prior.game_date < pd.Timestamp(game_date)).all()
        return {"raw_total": 210.0, "history_cutoff": pd.Timestamp(game_date)}

    def fake_pace(home, away, target, pace_history, team_map):
        return {
            "available": True,
            "adjustment": 2.0,
            "expected_matchup_pace_l10": 100.5,
            "league_pace_prior": 99.8,
        }

    monkeypatch.setattr(engine, "calculate_matchup_points_v5", fake_v5)
    monkeypatch.setattr(engine, "pace_component", fake_pace)
    rows, summary = evaluate_pace_challenger(
        history, {"seasons_loaded": [2025, 2026]}, {}, bootstrap_resamples=200, seed=9
    )
    assert len(rows) == 3
    assert summary["evaluation_games"] == 3
    assert summary["evaluation_dates"] == 2
    assert summary["v5_baseline"]["mae"] == pytest.approx(5.0/3.0)
    assert summary["v5_plus_pace"]["mae"] == pytest.approx(1.0)
    assert summary["mae_change_pace_minus_v5"] == pytest.approx(-2.0/3.0)
    assert summary["strict_prior_date_history"] is True


def test_pace_challenger_fails_when_no_supported_games():
    history = pd.DataFrame([
        ["2024-03-01", "Boston Celtics", "Miami Heat", 110, 100],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"])
    history["game_date"] = pd.to_datetime(history.game_date)
    with pytest.raises(ValueError, match="No games could be evaluated"):
        evaluate_pace_challenger(
            history, {"seasons_loaded": [2027]}, {}, bootstrap_resamples=200
        )



def test_player_calibration_uses_training_imputer_for_missing_profile_value(tmp_path):
    import json

    artifact = {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "research_shadow_only",
        "forecast_enabled": False,
        "training_cutoff_exclusive": "2026-07-01",
        "features": [
            "home_off_delta", "home_def_delta", "away_off_delta",
            "away_def_delta", "home_profile_expected_pts",
        ],
        "rapm_feature_nonzero_rows": {
            "home_off_delta": 10, "home_def_delta": 10,
            "away_off_delta": 10, "away_def_delta": 10,
        },
        "imputer": {"statistics": [0.0, 0.0, 0.0, 0.0, 10.0]},
        "standardizer": {
            "mean": [0.0, 0.0, 0.0, 0.0, 10.0],
            "scale": [1.0, 1.0, 1.0, 1.0, 2.0],
        },
        "ridge": {"intercept": 0.5, "coefficients": [0.0, 0.0, 0.0, 0.0, 1.0]},
        "validation": {"gate_passed": False},
    }
    path = tmp_path / "combined_player_calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    result = apply_player_strength_calibration(
        220.0,
        {
            "home_off_delta": 0.1,
            "home_def_delta": 0.2,
            "away_off_delta": 0.3,
            "away_def_delta": 0.4,
            "home_profile_expected_pts": np.nan,
        },
        "2026-11-05",
        path,
    )
    assert result["available"] is True
    assert result["forecast_eligible"] is False
    assert result["player_adjusted_total"] == pytest.approx(219.5)
    assert result["imputed_features"] == ["home_profile_expected_pts"]


def test_player_calibration_applies_only_after_validation_gate(tmp_path):
    import json

    artifact = {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "validated_challenger",
        "forecast_enabled": True,
        "training_cutoff_exclusive": "2026-07-01",
        "features": [
            "home_off_delta", "home_def_delta", "away_off_delta",
            "away_def_delta", "expected_possessions",
        ],
        "rapm_feature_nonzero_rows": {
            "home_off_delta": 40, "home_def_delta": 35,
            "away_off_delta": 38, "away_def_delta": 33,
        },
        "standardizer": {"mean": [0.0, 0.0, 0.0, 0.0, 100.0],
                         "scale": [1.0, 1.0, 1.0, 1.0, 2.0]},
        "ridge": {"intercept": 0.5, "coefficients": [2.0, 0.0, 0.0, 0.0, 1.0]},
        "validation": {"gate_passed": True, "games": 700, "dates": 120},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = apply_player_strength_calibration(
        220.0, {
            "home_off_delta": 1.0, "home_def_delta": 0.0,
            "away_off_delta": 0.0, "away_def_delta": 0.0,
            "expected_possessions": 102.0,
        }, "2026-11-05", path,
    )

    assert result["available"] is True
    assert result["predicted_v5_error"] == pytest.approx(3.5)
    assert result["player_adjusted_total"] == pytest.approx(216.5)


def test_player_calibration_remains_shadow_only_when_validation_gate_fails(tmp_path):
    import json

    artifact = {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "research_shadow_only",
        "forecast_enabled": False,
        "training_cutoff_exclusive": "2026-07-01",
        "features": [
            "home_off_delta", "home_def_delta", "away_off_delta",
            "away_def_delta", "expected_possessions",
        ],
        "rapm_feature_nonzero_rows": {
            "home_off_delta": 5, "home_def_delta": 4,
            "away_off_delta": 6, "away_def_delta": 3,
        },
        "standardizer": {"mean": [0.0, 0.0, 0.0, 0.0, 100.0],
                         "scale": [1.0, 1.0, 1.0, 1.0, 2.0]},
        "ridge": {"intercept": 0.0, "coefficients": [1.0, 0.0, 0.0, 0.0, 0.0]},
        "validation": {"gate_passed": False},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = apply_player_strength_calibration(
        220.0, {
            "home_off_delta": 1.0, "home_def_delta": 0.0,
            "away_off_delta": 0.0, "away_def_delta": 0.0,
            "expected_possessions": 100.0,
        }, "2026-11-05", path,
    )

    assert result["available"] is True
    assert result["forecast_eligible"] is False
    assert result["player_adjusted_total"] == pytest.approx(219.0)


def test_bundled_score_history_applies_verified_patch_priority(tmp_path, monkeypatch):
    import live_market_testbench_engine as engine

    root = tmp_path
    (root / "data").mkdir()
    patch_dir = root / "research" / "data"
    patch_dir.mkdir(parents=True)
    pd.DataFrame([
        ["2026-10-30", "Boston Celtics", "Chicago Bulls", 100, 99],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
        root / "data" / "historical_nba_scores.csv", index=False
    )
    pd.DataFrame([
        ["2026-10-30", "Boston Celtics", "Chicago Bulls", 125, 111],
        ["2026-10-31", "Miami Heat", "Orlando Magic", 110, 101],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
        patch_dir / "score_history_patch_2026-10-30.csv", index=False
    )
    monkeypatch.setattr(engine, "ROOT", root)

    history = engine.load_score_history(patch_dir=patch_dir)

    patched = history[history.game_date.eq(pd.Timestamp("2026-10-30"))].iloc[0]
    assert patched.home_score == 125
    assert patched.away_score == 111
    assert len(history) == 2
    assert history.attrs["source_conflict_overrides"] == 1


def test_uploaded_score_history_overrides_repository_patch(tmp_path):
    base = tmp_path / "uploaded.csv"
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    pd.DataFrame([
        ["2026-10-30", "Boston Celtics", "Chicago Bulls", 130, 110],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
        base, index=False
    )
    pd.DataFrame([
        ["2026-10-30", "Boston Celtics", "Chicago Bulls", 125, 111],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
        patch_dir / "score_history_patch_2026-10-30.csv", index=False
    )

    history = load_score_history(base, patch_dir=patch_dir)

    assert history.iloc[0].home_score == 130
    assert history.iloc[0].away_score == 110


def test_conflicting_score_patches_at_same_priority_fail_closed(tmp_path):
    base = tmp_path / "base.csv"
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    pd.DataFrame([
        ["2026-10-29", "Boston Celtics", "Chicago Bulls", 100, 99],
    ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
        base, index=False
    )
    for name, home_score in [("a", 120), ("b", 121)]:
        pd.DataFrame([
            ["2026-10-30", "Boston Celtics", "Chicago Bulls", home_score, 110],
        ], columns=["game_date", "home_team", "away_team", "home_score", "away_score"]).to_csv(
            patch_dir / f"score_history_patch_{name}.csv", index=False
        )

    with pytest.raises(ValueError, match="Conflicting scores within the same source priority"):
        load_score_history(base, patch_dir=patch_dir)


def test_stale_player_calibration_contract_fails_closed_without_stopping_diagnostics(tmp_path):
    import json

    names = [
        "home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta",
        "raw_injury_total_delta", "expected_possessions",
    ]
    artifact = {
        "schema_version": "stage-c-live-player-strength-calibration-v1",
        "status": "research_shadow_only",
        "forecast_enabled": False,
        "training_cutoff_exclusive": "2026-07-01",
        "features": names,
        "rapm_feature_nonzero_rows": {name: 5 for name in names[:4]},
        "standardizer": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "ridge": {"intercept": 0.0, "coefficients": [0.1] * 6},
        "validation": {"gate_passed": False},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    result = apply_player_strength_calibration(
        220.0, {
            "home_off_delta": 1.0, "home_def_delta": 0.0,
            "away_off_delta": 0.0, "away_def_delta": 0.0,
        }, "2026-11-05", path,
    )

    assert result["available"] is False
    assert "feature contract differs" in result["reason"]


def test_live_quote_requires_fresh_observation_and_publication_and_future_tip():
    now = pd.Timestamp("2026-11-05T10:00:00Z")
    quotes = pd.DataFrame([
        {
            "observed_at_utc": "2026-11-05T09:50:00Z",
            "published_as_of_utc": "2026-11-05T09:45:00Z",
            "commence_time_utc": "2026-11-05T12:00:00Z",
        },
        {
            "observed_at_utc": "2026-11-05T09:55:00Z",
            "published_as_of_utc": "2026-11-05T09:25:00Z",
            "commence_time_utc": "2026-11-05T12:00:00Z",
        },
        {
            "observed_at_utc": "2026-11-05T09:30:00Z",
            "published_as_of_utc": "2026-11-05T09:55:00Z",
            "commence_time_utc": "2026-11-05T12:00:00Z",
        },
        {
            "observed_at_utc": "2026-11-05T09:50:00Z",
            "published_as_of_utc": "2026-11-05T09:45:00Z",
            "commence_time_utc": "2026-11-05T09:59:00Z",
        },
    ])

    result = apply_live_market_freshness(quotes, now_utc=now, max_age_minutes=20)

    assert result["Live quote eligible"].tolist() == [True, False, False, False]
    assert result.loc[0, "Observation age (min)"] == pytest.approx(10.0)
    assert result.loc[0, "Publication age (min)"] == pytest.approx(15.0)
