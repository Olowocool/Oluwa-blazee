from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.objective3_v5_market_residual import (
    american_to_decimal,
    build_game_dataset,
    date_cluster_bootstrap_roi_difference,
    no_vig,
    discover_score_patch_files,
    load_combined_score_history,
    paired_bootstrap_mae_difference,
    prepare_snapshots,
    select_entry_and_last,
    summarize,
    walk_forward,
)


def market_row(key, observed, start, total):
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return {
        "game_key": key,
        "game_date": start_dt.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        "commence_time_utc": start,
        "away_team": "Charlotte Hornets",
        "home_team": "Washington Wizards",
        "bookmaker": "DraftKings",
        "home_ml_american": -150,
        "away_ml_american": 130,
        "home_spread": -3.5,
        "home_spread_odds_american": -110,
        "total_line": total,
        "over_odds_american": -108,
        "under_odds_american": -112,
        "published_as_of_utc": observed,
        "observed_at_utc": observed,
    }


def test_american_price_conversion_and_no_vig():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    over, under, vig = no_vig(-110, -110)
    assert over == pytest.approx(0.5)
    assert under == pytest.approx(0.5)
    assert vig > 0


def test_fixed_entry_uses_only_pre_cutoff_and_retains_later_clv_observation():
    start = "2026-11-01T00:00:00Z"
    rows = [
        market_row("g1", "2026-10-31T19:00:00Z", start, 221.5),
        market_row("g1", "2026-10-31T21:30:00Z", start, 223.5),
        market_row("g1", "2026-10-31T23:00:00Z", start, 225.5),
        market_row("g1", "2026-11-01T00:05:00Z", start, 230.5),
    ]
    clean, _ = prepare_snapshots(pd.DataFrame(rows))
    chosen, diag = select_entry_and_last(clean, entry_minutes=120, max_age_minutes=240)
    assert len(chosen) == 1
    row = chosen.iloc[0]
    assert row.entry_total == 223.5
    assert row.first_observed_total == 221.5
    assert row.move_from_first_observed == 2.0
    assert row.last_observed_total == 225.5
    assert bool(row.has_later_pregame_observation)
    assert diag["games_with_entry_observation"] == 1


def test_stale_page_observation_is_rejected():
    start = "2026-11-01T00:00:00Z"
    row = market_row("g1", "2026-10-31T19:00:00Z", start, 221.5)
    row["observed_at_utc"] = "2026-11-01T00:00:00Z"
    with pytest.raises(ValueError, match="more than 180 minutes old"):
        prepare_snapshots(pd.DataFrame([row]))


def test_walk_forward_excludes_same_day_and_freezes_holdout():
    rows = []
    for day in range(1, 5):
        for game in range(2):
            entry = 220.0 + day + game
            rows.append({
                "game_date": pd.Timestamp(f"2026-11-{day:02d}"),
                "commence_time_utc": pd.Timestamp(f"2026-11-{day:02d}T20:00:00Z"),
                "baseline_edge": float(game),
                "entry_total": entry,
                "home_spread": -2.5,
                "move_from_first_observed": float(day - 1),
                "over_probability_no_vig": 0.5,
                "market_overround": 0.045,
                "entry_over_odds_decimal": 1.93,
                "entry_under_odds_decimal": 1.93,
                "home_spread_price_implied_probability": 0.5238,
                "market_residual_target": float(day + game - 2),
            })
    out = walk_forward(pd.DataFrame(rows), min_train_games=2, alpha=25.0)
    day1 = out.loc[pd.to_datetime(out.game_date).dt.day == 1]
    day2 = out.loc[pd.to_datetime(out.game_date).dt.day == 2]
    assert set(day1.residual_model_status) == {"warmup"}
    assert set(day2.residual_model_status) == {"walk_forward_scored"}
    assert set(day2.residual_training_games) == {2}
    holdout = out.loc[pd.to_datetime(out.game_date).dt.day == 4]
    assert set(holdout.residual_model_status) == {"frozen_holdout_scored"}
    assert set(holdout.residual_training_games) == {6}


def test_holdout_mae_bootstrap_is_paired_and_summary_uses_same_games_for_bets():
    records = []
    for i, status in enumerate([
        "walk_forward_scored", "walk_forward_scored", "frozen_holdout_scored", "warmup"
    ]):
        records.append({
            "game_date": pd.Timestamp(f"2026-11-0{i+1}"),
            "evaluation_split": "reserved_holdout" if i >= 2 else "development",
            "residual_model_status": status,
            "entry_total": 220.0,
            "v5_total": 223.0,
            "residual_model_total": 222.0 if status != "warmup" else np.nan,
            "actual_total": 224.0 + i,
            "v5_side": "Over",
            "v5_settlement": "Loss",
            "v5_profit_units": -1.0,
            "market_residual_model_side": "Over" if status != "warmup" else "",
            "market_residual_model_settlement": "Loss" if status != "warmup" else "No Bet",
            "market_residual_model_profit_units": -1.0 if status != "warmup" else 0.0,
            "last_observed_total": 221.0,
            "last_observed_over_probability_no_vig": 0.52,
            "last_observed_under_probability_no_vig": 0.48,
            "over_probability_no_vig": 0.50,
            "under_probability_no_vig": 0.50,
        })
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    bets = summary.loc[
        (summary.split == "reserved_holdout")
        & summary.predictor.isin(["v5_betting", "market_residual_model_betting"])
    ]
    assert set(bets.games_scored) == {1}
    boot_frame = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-11-01", "2026-11-01", "2026-11-02", "2026-11-02"]),
        "evaluation_split": ["reserved_holdout"] * 4,
        "entry_total": [220.0] * 4,
        "residual_model_total": [223.0] * 4,
        "actual_total": [224.0] * 4,
    })
    boot = paired_bootstrap_mae_difference(boot_frame, n_resamples=500, seed=42)
    assert boot["observed_mae_difference_candidate_minus_baseline"] == pytest.approx(-3.0)
    assert boot["game_dates"] == 2
    assert boot["bootstrap_95_ci"] == pytest.approx([-3.0, -3.0])



def test_score_history_patches_are_merged_and_conflicting_duplicates_fail(tmp_path):
    base = pd.DataFrame([
        {
            "game_date": "2026-10-29",
            "home_team": "Atlanta Hawks",
            "away_team": "Cleveland Cavaliers",
            "home_score": 114,
            "away_score": 111,
        }
    ])
    patch = pd.DataFrame([
        {
            "game_date": "2026-10-30",
            "home_team": "Washington Wizards",
            "away_team": "Charlotte Hornets",
            "home_score": 95,
            "away_score": 79,
        }
    ])
    base_path = tmp_path / "base.csv"
    patch_path = tmp_path / "patch.csv"
    base.to_csv(base_path, index=False)
    patch.to_csv(patch_path, index=False)

    merged, diag = load_combined_score_history(str(base_path), [str(patch_path)])
    assert len(merged) == 2
    assert str(pd.Timestamp(merged["game_date"].max()).date()) == "2026-10-30"
    assert diag["score_history_input_rows"] == 2
    assert diag["score_history_rows_after_dedup"] == 2
    assert diag["score_history_latest_game_date"] == "2026-10-30"

    conflict = patch.copy()
    conflict.loc[0, "home_score"] = 96
    conflict_path = tmp_path / "conflict.csv"
    conflict.to_csv(conflict_path, index=False)
    with pytest.raises(ValueError, match="Conflicting score-history records"):
        load_combined_score_history(str(patch_path), [str(conflict_path)])



def test_default_score_patch_discovery_finds_all_dated_research_patches():
    paths = discover_score_patch_files()
    names = [path.name for path in paths]
    assert names == sorted(names)
    assert "score_history_patch_2026-10-21_to_2026-10-29.csv" in names
    assert "score_history_patch_2026-10-30.csv" in names


def test_score_patch_discovery_ignores_unrelated_csv_files(tmp_path):
    (tmp_path / "score_history_patch_2026-10-31.csv").write_text(
        "game_date,home_team,away_team,home_score,away_score\\n",
        encoding="utf-8",
    )
    (tmp_path / "historical_nba_scores.csv").write_text(
        "this,is,not,a,patch\\n",
        encoding="utf-8",
    )
    discovered = discover_score_patch_files(patch_dir=tmp_path)
    assert [p.name for p in discovered] == ["score_history_patch_2026-10-31.csv"]



def test_prepare_snapshots_excludes_legacy_one_year_future_date_anomaly():
    valid = market_row(
        "valid", "2026-10-31T17:00:00Z", "2026-10-31T20:00:00Z", 220.5
    )
    anomaly = market_row(
        "rolled", "2026-10-31T17:00:00Z", "2027-10-31T20:00:00Z", 220.5
    )
    clean, diag = prepare_snapshots(pd.DataFrame([valid, anomaly]))
    assert clean["game_key"].tolist() == ["valid"]
    assert diag["future_start_anomaly_rows_excluded"] == 1
    assert diag["unique_games"] == 1



def test_v5_market_baseline_uses_only_strictly_prior_score_history(monkeypatch):
    import research.objective3_v5_market_residual as evaluator
    from totals_betting_backtest import game_key
    import points_engine_v5

    target_day = pd.Timestamp("2026-10-30")
    home, away = "Washington Wizards", "Charlotte Hornets"
    prior_day = pd.Timestamp("2026-10-29")
    scores = pd.DataFrame([
        {"game_date": prior_day, "home_team": "Atlanta Hawks", "away_team": "Cleveland Cavaliers",
         "home_score": 114, "away_score": 111, "actual_total": 225,
         "_game_key": game_key(prior_day, "Atlanta Hawks", "Cleveland Cavaliers")},
        {"game_date": target_day, "home_team": home, "away_team": away,
         "home_score": 113, "away_score": 112, "actual_total": 225,
         "_game_key": game_key(target_day, home, away)},
    ])
    monkeypatch.setattr(evaluator, "load_combined_score_history", lambda *args: (scores.copy(), {}))

    observed = {}
    def fake_v5(home_team, away_team, history_df, game_date, _prepared_cache):
        observed["history_dates"] = pd.to_datetime(history_df.game_date).tolist()
        observed["target_day"] = pd.Timestamp(game_date)
        assert pd.to_datetime(history_df.game_date).max() < pd.Timestamp(game_date)
        return {"raw_total": 224.0, "history_cutoff": str(pd.Timestamp(game_date).date())}

    monkeypatch.setattr(points_engine_v5, "calculate_matchup_points_v5", fake_v5)
    selected = pd.DataFrame([{
        "game_date": target_day, "commence_time_utc": pd.Timestamp("2026-10-30T23:00:00Z"),
        "home_team": home, "away_team": away, "entry_total": 220.0,
        "bookmaker": "DraftKings",
    }])
    result, diagnostics = build_game_dataset(selected, "unused-base.csv", [])
    row = result.iloc[0]
    assert row.v5_total == pytest.approx(224.0)
    assert row.baseline_edge == pytest.approx(4.0)
    assert row.market_residual_target == pytest.approx(5.0)
    assert pd.Timestamp(row.prior_max_date) < target_day
    assert pd.Timestamp(row.v5_history_cutoff) == target_day
    assert diagnostics["v5_history_strictly_prior_to_target_date"] is True
    assert diagnostics["baseline_engine"] == "V5 raw (55% V3 + 45% V4-90)"


def test_date_cluster_roi_bootstrap_compares_residual_and_v5_on_common_games():
    frame = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-11-01", "2026-11-01", "2026-11-02", "2026-11-02"]),
        "evaluation_split": ["reserved_holdout"] * 4,
        "market_residual_model_side": ["Over", "Under", "Over", ""],
        "market_residual_model_profit_units": [0.91, -1.0, 0.91, 0.0],
        "v5_side": ["Over", "Under", "", "Over"],
        "v5_profit_units": [0.91, -1.0, 0.0, -1.0],
    })
    result = date_cluster_bootstrap_roi_difference(frame, n_resamples=500, seed=43)
    assert result["n_games"] == 4
    assert result["game_dates"] == 2
    assert result["candidate_bets"] == 3
    assert result["baseline_bets"] == 3
    assert result["roi_difference_candidate_minus_baseline_pp"] == pytest.approx(63.6666667)
