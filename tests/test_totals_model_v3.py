import pandas as pd

from totals_model_v3 import (
    calculate_team_schedule_state,
    estimate_pregame_ratings,
    predict_game_total_v3,
    prepare_team_game_logs,
    run_walk_forward_backtest,
)


def make_raw_games(start_date="2021-10-20", count=32, season_id="22021"):
    rows = []
    start = pd.Timestamp(start_date)
    for index in range(count):
        game_date = start + pd.Timedelta(days=2 * index)
        game_id = f"{season_id}-{index:04d}"
        home_points = 108 + (index % 7)
        away_points = 103 + (index % 5)
        rows.extend([
            {
                "GAME_ID": game_id,
                "GAME_DATE": game_date.strftime("%Y-%m-%d"),
                "SEASON_ID": season_id,
                "TEAM_NAME": "Team A",
                "MATCHUP": "Team A vs. Team B",
                "PTS": home_points,
                "FGA": 88 + (index % 3),
                "FTA": 21 + (index % 4),
                "OREB": 10 + (index % 2),
                "TOV": 12 + (index % 3),
                "FG3A": 35 + (index % 5),
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": game_date.strftime("%Y-%m-%d"),
                "SEASON_ID": season_id,
                "TEAM_NAME": "Team B",
                "MATCHUP": "Team B @ Team A",
                "PTS": away_points,
                "FGA": 86 + (index % 4),
                "FTA": 19 + (index % 3),
                "OREB": 9 + (index % 2),
                "TOV": 13 + (index % 2),
                "FG3A": 33 + (index % 6),
            },
        ])
    return pd.DataFrame(rows)


def test_true_possessions_use_box_score_formula():
    raw = make_raw_games(count=1)
    raw.loc[raw["TEAM_NAME"] == "Team A", ["FGA", "FTA", "OREB", "TOV"]] = [85, 20, 10, 12]

    logs = prepare_team_game_logs(raw)
    home = logs.loc[logs["team_name"] == "Team A"].iloc[0]

    assert abs(home["possessions"] - 95.8) < 1e-9
    assert home["possessions_source"] == "box_score_formula"


def test_source_possessions_are_fallback_when_formula_columns_are_missing():
    raw = make_raw_games(count=1).drop(columns=["FGA", "FTA", "OREB", "TOV"])
    raw["POSS"] = [101.2, 99.8]

    logs = prepare_team_game_logs(raw)

    assert sorted(logs["possessions"].round(1).tolist()) == [99.8, 101.2]
    assert set(logs["possessions_source"]) == {"source_poss_fallback"}


def test_opponent_link_and_home_away_are_built_from_matchup():
    logs = prepare_team_game_logs(make_raw_games(count=1))

    home = logs.loc[logs["team_name"] == "Team A"].iloc[0]
    away = logs.loc[logs["team_name"] == "Team B"].iloc[0]

    assert home["opponent_name"] == "Team B"
    assert away["opponent_name"] == "Team A"
    assert bool(home["is_home"]) is True
    assert bool(away["is_home"]) is False


def test_playoff_season_id_is_not_mixed_into_regular_season_training():
    raw = make_raw_games(count=2, season_id="22021")
    playoff = make_raw_games(
        start_date="2022-04-20", count=1, season_id="42021"
    )
    logs = prepare_team_game_logs(pd.concat([raw, playoff], ignore_index=True))

    assert len(logs) == 4
    assert set(logs["season"]) == {2021}


def test_pregame_ratings_exclude_target_day_and_future_games():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=12, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=16, season_id="22021"
    )
    raw = pd.concat([prior_season, current_season], ignore_index=True)
    target_date = pd.Timestamp("2021-11-13")

    full_logs = prepare_team_game_logs(raw)
    strict_logs = full_logs.loc[full_logs["game_date"] < target_date].copy()

    from_full = estimate_pregame_ratings(full_logs, target_date)
    from_strict = estimate_pregame_ratings(strict_logs, target_date)

    assert from_full["history_rows_used"] == len(strict_logs)
    assert from_full["history_rows_used"] == from_strict["history_rows_used"]
    assert from_full["teams"] == from_strict["teams"]


def test_prepared_log_fast_path_matches_full_preparation():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=18, season_id="22021"
    )
    raw = pd.concat([prior_season, current_season], ignore_index=True)
    prepared = prepare_team_game_logs(raw)
    target_date = "2021-12-15"

    normal = estimate_pregame_ratings(prepared, target_date)
    fast = estimate_pregame_ratings(
        prepared, target_date, prepared_logs=True
    )

    assert normal["history_rows_used"] == fast["history_rows_used"]
    assert normal["teams"] == fast["teams"]
    assert normal["league_efficiency"] == fast["league_efficiency"]
    assert normal["league_pace"] == fast["league_pace"]
    assert normal["shot_betas"] == fast["shot_betas"]
    assert normal["schedule_betas"] == fast["schedule_betas"]


def test_v3_prediction_is_finite_and_team_points_sum_to_total():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=18, season_id="22021"
    )
    logs = prepare_team_game_logs(
        pd.concat([prior_season, current_season], ignore_index=True)
    )

    prediction = predict_game_total_v3(
        logs, "Team A", "Team B", "2021-12-15"
    )

    assert prediction["model_version"] == "totals-v3.3.1-possession-ridge-shot-schedule-reset"
    assert pd.notna(prediction["home_shot_profile_adjustment"])
    assert pd.notna(prediction["away_shot_profile_adjustment"])
    assert set(prediction["shot_profile_betas"]) == {
        "three_pa_rate",
        "free_throw_rate",
        "opponent_three_pa_rate",
        "opponent_free_throw_rate",
    }
    assert 80 <= prediction["expected_possessions"] <= 120
    assert prediction["projected_total"] > 0
    assert abs(
        prediction["projected_home_points"]
        + prediction["projected_away_points"]
        - prediction["projected_total"]
    ) < 0.02
    assert prediction["home_current_season_games"] > 0


def test_neutral_site_prediction_does_not_apply_home_advantage():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=18, season_id="22021"
    )
    logs = prepare_team_game_logs(
        pd.concat([prior_season, current_season], ignore_index=True)
    )

    normal = predict_game_total_v3(logs, "Team A", "Team B", "2021-12-15")
    neutral = predict_game_total_v3(
        logs, "Team A", "Team B", "2021-12-15", neutral_site=True
    )

    assert neutral["home_court_adjustment"] == 0.0
    assert normal["expected_possessions"] == neutral["expected_possessions"]


def test_walk_forward_backtest_produces_chronological_records():
    raw = make_raw_games(
        start_date="2020-10-20", count=38, season_id="22020"
    )
    results, summary = run_walk_forward_backtest(
        raw, max_games=12, min_prior_team_rows=20
    )

    assert len(results) > 0
    assert summary["test_games"] == len(results)
    assert "v2_projected_total" in results.columns
    assert "v3_2_projected_total" in results.columns
    assert "v3_2_mean_absolute_error_same_games" in summary
    assert "home_schedule_adjustment" in results.columns
    assert "v2_mean_absolute_error_same_games" in summary
    assert "pace_error" in results.columns
    assert "efficiency_error" in results.columns
    assert "mean_efficiency_error" in summary
    assert (results["absolute_error"] >= 0).all()
    assert (pd.to_datetime(results["game_date"]) > pd.Timestamp("2020-10-20")).all()


def test_repository_wide_game_csv_schema_builds_both_team_rows_and_possessions():
    raw = pd.DataFrame([{
        "game_id": "wide-1",
        "game_date": "2025-11-01",
        "season_id": "22025",
        "season_type": "Regular Season",
        "team_name_home": "Boston Celtics",
        "team_name_away": "New York Knicks",
        "pts_home": 118,
        "pts_away": 110,
        "fga_home": 88,
        "fta_home": 22,
        "oreb_home": 10,
        "tov_home": 11,
        "fg3a_home": 40,
        "fga_away": 84,
        "fta_away": 18,
        "oreb_away": 8,
        "tov_away": 13,
        "fg3a_away": 34,
    }])

    logs = prepare_team_game_logs(raw)
    home = logs.loc[logs["team_name"] == "Boston Celtics"].iloc[0]
    away = logs.loc[logs["team_name"] == "New York Knicks"].iloc[0]

    assert len(logs) == 2
    assert abs(home["possessions"] - (88 + 0.44 * 22 - 10 + 11)) < 1e-9
    assert abs(away["possessions"] - (84 + 0.44 * 18 - 8 + 13)) < 1e-9
    assert home["opponent_name"] == "New York Knicks"
    assert away["opponent_name"] == "Boston Celtics"
    assert bool(home["is_home"]) is True
    assert bool(away["is_home"]) is False


def test_ridge_defence_rating_rewards_opponents_it_suppresses():
    teams = ["Team A", "Team B", "Team C", "Team D"]
    defensive_effect = {"Team A": -8.0, "Team B": 8.0, "Team C": 0.0, "Team D": 0.0}
    rows = []
    game_number = 0
    for round_number in range(5):
        for home_index in range(len(teams)):
            for away_index in range(home_index + 1, len(teams)):
                home = teams[home_index]
                away = teams[away_index]
                game_id = f"rating-{game_number}"
                date = pd.Timestamp("2021-10-20") + pd.Timedelta(days=2 * game_number)
                rows.extend([
                    {
                        "game_id": game_id,
                        "game_date": date,
                        "season": 2021,
                        "team_name": home,
                        "opponent_name": away,
                        "points": 110 + defensive_effect[away],
                        "possessions": 100.0,
                        "is_home": True,
                    },
                    {
                        "game_id": game_id,
                        "game_date": date,
                        "season": 2021,
                        "team_name": away,
                        "opponent_name": home,
                        "points": 110 + defensive_effect[home],
                        "possessions": 100.0,
                        "is_home": False,
                    },
                ])
                game_number += 1

    ratings = estimate_pregame_ratings(
        pd.DataFrame(rows), "2021-12-30"
    )
    team_ratings = ratings["teams"]

    assert team_ratings["Team A"]["def_rating"] < team_ratings["Team C"]["def_rating"]
    assert team_ratings["Team B"]["def_rating"] > team_ratings["Team C"]["def_rating"]


def test_wide_game_csv_excludes_playoffs_and_preseason():
    base = {
        "season_id": "22025",
        "team_name_home": "Boston Celtics",
        "team_name_away": "New York Knicks",
        "pts_home": 118,
        "pts_away": 110,
        "fga_home": 88,
        "fta_home": 22,
        "oreb_home": 10,
        "tov_home": 11,
        "fga_away": 84,
        "fta_away": 18,
        "oreb_away": 8,
        "tov_away": 13,
    }
    regular = {**base, "game_id": "reg", "game_date": "2025-11-01",
               "season_type": "Regular Season"}
    playoff = {**base, "game_id": "playoff", "game_date": "2026-04-20",
               "season_type": "Playoffs"}

    logs = prepare_team_game_logs(pd.DataFrame([regular, playoff]))

    assert set(logs["game_id"]) == {"reg"}
    assert len(logs) == 2


def test_v3_diagnostics_can_skip_duplicate_v2_benchmark():
    raw = make_raw_games(
        start_date="2020-10-20", count=38, season_id="22020"
    )
    results, summary = run_walk_forward_backtest(
        raw,
        max_games=12,
        min_prior_team_rows=20,
        include_v2_benchmark=False,
    )

    assert summary["v2_benchmark_included"] is False
    assert summary["v2_mean_absolute_error_same_games"] is None
    assert results["v2_projected_total"].isna().all()
    assert "pace_error" in results.columns
    assert "efficiency_error" in results.columns


def test_precomputed_pregame_ratings_match_direct_prediction():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=18, season_id="22021"
    )
    logs = prepare_team_game_logs(
        pd.concat([prior_season, current_season], ignore_index=True)
    )
    target_date = "2021-12-15"
    ratings = estimate_pregame_ratings(logs, target_date)

    direct = predict_game_total_v3(logs, "Team A", "Team B", target_date)
    reused = predict_game_total_v3(
        logs, "Team A", "Team B", target_date, pregame_ratings=ratings
    )

    assert direct["projected_total"] == reused["projected_total"]
    assert direct["expected_possessions"] == reused["expected_possessions"]


def test_league_baseline_blends_from_eight_seasons_to_current_season():
    prior = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current = make_raw_games(
        start_date="2021-10-20", count=20, season_id="22021"
    )
    current["PTS"] = current["PTS"] + 12
    logs = prepare_team_game_logs(
        pd.concat([prior, current], ignore_index=True)
    )

    early = estimate_pregame_ratings(logs, "2021-10-24")
    mature = estimate_pregame_ratings(logs, "2021-12-01")

    assert abs(early["league_season_blend_weight"] - 0.10) < 1e-9
    assert mature["league_season_blend_weight"] == 1.0
    assert abs(
        mature["league_efficiency"] - mature["league_efficiency_current"]
    ) < 1e-9
    assert mature["league_efficiency"] > early["league_efficiency"]



def test_shot_rates_use_attempts_and_correct_opponent_pairing():
    logs = prepare_team_game_logs(make_raw_games(count=1))
    home = logs.loc[logs["team_name"] == "Team A"].iloc[0]
    away = logs.loc[logs["team_name"] == "Team B"].iloc[0]

    assert abs(home["three_pa_rate"] - (35 / 88)) < 1e-9
    assert abs(home["free_throw_rate"] - (21 / 88)) < 1e-9
    assert abs(home["opponent_three_pa_rate"] - (33 / 86)) < 1e-9
    assert abs(home["opponent_free_throw_rate"] - (19 / 86)) < 1e-9

    assert abs(away["three_pa_rate"] - (33 / 86)) < 1e-9
    assert abs(away["free_throw_rate"] - (19 / 86)) < 1e-9
    assert abs(away["opponent_three_pa_rate"] - (35 / 88)) < 1e-9
    assert abs(away["opponent_free_throw_rate"] - (21 / 88)) < 1e-9


def test_missing_shot_profile_data_falls_back_to_neutral_baseline():
    raw = make_raw_games(start_date="2020-10-20", count=24, season_id="22020")
    current = make_raw_games(start_date="2021-10-20", count=18, season_id="22021")
    raw = pd.concat([raw, current], ignore_index=True)
    logs = prepare_team_game_logs(raw)
    logs = logs.drop(columns=[
        "fga", "fta", "fg3a", "three_pa_rate", "free_throw_rate",
        "opponent_three_pa_rate", "opponent_free_throw_rate",
    ])

    prediction = predict_game_total_v3(
        logs, "Team A", "Team B", "2021-12-15"
    )

    assert prediction["home_shot_profile_adjustment"] == 0.0
    assert prediction["away_shot_profile_adjustment"] == 0.0
    assert all(value == 0.0 for value in prediction["shot_profile_betas"].values())


def test_shot_profile_ratings_use_strictly_pregame_rates():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=12, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=20, season_id="22021"
    )
    target_date = pd.Timestamp("2021-11-10")
    full_logs = prepare_team_game_logs(
        pd.concat([prior_season, current_season], ignore_index=True)
    )
    strict_logs = full_logs.loc[full_logs["game_date"] < target_date].copy()

    from_full = estimate_pregame_ratings(full_logs, target_date)
    from_strict = estimate_pregame_ratings(strict_logs, target_date)

    assert from_full["league_shot_profile"] == from_strict["league_shot_profile"]
    assert from_full["shot_betas"] == from_strict["shot_betas"]
    assert from_full["teams"] == from_strict["teams"]


def test_attempt_rates_are_not_shooting_percentages():
    logs = prepare_team_game_logs(make_raw_games(count=1))
    forbidden = {"fg3_pct", "fg_pct", "ft_pct", "three_point_percentage"}
    assert forbidden.isdisjoint(set(logs.columns))


def test_preparing_team_logs_twice_preserves_shot_profile_columns():
    once = prepare_team_game_logs(make_raw_games(count=3))
    twice = prepare_team_game_logs(once)

    for column in (
        "three_pa_rate",
        "free_throw_rate",
        "opponent_three_pa_rate",
        "opponent_free_throw_rate",
    ):
        assert column in twice.columns
        pd.testing.assert_series_equal(
            once[column].reset_index(drop=True),
            twice[column].reset_index(drop=True),
            check_names=False,
        )



def make_compressed_schedule_games():
    rows = []
    for index, game_date in enumerate(("2021-10-01", "2021-10-02", "2021-10-04")):
        rows.extend([
            {
                "GAME_ID": f"schedule-{index}",
                "GAME_DATE": game_date,
                "SEASON_ID": "22021",
                "TEAM_NAME": "Team A",
                "MATCHUP": "Team A vs. Team B",
                "PTS": 110 + index,
                "FGA": 88,
                "FTA": 20,
                "OREB": 10,
                "TOV": 12,
                "FG3A": 35,
            },
            {
                "GAME_ID": f"schedule-{index}",
                "GAME_DATE": game_date,
                "SEASON_ID": "22021",
                "TEAM_NAME": "Team B",
                "MATCHUP": "Team B @ Team A",
                "PTS": 105 + index,
                "FGA": 86,
                "FTA": 18,
                "OREB": 9,
                "TOV": 13,
                "FG3A": 33,
            },
        ])
    return pd.DataFrame(rows)


def test_schedule_features_capture_rest_back_to_back_and_three_in_four():
    logs = prepare_team_game_logs(make_compressed_schedule_games())
    team_a = logs.loc[logs["team_name"] == "Team A"].sort_values("game_date")

    second_game = team_a.iloc[1]
    third_game = team_a.iloc[2]
    assert second_game["rest_days"] == 0
    assert second_game["back_to_back"] == 1
    assert second_game["three_in_four"] == 0
    assert third_game["rest_days"] == 1
    assert third_game["back_to_back"] == 0
    assert third_game["three_in_four"] == 1


def test_target_schedule_state_uses_only_prior_dates():
    logs = prepare_team_game_logs(make_compressed_schedule_games())
    state = calculate_team_schedule_state(logs, "Team A", "2021-10-05")
    assert state == {"rest_days": 0.0, "back_to_back": 1.0, "three_in_four": 1.0}

    target_day_row = make_raw_games(
        start_date="2021-10-05", count=1, season_id="22021"
    )
    with_target = pd.concat([make_compressed_schedule_games(), target_day_row], ignore_index=True)
    with_future = pd.concat([
        with_target,
        make_raw_games(start_date="2021-10-07", count=1, season_id="22021"),
    ], ignore_index=True)
    from_target = calculate_team_schedule_state(
        prepare_team_game_logs(with_target), "Team A", "2021-10-05"
    )
    from_future = calculate_team_schedule_state(
        prepare_team_game_logs(with_future), "Team A", "2021-10-05"
    )
    assert from_target == from_future == state


def test_schedule_feature_preparation_is_idempotent():
    once = prepare_team_game_logs(make_compressed_schedule_games())
    twice = prepare_team_game_logs(once)
    for column in ("rest_days", "back_to_back", "three_in_four"):
        pd.testing.assert_series_equal(
            once[column].reset_index(drop=True),
            twice[column].reset_index(drop=True),
            check_names=False,
        )


def test_schedule_features_reset_at_season_boundary_and_cap_extended_rest():
    prior_season = make_raw_games(
        start_date="2025-04-10", count=1, season_id="22024"
    )
    season_opener = make_raw_games(
        start_date="2025-10-21", count=1, season_id="22025"
    )
    after_long_break = make_raw_games(
        start_date="2025-10-30", count=1, season_id="22025"
    )
    logs = prepare_team_game_logs(
        pd.concat([prior_season, season_opener, after_long_break], ignore_index=True)
    )

    opener_rows = logs.loc[logs["game_date"] == pd.Timestamp("2025-10-21")]
    assert set(opener_rows["rest_days"]) == {2.0}
    assert set(opener_rows["back_to_back"]) == {0.0}
    assert set(opener_rows["three_in_four"]) == {0.0}

    long_break_rows = logs.loc[logs["game_date"] == pd.Timestamp("2025-10-30")]
    assert set(long_break_rows["rest_days"]) == {3.0}
    assert set(long_break_rows["back_to_back"]) == {0.0}
    assert set(long_break_rows["three_in_four"]) == {0.0}

    old_season_only = logs.loc[logs["game_date"] < pd.Timestamp("2025-10-21")]
    opener_state = calculate_team_schedule_state(
        old_season_only, "Team A", "2025-10-21"
    )
    assert opener_state == {
        "rest_days": 2.0, "back_to_back": 0.0, "three_in_four": 0.0
    }

    prior_to_long_break = logs.loc[logs["game_date"] < pd.Timestamp("2025-10-30")]
    long_break_state = calculate_team_schedule_state(
        prior_to_long_break, "Team A", "2025-10-30"
    )
    assert long_break_state == {
        "rest_days": 3.0, "back_to_back": 0.0, "three_in_four": 0.0
    }


def test_v3_2_ablation_removes_only_schedule_adjustment():
    prior_season = make_raw_games(
        start_date="2020-10-20", count=24, season_id="22020"
    )
    current_season = make_raw_games(
        start_date="2021-10-20", count=18, season_id="22021"
    )
    logs = prepare_team_game_logs(
        pd.concat([prior_season, current_season], ignore_index=True)
    )
    v3_3 = predict_game_total_v3(logs, "Team A", "Team B", "2021-12-15")
    v3_2 = predict_game_total_v3(
        logs, "Team A", "Team B", "2021-12-15",
        include_schedule_features=False,
    )

    assert v3_2["model_version"] == "totals-v3.2-possession-ridge-shot-profile"
    assert v3_2["home_schedule_adjustment"] == 0.0
    assert v3_2["away_schedule_adjustment"] == 0.0
    assert v3_3["model_version"] == "totals-v3.3.1-possession-ridge-shot-schedule-reset"
    assert v3_3["expected_possessions"] == v3_2["expected_possessions"]
