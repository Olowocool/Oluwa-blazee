import pandas as pd

from scripts.audit_injury_data import (
    audit_frames,
    build_pre_tip_snapshot,
    parse_minutes,
)


def test_parse_minutes_handles_mmss_and_decimal():
    assert parse_minutes("31:30") == 31.5
    assert parse_minutes("12.5") == 12.5
    assert parse_minutes("DNP - Coach's Decision") is None


def test_audit_frames_profiles_ids_and_statuses():
    injury = pd.DataFrame([
        {
            "season": 2025, "report_datetime": "2024-10-22 17:00:00",
            "game_date": "2024-10-22", "game_time": "07:30 (ET)",
            "report_matchup": "BOS@NYK", "away_team_abbr": "BOS",
            "home_team_abbr": "NYK", "team": "Boston Celtics",
            "report_player_name": "Player, Test", "current_status": "Out",
            "reason": "Injury/Illness - knee", "category": None,
            "previous_status": None, "game_id": "0022400001",
            "team_id": 1610612738, "person_id": 123,
            "match_method": "exact",
        },
        {
            "season": 2025, "report_datetime": "2024-10-22 18:00:00",
            "game_date": "2024-10-22", "game_time": "07:30 (ET)",
            "report_matchup": "BOS@NYK", "away_team_abbr": "BOS",
            "home_team_abbr": "NYK", "team": "Boston Celtics",
            "report_player_name": "Player, Test", "current_status": "Out",
            "reason": "Injury/Illness - knee", "category": None,
            "previous_status": None, "game_id": "0022400001",
            "team_id": 1610612738, "person_id": 123,
            "match_method": "exact",
        },
    ])
    box = pd.DataFrame([{
        "game_id": "0022400001", "person_id": 123,
        "minutes": "0:00", "points": 0,
    }])
    schedule = pd.DataFrame([{
        "game_id": "0022400001", "game_date": "2024-10-22",
        "game_datetime_est": "2024-10-22T19:30:00",
    }])
    roster = pd.DataFrame([{
        "game_id": "0022400001", "player_id": 123,
    }])
    result = audit_frames(injury, box, schedule, roster)
    assert result["totals"]["injury_report_rows_including_placeholders"] == 2
    assert result["totals"]["player_listing_rows"] == 2
    assert result["totals"]["unique_games_in_reports"] == 1
    assert result["join_coverage"]["player_game_report_rows_with_boxscore_record"]["match_rate_pct"] == 100.0
    assert result["join_coverage"]["report_game_id_schedule_coverage_pct"] == 100.0



def test_pre_tip_snapshot_uses_latest_report_at_or_before_60_minute_cutoff():
    base = {
        "season": 2025,
        "game_date": "2024-10-22",
        "game_time": "07:30 (ET)",
        "report_matchup": "BOS@NYK",
        "away_team_abbr": "BOS",
        "home_team_abbr": "NYK",
        "team": "Boston Celtics",
        "report_player_name": "Player, Test",
        "reason": "Injury/Illness - knee",
        "game_id": "0022400001",
        "team_id": 1610612738,
        "person_id": 123,
        "match_method": "exact",
        "is_submission_placeholder": False,
    }
    injury = pd.DataFrame([
        {**base, "report_datetime": "2024-10-22 17:00:00", "current_status": "Questionable"},
        {**base, "report_datetime": "2024-10-22 18:15:00", "current_status": "Doubtful"},
        # This later update is only 45 minutes before tip and must not be used.
        {**base, "report_datetime": "2024-10-22 18:45:00", "current_status": "Out"},
    ])
    schedule = pd.DataFrame([{
        "id": 100,
        "game_date": "2024-10-22",
        "game_date_time": "2024-10-22T19:30:00-04:00",
        # Deliberately use different abbreviations: crosswalk should still join.
        "away_abbreviation": "GS",
        "home_abbreviation": "NY",
        "time_valid": True,
    }])
    crosswalk = pd.DataFrame([{
        "nba_game_id": "0022400001",
        "espn_game_id": 100,
    }])

    selected, metrics = build_pre_tip_snapshot(
        injury, schedule, cutoff_minutes=60, crosswalk=crosswalk
    )

    assert len(selected) == 1
    assert selected.iloc[0]["current_status"] == "Doubtful"
    assert str(selected.iloc[0]["selected_report_datetime"]) == "2024-10-22 18:15:00"
    assert metrics["tipoff_match_coverage_pct"] == 100.0
    assert metrics["report_games_matched_via_nba_espn_crosswalk"] == 1
    assert metrics["report_game_clock_matches_espn_pct"] == 100.0
    assert metrics["pre_tip_snapshot_coverage_pct"] == 100.0


def test_player_game_keys_are_joined_as_pairs_not_independent_non_null_lists():
    injury = pd.DataFrame([
        {
            "season": 2025, "report_datetime": "2024-10-22 17:00:00",
            "game_date": "2024-10-22", "game_time": "07:30 (ET)",
            "report_matchup": "BOS@NYK", "away_team_abbr": "BOS",
            "home_team_abbr": "NYK", "team": "Boston Celtics",
            "report_player_name": "Player, One", "current_status": "Out",
            "reason": "Injury/Illness - knee", "category": None,
            "previous_status": None, "game_id": "G1",
            "team_id": 1610612738, "person_id": 123, "match_method": "exact",
        },
        {
            "season": 2025, "report_datetime": "2024-10-22 17:00:00",
            "game_date": "2024-10-22", "game_time": "07:30 (ET)",
            "report_matchup": "BOS@NYK", "away_team_abbr": "BOS",
            "home_team_abbr": "NYK", "team": "Boston Celtics",
            "report_player_name": "Player, Two", "current_status": "Out",
            "reason": "Injury/Illness - foot", "category": None,
            "previous_status": None, "game_id": "G2",
            "team_id": 1610612738, "person_id": 456, "match_method": "exact",
        },
    ])
    box = pd.DataFrame([
        {"game_id": "G1", "person_id": 123, "minutes": "0:00", "points": 0},
        # These two partial-null rows must not fabricate the key (G2, 456).
        {"game_id": "G2", "person_id": None, "minutes": None, "points": None},
        {"game_id": None, "person_id": 456, "minutes": None, "points": None},
    ])
    schedule = pd.DataFrame([
        {"game_id": "G1"}, {"game_id": "G2"},
    ])
    roster = pd.DataFrame([
        {"game_id": "G1", "player_id": 123},
        {"game_id": "G2", "player_id": None},
        {"game_id": None, "player_id": 456},
    ])

    result = audit_frames(injury, box, schedule, roster)

    assert result["join_coverage"]["player_game_report_rows_with_boxscore_record"]["eligible_rows"] == 2
    assert result["join_coverage"]["player_game_report_rows_with_boxscore_record"]["matched_rows"] == 1
    assert result["join_coverage"]["player_game_report_rows_with_boxscore_record"]["match_rate_pct"] == 50.0
