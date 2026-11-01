import pandas as pd

from scripts.audit_injury_data import audit_frames, parse_minutes


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
    assert result["join_coverage"]["player_id_matches_boxscores"]["match_rate_pct"] == 100.0
    assert "game_datetime_est" in result["schedule_time_columns_for_cutoff_audit"]
