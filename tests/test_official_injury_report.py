import pandas as pd
import pytest

from research.official_injury_report import parse_official_injury_report_text


TEAMS = ["Boston Celtics", "Orlando Magic", "Indiana Pacers", "Milwaukee Bucks", "LA Clippers"]


def test_parse_official_report_rows_and_multiline_reasons():
    text = """Injury Report: 04/25/25 05:30 PM
Page 1 of 2
Game Date Game Time Matchup Team Player Name Current Status Reason
04/25/2025 07:00 (ET) BOS@ORL Boston Celtics Brown, Jaylen Questionable
Injury/Illness - Right Knee; Posterior
Impingement
Holiday, Jrue Questionable
Injury/Illness - Right Hamstring;
Strain
Orlando Magic Suggs, Jalen Out
Injury/Illness - Left Knee; Trochlea
cartilage tear
08:00 (ET) IND@MIL Indiana Pacers Jackson, Isaiah Out
Injury/Illness - Right Achilles Tendon; Tear
Milwaukee Bucks Smith, Tyler Questionable Injury/Illness - Left Ankle; Sprain
08:30 (ET) DEN@LAC Los Angeles Clippers Leonard, Kawhi Out
Injury/Illness - Right Knee; Soreness
"""
    rows, unsubmitted = parse_official_injury_report_text(text, "2025-04-25", TEAMS)
    assert not unsubmitted
    assert len(rows) == 6
    assert rows.iloc[0]["Team"] == "Boston Celtics"
    assert rows.iloc[0]["Player Name"] == "Brown, Jaylen"
    assert rows.iloc[0]["Reason"] == "Injury/Illness - Right Knee; Posterior Impingement"
    assert rows.iloc[2]["Team"] == "Orlando Magic"
    assert rows.iloc[3]["Matchup"] == "IND@MIL"
    assert rows.iloc[4]["Current Status"] == "QUESTIONABLE"
    assert rows.iloc[5]["Team"] == "LA Clippers"


def test_parser_flags_unsubmitted_team_reports():
    text = """Injury Report: 04/25/25 05:30 PM
04/25/2025 07:00 (ET) BOS@ORL Boston Celtics Brown, Jaylen Questionable
Injury/Illness - Right Knee; Posterior Impingement
Orlando Magic NOT YET SUBMITTED
"""
    rows, unsubmitted = parse_official_injury_report_text(text, "2025-04-25", TEAMS)
    assert len(rows) == 1
    assert unsubmitted == ["Orlando Magic"]


def test_parser_filters_other_report_dates():
    text = """Injury Report: 04/25/25 05:30 PM
04/25/2025 07:00 (ET) BOS@ORL Boston Celtics Brown, Jaylen Questionable
Injury/Illness - Right Knee; Posterior Impingement
04/26/2025 01:00 (ET) CLE@MIA Orlando Magic Suggs, Jalen Out
Injury/Illness - Left Knee; Sprain
"""
    rows, unsubmitted = parse_official_injury_report_text(text, "2025-04-25", TEAMS)
    assert not unsubmitted
    assert len(rows) == 1
    assert rows.iloc[0]["Game Date"] == "04/25/2025"
