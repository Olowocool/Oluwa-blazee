import pandas as pd

from research.objective3_readiness import assess_readiness


def test_readiness_requires_150_development_games_and_a_reserved_holdout():
    rows = []
    scores = []
    for day in range(1, 201):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day - 1)
        key = f"game-{day}"
        rows.append({
            "game_date": date,
            "game_key": key,
            "home_team": "Washington Wizards",
            "away_team": "Charlotte Hornets",
        })
        scores.append({
            "_game_key": key,
            "home_team": "Washington Wizards",
            "away_team": "Charlotte Hornets",
            "actual_total": 220.0 + (day % 10),
        })
    result = assess_readiness(pd.DataFrame(rows), pd.DataFrame(scores))
    assert result["ready"] is True
    assert result["development_games_before_frozen_holdout"] >= 150
    assert result["reserved_holdout_games"] > 0
    assert result["status"] == "READY_FOR_FULL_OBJECTIVE3_EVALUATION"
    assert result["prices_or_game_level_rows_emitted"] is False
    assert result["files_written"] is False


def test_readiness_does_not_count_missing_or_wrong_orientation_scores():
    selected = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-01-01"), "game_key": "good",
         "home_team": "Washington Wizards", "away_team": "Charlotte Hornets"},
        {"game_date": pd.Timestamp("2026-01-02"), "game_key": "wrong",
         "home_team": "Washington Wizards", "away_team": "Charlotte Hornets"},
        {"game_date": pd.Timestamp("2026-01-03"), "game_key": "missing",
         "home_team": "Washington Wizards", "away_team": "Charlotte Hornets"},
    ])
    scores = pd.DataFrame([
        {"_game_key": "good", "home_team": "Washington Wizards",
         "away_team": "Charlotte Hornets", "actual_total": 220.0},
        {"_game_key": "wrong", "home_team": "Charlotte Hornets",
         "away_team": "Washington Wizards", "actual_total": 221.0},
    ])
    result = assess_readiness(selected, scores)
    assert result["settled_score_matched_games"] == 1
    assert result["games_missing_final_score"] == 1
    assert result["home_away_orientation_mismatches"] == 1
    assert result["ready"] is False
