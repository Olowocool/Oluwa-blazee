import pandas as pd


def calculate_team_pace(history_df, team_name, last_n=20):
    """
    Estimate team pace from recent total scores.
    Uses real historical NBA scores.
    """

    if history_df.empty:
        return 100.0

    team_games = history_df[
        (history_df["home_team"] == team_name)
        |
        (history_df["away_team"] == team_name)
    ]

    if len(team_games) == 0:
        return 100.0

    recent = team_games.tail(last_n)

    avg_total = recent["total_score"].mean()

    league_average = 225.0

    pace = 100 + (
        (avg_total - league_average)
        / league_average
        * 20
    )

    return round(float(pace), 2)


def calculate_matchup_pace(
    history_df,
    home_team,
    away_team
):

    home_pace = calculate_team_pace(
        history_df,
        home_team
    )

    away_pace = calculate_team_pace(
        history_df,
        away_team
    )

    combined_pace = (
        home_pace + away_pace
    ) / 2

    pace_adjustment = (
        combined_pace - 100
    ) * 0.5

    return {
        "home_pace": round(home_pace, 2),
        "away_pace": round(away_pace, 2),
        "combined_pace": round(combined_pace, 2),
        "pace_adjustment": round(pace_adjustment, 2)
    }
