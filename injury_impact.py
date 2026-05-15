TEAM_INJURY_IMPACT = {
    "Boston Celtics": 0,
    "Los Angeles Lakers": 0,
    "Cleveland Cavaliers": 0,
    "Minnesota Timberwolves": 0,
    "San Antonio Spurs": 0,
    "Detroit Pistons": 0,
    "Oklahoma City Thunder": 0,
    "Denver Nuggets": 0,
}


STAR_PLAYER_IMPACT = {
    "LeBron James": 8,
    "Anthony Davis": 8,
    "Jayson Tatum": 8,
    "Jaylen Brown": 6,
    "Donovan Mitchell": 7,
    "Anthony Edwards": 7,
    "Nikola Jokic": 10,
    "Shai Gilgeous-Alexander": 10,
    "Victor Wembanyama": 9,
}


def get_team_injury_penalty(team_name):
    return TEAM_INJURY_IMPACT.get(team_name, 0)


def get_player_injury_impact(player_name):
    return STAR_PLAYER_IMPACT.get(player_name, 3)


def calculate_matchup_injury_adjustment(home_team, away_team):
    home_penalty = get_team_injury_penalty(home_team)
    away_penalty = get_team_injury_penalty(away_team)

    injury_diff = away_penalty - home_penalty

    return {
        "home_injury_penalty": home_penalty,
        "away_injury_penalty": away_penalty,
        "injury_diff": injury_diff
    }