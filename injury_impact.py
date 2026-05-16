TEAM_INJURIES = {
    "Cleveland Cavaliers": [],
    "Detroit Pistons": [],
    "Minnesota Timberwolves": [
    {
        "player": "Anthony Edwards",
        "status": "Questionable",
        "impact": 7
    }
],
    "San Antonio Spurs": [],
    "Denver Nuggets": [],
    "Oklahoma City Thunder": [],
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
    "Joel Embiid": 9,
    "Luka Doncic": 10,
    "Stephen Curry": 9,
    "Kevin Durant": 8,
    "Giannis Antetokounmpo": 10,
    "Jalen Brunson": 8,
}

TEAM_DEPTH_FACTOR = {
    "Boston Celtics": 0.85,
    "Denver Nuggets": 1.30,
    "Minnesota Timberwolves": 1.10,
    "San Antonio Spurs": 1.05,
    "Cleveland Cavaliers": 1.00,
    "Detroit Pistons": 0.95,
    "Oklahoma City Thunder": 1.15,
    "Los Angeles Lakers": 1.20,
    "Golden State Warriors": 1.15,
    "Philadelphia 76ers": 1.25
}

def status_multiplier(status):
    status = status.lower()

    if status == "out":
        return 1.0

    if status == "doubtful":
        return 0.75

    if status == "questionable":
        return 0.5

    if status == "probable":
        return 0.15

    return 0


def get_player_impact(player_name):
    return STAR_PLAYER_IMPACT.get(player_name, 3)


def calculate_team_injury_penalty(team_name):
    injuries = TEAM_INJURIES.get(team_name, [])

    total_penalty = 0

    for injury in injuries:
        player = injury["player"]
        status = injury["status"]

        impact = injury.get("impact", get_player_impact(player))
        multiplier = status_multiplier(status)

        total_penalty += impact * multiplier

    depth_factor = TEAM_DEPTH_FACTOR.get(team_name, 1.0)

    adjusted_penalty = total_penalty * depth_factor
    
    return round(adjusted_penalty, 2)

def calculate_matchup_injury_adjustment(home_team, away_team):
    home_penalty = calculate_team_injury_penalty(home_team)
    away_penalty = calculate_team_injury_penalty(away_team)

    injury_diff = away_penalty - home_penalty

    return {
        "home_injury_penalty": home_penalty,
        "away_injury_penalty": away_penalty,
        "injury_diff": injury_diff,
        "home_injuries": TEAM_INJURIES.get(home_team, []),
        "away_injuries": TEAM_INJURIES.get(away_team, [])
    }
