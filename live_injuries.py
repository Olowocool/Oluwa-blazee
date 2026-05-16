import requests

BALLDONTLIE_API_KEY = "56a1d133-c182-45f6-a7d5-50373e959100"

BALLDONTLIE_INJURY_URL = "https://api.balldontlie.io/v1/player_injuries"


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


def get_player_impact(player_name):
    return STAR_PLAYER_IMPACT.get(player_name, 3)


def normalize_status(status):
    if not status:
        return ""

    return str(status).lower()


def fetch_live_injuries():
    headers = {
        "Authorization": BALLDONTLIE_API_KEY
    }

    try:
        response = requests.get(
            BALLDONTLIE_INJURY_URL,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            return {}

        payload = response.json()
        injuries = {}

        for item in payload.get("data", []):
            player = item.get("player", {})
            team = item.get("team", {})

            first_name = player.get("first_name", "")
            last_name = player.get("last_name", "")
            player_name = f"{first_name} {last_name}".strip()

            team_name = team.get("full_name", "")

            status = item.get("status", "")
            description = item.get("description", "")

            if not team_name or not player_name:
                continue

            if team_name not in injuries:
                injuries[team_name] = []

            injuries[team_name].append({
                "player": player_name,
                "status": status,
                "description": description,
                "impact": get_player_impact(player_name)
            })

        return injuries

    except Exception:
        return {}


def calculate_team_live_injury_penalty(team_name):
    injuries = fetch_live_injuries()

    team_injuries = injuries.get(team_name, [])

    penalty = 0

    for injury in team_injuries:
        status = normalize_status(injury["status"])
        impact = injury["impact"]

        if "out" in status:
            penalty += impact

        elif "doubtful" in status:
            penalty += impact * 0.75

        elif "questionable" in status:
            penalty += impact * 0.5

        elif "probable" in status:
            penalty += impact * 0.15

    return round(penalty, 2)


def calculate_live_matchup_injury_adjustment(home_team, away_team):
    home_penalty = calculate_team_live_injury_penalty(home_team)
    away_penalty = calculate_team_live_injury_penalty(away_team)

    injury_diff = away_penalty - home_penalty

    return {
        "home_injury_penalty": home_penalty,
        "away_injury_penalty": away_penalty,
        "injury_diff": injury_diff
    }
