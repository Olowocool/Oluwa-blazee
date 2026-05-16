import requests
from bs4 import BeautifulSoup


ESPN_INJURY_URL = "https://www.espn.com/nba/injuries"


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


def fetch_live_injuries():
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        ESPN_INJURY_URL,
        headers=headers,
        timeout=30
    )

    if response.status_code != 200:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    injuries = {}

    current_team = None

    for tag in soup.find_all(["h2", "tr"]):
        if tag.name == "h2":
            current_team = tag.get_text(strip=True)
            injuries[current_team] = []

        elif tag.name == "tr" and current_team:
            cells = tag.find_all("td")

            if len(cells) >= 4:
                player = cells[0].get_text(strip=True)
                status = cells[2].get_text(strip=True)
                comment = cells[3].get_text(strip=True)

                injuries[current_team].append({
                    "player": player,
                    "status": status,
                    "comment": comment,
                    "impact": get_player_impact(player)
                })
    print(injuries)
    return injuries


def calculate_team_live_injury_penalty(team_name):
    injuries = fetch_live_injuries()

    team_injuries = injuries.get(team_name, [])

    penalty = 0

    for injury in team_injuries:
        status = injury["status"].lower()
        impact = injury["impact"]

        if "out" in status:
            penalty += impact

        elif "doubtful" in status:
            penalty += impact * 0.75

        elif "questionable" in status:
            penalty += impact * 0.5

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
