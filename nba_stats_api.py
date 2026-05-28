import pandas as pd


def get_dynamic_team_stats():
    url = "https://www.basketball-reference.com/leagues/NBA_2026.html"

    try:
        tables = pd.read_html(url)

        team_table = None

        for table in tables:
            if "Team" in table.columns and "ORtg" in table.columns:
                team_table = table
                break

        if team_table is None:
            return {}

        team_table = team_table[team_table["Team"] != "League Average"]

        stats = {}

        for _, row in team_table.iterrows():
            team = str(row["Team"]).replace("*", "").strip()

            stats[team] = {
                "off_rating": float(row.get("ORtg", 112)),
                "def_rating": float(row.get("DRtg", 112)),
                "pace": float(row.get("Pace", 100)),
                "recent_wins": 5
            }

        return stats

    except Exception:
        return {}


def get_team_stats(team_name):
    dynamic_stats = get_dynamic_team_stats()

    fallback = {
        "off_rating": 112,
        "def_rating": 112,
        "pace": 100,
        "recent_wins": 5,
    }

    return dynamic_stats.get(team_name, fallback)
