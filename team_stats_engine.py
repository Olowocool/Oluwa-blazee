TEAM_STATS = {
    "Boston Celtics": {
        "off_rating": 120.5,
        "def_rating": 111.2,
        "pace": 99.5,
        "recent_wins": 8,
    },
    "Denver Nuggets": {
        "off_rating": 118.1,
        "def_rating": 112.4,
        "pace": 97.2,
        "recent_wins": 7,
    },
    "Oklahoma City Thunder": {
        "off_rating": 119.4,
        "def_rating": 110.8,
        "pace": 101.4,
        "recent_wins": 9,
    },
    "Minnesota Timberwolves": {
        "off_rating": 114.3,
        "def_rating": 108.1,
        "pace": 96.8,
        "recent_wins": 6,
    },
    "Cleveland Cavaliers": {
        "off_rating": 116.2,
        "def_rating": 109.9,
        "pace": 98.4,
        "recent_wins": 6,
    },
    "San Antonio Spurs": {
        "off_rating": 111.5,
        "def_rating": 117.8,
        "pace": 100.2,
        "recent_wins": 3,
    },
    "Detroit Pistons": {
        "off_rating": 109.8,
        "def_rating": 118.9,
        "pace": 99.1,
        "recent_wins": 2,
    }
}


def get_team_stats(team_name):
    return TEAM_STATS.get(
        team_name,
        {
            "off_rating": 112,
            "def_rating": 112,
            "pace": 100,
            "recent_wins": 5,
        }
    )
