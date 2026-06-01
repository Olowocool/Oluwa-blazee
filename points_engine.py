import pandas as pd


TEAM_POINTS = {
    "Boston Celtics": {
        "points_for": 120,
        "points_allowed": 110
    },
    "Denver Nuggets": {
        "points_for": 116,
        "points_allowed": 111
    },
    "Oklahoma City Thunder": {
        "points_for": 118,
        "points_allowed": 109
    },
    "Minnesota Timberwolves": {
        "points_for": 113,
        "points_allowed": 108
    },
    "Cleveland Cavaliers": {
        "points_for": 115,
        "points_allowed": 109
    },
    "San Antonio Spurs": {
        "points_for": 111,
        "points_allowed": 118
    },
    "Detroit Pistons": {
        "points_for": 109,
        "points_allowed": 119
    },
}


def get_team_points(team_name):
    return TEAM_POINTS.get(
        team_name,
        {
            "points_for": 112,
            "points_allowed": 112
        }
    )


def add_points_features(df):
    df = df.copy()

    for idx, row in df.iterrows():
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")

        home_points = get_team_points(home_team)
        away_points = get_team_points(away_team)

        df.loc[idx, "home_points_avg"] = home_points["points_for"]
        df.loc[idx, "away_points_avg"] = away_points["points_for"]

        df.loc[idx, "home_points_allowed_avg"] = home_points["points_allowed"]
        df.loc[idx, "away_points_allowed_avg"] = away_points["points_allowed"]

        df.loc[idx, "point_diff"] = (
            home_points["points_for"]
            - away_points["points_for"]
        )

        df.loc[idx, "defensive_points_edge"] = (
            away_points["points_allowed"]
            - home_points["points_allowed"]
        )

        df.loc[idx, "total_points_avg"] = (
            home_points["points_for"]
            + away_points["points_for"]
        )

    return df
