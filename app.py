from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random

app = FastAPI(title="NBA Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
    "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
    "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]


@app.get("/")
def home():
    return {
        "message": "NBA Prediction API is running",
        "status": "ok"
    }


@app.get("/version")
def version():
    return {
        "version": "basketball-model-v1",
        "model": "NBA prediction engine",
        "status": "active"
    }


def create_prediction(home_team, away_team):
    home_strength = random.uniform(0.45, 0.75)
    away_strength = random.uniform(0.35, 0.65)

    total = home_strength + away_strength
    home_prob = round((home_strength / total) * 100, 1)
    away_prob = round(100 - home_prob, 1)

    home_score = random.randint(104, 124)
    away_score = random.randint(98, 121)

    total_points = home_score + away_score

    if home_prob >= 60:
        confidence = "High"
    elif home_prob >= 53:
        confidence = "Medium"
    else:
        confidence = "Low"

    recommendation = "BET" if confidence in ["High", "Medium"] else "NO BET"

    if total_points > 224:
        total_pick = "Over 224.5"
    else:
        total_pick = "Under 224.5"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "predicted_total_points": total_points,
        "moneyline_pick": home_team if home_prob > away_prob else away_team,
        "totals_pick": total_pick,
        "confidence": confidence,
        "recommendation": recommendation,
        "value_edge": round(random.uniform(1.5, 8.5), 2)
    }


@app.get("/predict")
def predict(
    home_team: str = Query(...),
    away_team: str = Query(...)
):
    prediction = create_prediction(home_team, away_team)

    return {
        "status": "success",
        "prediction": prediction
    }


@app.get("/daily-predictions")
def daily_predictions(
    date: str = Query(...)
):
    games = []

    shuffled_teams = NBA_TEAMS.copy()
    random.shuffle(shuffled_teams)

    for i in range(0, 10, 2):
        home_team = shuffled_teams[i]
        away_team = shuffled_teams[i + 1]

        prediction = create_prediction(home_team, away_team)
        games.append(prediction)

    return {
        "status": "success",
        "date": date,
        "games": games
    }


@app.get("/performance")
def performance():
    return {
        "status": "success",
        "summary": {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": "0%",
            "roi": "0%",
            "units_won": 0
        },
        "message": "No saved bet picks yet."
    }
