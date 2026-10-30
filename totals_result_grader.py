import math
import pandas as pd
import requests

from totals_clv import calculate_totals_clv
from totals_tracker import load_totals_history, save_totals_history


API_URL = "https://oluwa-blazee-new.onrender.com"


def fetch_final_score(game_date, home_team, away_team):
    attempts = [
        {
            "home_team": home_team,
            "away_team": away_team,
            "best_bet": home_team,
        },
        {
            "home_team": away_team,
            "away_team": home_team,
            "best_bet": away_team,
        },
    ]

    for params in attempts:
        try:
            response = requests.get(
                f"{API_URL}/score_result",
                params={
                    "date": game_date,
                    "home_team": params["home_team"],
                    "away_team": params["away_team"],
                    "best_bet": params["best_bet"],
                },
                timeout=30,
            )
            if response.status_code != 200:
                continue

            data = response.json()
            if data.get("status") != "completed":
                continue

            home_score = float(data.get("home_score", 0))
            away_score = float(data.get("away_score", 0))
            if not (math.isfinite(home_score) and math.isfinite(away_score)):
                continue
            if home_score < 0 or away_score < 0:
                continue

            return {
                "home_score": home_score,
                "away_score": away_score,
                "matched_home_team": params["home_team"],
                "matched_away_team": params["away_team"],
            }
        except Exception:
            continue

    return None


def _finite_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _decimal_odds_for_side(row, side):
    field = "over_odds" if side == "over" else "under_odds"
    return _finite_positive(row.get(field)) if field in row else None


def _settle(actual_total, line, side):
    if math.isclose(actual_total, line, rel_tol=0.0, abs_tol=1e-9):
        return "Push"
    if side == "over":
        return "Win" if actual_total > line else "Loss"
    return "Win" if actual_total < line else "Loss"


def _profit_for_settlement(result, row, side):
    # A push returns the stake regardless of the quoted price.
    if result == "Push":
        return 0.0

    stake = _finite_positive(row.get("stake"))
    odds = _decimal_odds_for_side(row, side)
    if stake is None or odds is None or odds <= 1.0:
        # Grade the outcome, but never invent a fixed payout or a default price.
        return None
    if result == "Win":
        return round(stake * (odds - 1.0), 6)
    return round(-stake, 6)


def _valid_clv(saved_total, closing_total):
    saved = _finite_positive(saved_total)
    closing = _finite_positive(closing_total)
    if saved is None or closing is None:
        return None
    value = calculate_totals_clv(saved, closing)
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(value, 2) if math.isfinite(value) else None


def grade_totals_results():
    # Hydrate the CSV mirror from PostgreSQL when durable storage is configured.
    df = load_totals_history()
    if df.empty:
        return {"status": "error", "message": "No totals picks available"}

    defaults = {
        "saved_total": None,
        "closing_total": None,
        "clv": None,
        "actual_total": None,
        "home_score": None,
        "away_score": None,
        "matched_home_team": None,
        "matched_away_team": None,
        "result": "Pending",
        "profit_loss": None,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            if column == "saved_total" and "sportsbook_total" in df.columns:
                df[column] = df["sportsbook_total"]
            else:
                df[column] = default

    updated_rows = 0
    for idx, row in df.iterrows():
        current_result = str(row.get("result", "Pending")).strip().casefold()
        if current_result in {"win", "loss", "push"}:
            continue

        score_data = fetch_final_score(
            row.get("game_date"),
            row.get("home_team"),
            row.get("away_team"),
        )
        if score_data is None:
            continue

        try:
            home_score = float(score_data["home_score"])
            away_score = float(score_data["away_score"])
            sportsbook_total = float(row.get("sportsbook_total"))
        except (TypeError, ValueError, OverflowError, KeyError):
            continue

        actual_total = home_score + away_score
        if not all(math.isfinite(x) for x in (home_score, away_score, sportsbook_total)):
            continue
        if home_score < 0 or away_score < 0 or sportsbook_total <= 0:
            continue

        recommendation = str(row.get("recommendation", "")).casefold()
        if "over" in recommendation:
            side = "over"
        elif "under" in recommendation:
            side = "under"
        else:
            continue

        result = _settle(actual_total, sportsbook_total, side)
        profit_loss = _profit_for_settlement(result, row, side)

        saved_total = row.get("saved_total")
        if _finite_positive(saved_total) is None:
            saved_total = sportsbook_total

        # Missing closing lines stay missing. Treating entry as close would
        # manufacture zero CLV and overstate market-data coverage.
        closing_total = row.get("closing_total")
        clv = _valid_clv(saved_total, closing_total)

        df.loc[idx, "home_score"] = home_score
        df.loc[idx, "away_score"] = away_score
        df.loc[idx, "matched_home_team"] = score_data["matched_home_team"]
        df.loc[idx, "matched_away_team"] = score_data["matched_away_team"]
        df.loc[idx, "actual_total"] = actual_total
        df.loc[idx, "saved_total"] = saved_total
        df.loc[idx, "closing_total"] = (
            _finite_positive(closing_total) if _finite_positive(closing_total) is not None else None
        )
        df.loc[idx, "clv"] = clv
        df.loc[idx, "result"] = result
        df.loc[idx, "profit_loss"] = profit_loss
        updated_rows += 1

    if updated_rows:
        # Persist settlement updates to the durable ledger before the CSV mirror.
        save_totals_history(df)

    return {
        "status": "success",
        "updated_rows": updated_rows,
        "graded_outcomes_require_captured_entry_prices_for_pnl": True,
    }


if __name__ == "__main__":
    print(grade_totals_results())
