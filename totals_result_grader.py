import math
import os

import pandas as pd
import requests

from totals_clv import calculate_totals_clv
from totals_tracker import load_totals_history, save_totals_history


API_URL = os.getenv("NBA_API_URL", "https://oluwa-blazee-new.onrender.com").rstrip("/")


def _increment(diagnostics, key):
    if diagnostics is not None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + 1


def _local_score_result(game_date, home_team, away_team, diagnostics=None):
    """Use the research service's in-process backend when the shared API is unavailable."""
    try:
        import app as local_research_backend

        data = local_research_backend.score_result(
            date=game_date,
            home_team=home_team,
            away_team=away_team,
            best_bet="Over",
        )
        _increment(diagnostics, "local_fallback_calls")
        if isinstance(data, dict) and data.get("status") == "completed":
            _increment(diagnostics, "local_fallback_completed")
        return data
    except Exception:
        _increment(diagnostics, "local_fallback_errors")
        return None


def fetch_final_score(game_date, home_team, away_team, diagnostics=None):
    """Fetch a final score; fall back locally if the shared Render API is unavailable."""
    params = {
        "date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "best_bet": "Over",
    }

    data = None
    try:
        response = requests.get(
            f"{API_URL}/score_result",
            params=params,
            timeout=30,
        )
        if response.status_code in {429, 502, 503, 504}:
            _increment(diagnostics, "shared_api_unavailable")
            data = _local_score_result(
                game_date, home_team, away_team, diagnostics
            )
        elif response.status_code != 200:
            _increment(diagnostics, "shared_api_http_errors")
            return None
        else:
            data = response.json()
    except requests.RequestException:
        _increment(diagnostics, "shared_api_network_errors")
        data = _local_score_result(
            game_date, home_team, away_team, diagnostics
        )
    except (ValueError, TypeError):
        _increment(diagnostics, "shared_api_invalid_responses")
        return None

    if not isinstance(data, dict):
        _increment(diagnostics, "score_lookup_errors")
        return None

    status = str(data.get("status", "")).strip().casefold()
    if status == "pending":
        _increment(diagnostics, "pending_rows")
        return None
    if status == "not_found":
        _increment(diagnostics, "not_found_rows")
        return None
    if status != "completed":
        _increment(diagnostics, "score_lookup_errors")
        return None

    try:
        home_score = float(data.get("home_score"))
        away_score = float(data.get("away_score"))
    except (TypeError, ValueError, OverflowError):
        _increment(diagnostics, "score_lookup_errors")
        return None

    if not (
        math.isfinite(home_score)
        and math.isfinite(away_score)
        and home_score >= 0
        and away_score >= 0
    ):
        _increment(diagnostics, "score_lookup_errors")
        return None

    return {
        "home_score": home_score,
        "away_score": away_score,
        "matched_home_team": data.get("home_team") or home_team,
        "matched_away_team": data.get("away_team") or away_team,
    }

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

    diagnostics = {
        "pending_rows": 0,
        "not_found_rows": 0,
        "score_lookup_errors": 0,
        "shared_api_unavailable": 0,
        "shared_api_http_errors": 0,
        "shared_api_network_errors": 0,
        "shared_api_invalid_responses": 0,
        "local_fallback_calls": 0,
        "local_fallback_completed": 0,
        "local_fallback_errors": 0,
    }

    updated_rows = 0
    already_settled_rows = 0
    for idx, row in df.iterrows():
        current_result = str(row.get("result", "Pending")).strip().casefold()
        if current_result in {"win", "loss", "push"}:
            already_settled_rows += 1
            continue

        score_data = fetch_final_score(
            row.get("game_date"),
            row.get("home_team"),
            row.get("away_team"),
            diagnostics=diagnostics,
        )
        if score_data is None:
            continue

        try:
            home_score = float(score_data["home_score"])
            away_score = float(score_data["away_score"])
            sportsbook_total = float(row.get("sportsbook_total"))
        except (TypeError, ValueError, OverflowError, KeyError):
            diagnostics["score_lookup_errors"] += 1
            continue

        actual_total = home_score + away_score
        if not all(math.isfinite(x) for x in (home_score, away_score, sportsbook_total)):
            diagnostics["score_lookup_errors"] += 1
            continue
        if home_score < 0 or away_score < 0 or sportsbook_total <= 0:
            diagnostics["score_lookup_errors"] += 1
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
        "already_settled_rows": already_settled_rows,
        "graded_outcomes_require_captured_entry_prices_for_pnl": True,
        **diagnostics,
        "message": (
            f"Graded {updated_rows} completed totals pick(s)."
            if updated_rows
            else (
                "No completed totals results were available. "
                f"Still pending: {diagnostics['pending_rows']}; "
                f"not found: {diagnostics['not_found_rows']}; "
                f"lookup errors: {diagnostics['score_lookup_errors']}; "
                f"local fallback results: {diagnostics['local_fallback_completed']}."
            )
        ),
    }

if __name__ == "__main__":
    print(grade_totals_results())
