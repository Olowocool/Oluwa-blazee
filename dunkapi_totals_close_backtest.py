"""Chronological NBA totals backtest against genuine historical closing lines.

Data provider: DunkAPI's archived get_odds(day) response. The response contains
opening and closing total lines, closing over/under prices, and final scores.
It does NOT provide historical prices at a separately timestamped entry point
in this endpoint, so this script evaluates the model at the closing line and
reports opening-to-closing line movement separately. It never claims
bet-specific CLV without a recorded bet-entry price and closing reference.

No API call is made unless the user confirms the estimated credit budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from totals_model import classify_total_edge, predict_game_total
from totals_betting_backtest import game_key, prepare_score_history, settle_total_bet

API_URL = "https://mcp.dunkapi.com/v1/get_odds"
DEFAULT_SCORE_FILE = "data/historical_nba_scores.csv"
DEFAULT_MARKET_FILE = "data/dunkapi_historical_totals.csv"
DEFAULT_OUTPUT_DIR = "outputs"
CREDITS_PER_DAY_REQUEST = 10
EASTERN = ZoneInfo("America/New_York")

MARKET_FIELDS = [
    "game_id",
    "game_date",
    "commence_time_utc",
    "home_team",
    "away_team",
    "bookmaker",
    "total_open",
    "total_close",
    "over_odds_american",
    "under_odds_american",
    "over_odds_decimal",
    "under_odds_decimal",
    "actual_home_points",
    "actual_away_points",
    "actual_total",
    "source_captured_at",
    "market_source",
]


def parse_date_range(start_date: str, end_date: str) -> list[date]:
    try:
        first = date.fromisoformat(start_date)
        last = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format.") from exc
    if last < first:
        raise ValueError("--end-date must be on or after --start-date.")
    days = (last - first).days + 1
    return [first + timedelta(days=i) for i in range(days)]


def estimate_api_credits(day_count: int) -> int:
    if day_count < 0:
        raise ValueError("day_count cannot be negative.")
    return day_count * CREDITS_PER_DAY_REQUEST


def american_to_decimal(value: Any) -> float:
    """Convert an American price (e.g. -110 or +125) to decimal odds."""
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid American odds: {value!r}") from exc
    if not np.isfinite(price) or price == 0:
        raise ValueError(f"Invalid American odds: {value!r}")
    return round(1.0 + (100.0 / abs(price) if price < 0 else price / 100.0), 6)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _parse_start_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Historical game is missing starts_at.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid starts_at: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("starts_at must include a timezone.")
    return parsed.astimezone(timezone.utc)


def parse_daily_payload(
    payload: dict[str, Any],
    requested_date: date,
    bookmaker: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse finished games for one named sportsbook; never synthesize prices."""
    if not isinstance(payload, dict):
        raise ValueError("DunkAPI response must be a JSON object.")

    # DunkAPI's REST reference shows a root-level "games" array, while its
    # Python backtesting guide unwraps resp.json()["data"] first. Support both
    # documented response envelopes.
    if not isinstance(payload.get("games"), list):
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("games"), list):
            payload = data

    if not isinstance(payload.get("games"), list):
        keys = ", ".join(sorted(str(key) for key in payload.keys())) or "(none)"
        provider_message = payload.get("message") or payload.get("error") or payload.get("detail")
        detail = ""
        if provider_message is not None:
            message = str(provider_message).replace("\\n", " ")[:240]
            detail = f" Provider message: {message}"
        raise ValueError(
            "DunkAPI response must contain a 'games' array at the root or under 'data' "
            f"(top-level keys: {keys}).{detail}"
        )

    wanted = bookmaker.strip().casefold()
    if not wanted:
        raise ValueError("A single --bookmaker is required; do not combine books silently.")

    records: list[dict[str, Any]] = []
    available_books: set[str] = set()
    games_seen = 0
    finished_games_seen = 0
    target_date_games = 0
    missing_book_rows = 0
    missing_market_rows = 0
    invalid_rows = 0
    mismatched_dates = 0

    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        games_seen += 1
        game_id = str(game.get("game_id") or "").strip()
        if not game_id:
            invalid_rows += 1
            continue

        game_date_value = game.get("date")
        try:
            game_day = date.fromisoformat(str(game_date_value)[:10])
        except (ValueError, TypeError):
            try:
                game_day = _parse_start_time(game.get("starts_at")).astimezone(EASTERN).date()
            except ValueError:
                invalid_rows += 1
                continue

        if game_day != requested_date:
            mismatched_dates += 1
            continue
        target_date_games += 1

        status = str(game.get("status") or "").strip().casefold()
        home = game.get("home") if isinstance(game.get("home"), dict) else {}
        away = game.get("away") if isinstance(game.get("away"), dict) else {}
        home_score = _number(home.get("score"))
        away_score = _number(away.get("score"))
        is_finished = status in {"final", "archived", "finished", "complete", "completed"}
        if not is_finished or home_score is None or away_score is None:
            continue
        finished_games_seen += 1

        home_team = str(home.get("name") or "").strip()
        away_team = str(away.get("name") or "").strip()
        if not home_team or not away_team or home_team.casefold() == away_team.casefold():
            invalid_rows += 1
            continue

        books = game.get("books") if isinstance(game.get("books"), list) else []
        matches = []
        for book in books:
            if not isinstance(book, dict):
                continue
            provider = str(book.get("provider") or "").strip()
            if provider:
                available_books.add(provider)
            if provider.casefold() == wanted:
                matches.append(book)

        if len(matches) > 1:
            raise ValueError(
                f"Game {game_id} has duplicate rows for bookmaker {bookmaker!r}; "
                "resolve provider duplicates before betting evaluation."
            )
        if not matches:
            missing_book_rows += 1
            continue

        book = matches[0]
        total_close = _number(book.get("total_close"))
        over_american = _number(book.get("over_odds"))
        under_american = _number(book.get("under_odds"))
        total_open = _number(book.get("total_open"))
        if (
            total_close is None
            or total_close <= 0
            or over_american is None
            or under_american is None
            or over_american == 0
            or under_american == 0
            or (total_open is not None and total_open <= 0)
        ):
            missing_market_rows += 1
            continue

        try:
            commence = _parse_start_time(game.get("starts_at"))
            over_decimal = american_to_decimal(over_american)
            under_decimal = american_to_decimal(under_american)
        except ValueError:
            invalid_rows += 1
            continue

        records.append({
            "game_id": game_id,
            "game_date": game_day.isoformat(),
            "commence_time_utc": commence.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": str(book.get("provider") or bookmaker).strip(),
            "total_open": float(total_open) if total_open is not None else np.nan,
            "total_close": float(total_close),
            "over_odds_american": float(over_american),
            "under_odds_american": float(under_american),
            "over_odds_decimal": over_decimal,
            "under_odds_decimal": under_decimal,
            "actual_home_points": float(home_score),
            "actual_away_points": float(away_score),
            "actual_total": float(home_score + away_score),
            # This is provider collection metadata, not a claimed historical entry timestamp.
            "source_captured_at": str(book.get("captured_at") or ""),
            "market_source": "DunkAPI get_odds historical opening/closing archive",
        })

    return records, {
        "requested_date": requested_date.isoformat(),
        "games_seen": games_seen,
        "games_for_requested_date": target_date_games,
        "finished_games_seen": finished_games_seen,
        "rows_for_selected_bookmaker": len(records),
        "games_without_selected_bookmaker": missing_book_rows,
        "games_missing_close_line_or_prices": missing_market_rows,
        "invalid_rows": invalid_rows,
        "rows_for_other_dates_excluded": mismatched_dates,
        "available_bookmakers": sorted(available_books),
    }


def _read_market_archive(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        missing = set(MARKET_FIELDS).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Existing market archive missing columns: {sorted(missing)}")
        return list(reader)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().casefold() in {"", "nan", "none", "null"}:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def merge_market_rows(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Idempotent merge; fail on conflicts for the same game and bookmaker."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in [*existing, *new_rows]:
        key = (str(row.get("game_id") or ""), str(row.get("bookmaker") or "").casefold())
        if not all(key):
            raise ValueError("Every historical market row needs game_id and bookmaker.")
        if key not in merged:
            merged[key] = row
            continue
        old = merged[key]
        for field in [
            "game_date", "home_team", "away_team", "total_open", "total_close",
            "over_odds_american", "under_odds_american", "actual_home_points", "actual_away_points",
        ]:
            a, b = old.get(field), row.get(field)
            if _is_missing(a) or _is_missing(b):
                continue
            try:
                equal = float(a) == float(b)
            except (ValueError, TypeError):
                equal = str(a).strip().casefold() == str(b).strip().casefold()
            if not equal:
                raise ValueError(
                    f"Conflicting historical market data for game {key[0]}, bookmaker {key[1]}, field {field}."
                )
        for field, value in row.items():
            if _is_missing(old.get(field)) and not _is_missing(value):
                old[field] = value

    return sorted(merged.values(), key=lambda r: (str(r.get("game_date", "")), str(r.get("game_id", ""))))


def write_market_archive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=MARKET_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MARKET_FIELDS})
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_api_key() -> str:
    return os.getenv("DUNKAPI_API_KEY", "").strip()


def fetch_daily_odds(
    api_key: str,
    target_date: date,
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, str]]:
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "OluwaBlazeeChronologicalTotalsBacktest/1.0",
    }
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"date": target_date.isoformat()},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"DunkAPI connection failed ({type(exc).__name__}).") from None

    if response.status_code != 200:
        # Never log authentication headers or keys. Surface only a short provider
        # error message so date/parameter errors can be diagnosed without retries.
        if response.status_code in (401, 403):
            raise RuntimeError("DunkAPI rejected the key or account access (HTTP authentication/plan error).")
        if response.status_code == 429:
            raise RuntimeError("DunkAPI request limit or credits exhausted (HTTP 429).")
        if response.status_code == 400:
            message = ""
            try:
                error_payload = response.json()
                if isinstance(error_payload, dict):
                    candidate = (
                        error_payload.get("detail")
                        or error_payload.get("message")
                        or error_payload.get("error")
                    )
                    if isinstance(candidate, (str, int, float)):
                        message = str(candidate)
                    elif candidate is not None:
                        message = json.dumps(candidate, ensure_ascii=True)
            except (ValueError, TypeError):
                message = ""
            if not message:
                message = str(getattr(response, "text", "") or "")
            message = " ".join(message.split())[:300]
            suffix = f" Provider says: {message}" if message else " No provider message was included."
            raise RuntimeError(
                f"DunkAPI rejected requested date {target_date.isoformat()} (HTTP 400).{suffix}"
            )
        raise RuntimeError(f"DunkAPI returned HTTP {response.status_code}.")

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("DunkAPI response was not valid JSON.") from None
    if not isinstance(payload, dict):
        raise RuntimeError("DunkAPI response was not a JSON object.")

    usage = {
        "credits_used": str(response.headers.get("X-Credits-Used", "")),
        "credits_remaining": str(response.headers.get("X-Credits-Remaining", "")),
    }
    return payload, usage


def collect_market_data(args: argparse.Namespace) -> dict[str, Any]:
    dates = parse_date_range(args.start_date, args.end_date)
    estimated_credits = estimate_api_credits(len(dates))
    print(f"Provider: DunkAPI historical closing lines")
    print(f"Dates requested: {len(dates)} ({args.start_date} through {args.end_date})")
    print(f"Estimated cost: {estimated_credits} credits (10 per daily request; verify current provider quota).")
    print(f"Bookmaker: {args.bookmaker}")
    print("Prices used for ROI are the provider's closing-line prices; this endpoint does not supply an entry-time price timestamp.")
    if args.plan_only:
        return {"status": "plan_only", "daily_requests": len(dates), "estimated_credits": estimated_credits}
    if not args.confirm_api_cost:
        raise ValueError("No API calls made. Review the estimate, then pass --confirm-api-cost.")
    if args.max_api_credits is None or args.max_api_credits < 0:
        raise ValueError("A non-negative --max-api-credits cap is required before collection.")
    if estimated_credits > args.max_api_credits:
        raise ValueError(
            f"Estimated cost {estimated_credits} exceeds --max-api-credits {args.max_api_credits}; no API calls were made."
        )
    api_key = load_api_key()
    if not api_key:
        raise ValueError("DUNKAPI_API_KEY is not set in the environment. No API calls were made.")

    archive_path = Path(args.output_file)
    archive = _read_market_archive(archive_path)
    diagnostics: dict[str, Any] = {
        "provider": "DunkAPI",
        "requested_dates": len(dates),
        "daily_requests_attempted": 0,
        "credits_estimated": estimated_credits,
        "credits_used_header_last_response": "",
        "credits_remaining_header_last_response": "",
        "rows_parsed": 0,
        "bookmaker": args.bookmaker,
        "dates": {},
        "output_file": str(archive_path),
        "api_key_logged": False,
    }

    for target_date in dates:
        payload, usage = fetch_daily_odds(api_key, target_date)
        rows, day_diagnostics = parse_daily_payload(payload, target_date, args.bookmaker)
        diagnostics["daily_requests_attempted"] += 1
        diagnostics["credits_used_header_last_response"] = usage["credits_used"]
        diagnostics["credits_remaining_header_last_response"] = usage["credits_remaining"]
        diagnostics["rows_parsed"] += len(rows)
        diagnostics["dates"][target_date.isoformat()] = day_diagnostics
        if rows:
            archive = merge_market_rows(archive, rows)
            write_market_archive(archive_path, archive)
        print(
            f"{target_date.isoformat()}: {len(rows)} rows for {args.bookmaker}; "
            f"credits remaining header={usage['credits_remaining'] or 'not provided'}"
        )

    if not diagnostics["rows_parsed"]:
        books = sorted({
            book for value in diagnostics["dates"].values()
            for book in value.get("available_bookmakers", [])
        })
        raise RuntimeError(
            f"No historical rows were found for bookmaker {args.bookmaker!r}. "
            f"Bookmakers observed: {books or 'none'}. No new records were added."
        )

    diagnostics["archive_rows_for_selected_bookmaker"] = len(archive)
    diagnostics["status"] = "success"
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def _reserved_holdout_start(games: pd.DataFrame, holdout_games: int) -> pd.Timestamp:
    if holdout_games < 1:
        raise ValueError("holdout_games must be at least 1.")
    ordered = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No historical closing lines are available.")
    index = max(0, len(ordered) - int(holdout_games))
    # Keep all games on the same calendar date in the same split.
    return pd.Timestamp(ordered.iloc[index]["game_date"]).normalize()


def _score_split(results: pd.DataFrame, split: str, strategy: str) -> dict[str, Any]:
    subset = results.copy() if split == "all_games" else results.loc[
        results["evaluation_split"] == split
    ].copy()
    if strategy == "all_qualifying_edges":
        bets = subset.loc[subset["bet_side"].isin(["Over", "Under"])]
    elif strategy == "strong_only":
        bets = subset.loc[subset["recommendation"].astype(str).str.startswith("Strong")]
    elif strategy == "lean_only":
        bets = subset.loc[subset["recommendation"].astype(str).str.startswith("Lean")]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    outcomes = bets["bet_outcome"].astype(str) if not bets.empty else pd.Series(dtype=str)
    wins = int((outcomes == "Win").sum())
    losses = int((outcomes == "Loss").sum())
    pushes = int((outcomes == "Push").sum())
    resolved = wins + losses
    placed = wins + losses + pushes
    stake = float(pd.to_numeric(bets.get("stake_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(bets.get("profit_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    odds = pd.to_numeric(bets.get("selected_decimal_odds", pd.Series(dtype=float)), errors="coerce").dropna()

    model_errors = pd.to_numeric(subset.get("model_error", pd.Series(dtype=float)), errors="coerce").dropna()
    market_errors = pd.to_numeric(subset.get("market_close_error", pd.Series(dtype=float)), errors="coerce").dropna()
    moves = pd.to_numeric(subset.get("market_line_move_points", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "split": split,
        "strategy": strategy,
        "games_scored": int(len(subset)),
        "model_mae": float(model_errors.abs().mean()) if len(model_errors) else np.nan,
        "market_close_mae": float(market_errors.abs().mean()) if len(market_errors) else np.nan,
        "model_minus_market_close_mae": (
            float(model_errors.abs().mean() - market_errors.abs().mean())
            if len(model_errors) and len(market_errors) else np.nan
        ),
        "model_bias": float(model_errors.mean()) if len(model_errors) else np.nan,
        "mean_open_to_close_total_move": float(moves.mean()) if len(moves) else np.nan,
        "bets": placed,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_pct_excluding_pushes": 100 * wins / resolved if resolved else np.nan,
        "average_decimal_odds": float(odds.mean()) if len(odds) else np.nan,
        "stake_units_risked": stake,
        "net_profit_units": profit,
        "roi_pct": 100 * profit / stake if stake else np.nan,
        "bet_level_clv_available": False,
        "interpretation": (
            "ROI is simulated using the provider's closing total and closing Over/Under price. "
            "Open-to-close line movement is descriptive; no entry-timestamp CLV is claimed."
        ),
    }


def run_closing_line_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_path = Path(args.market_file)
    score_path = Path(args.scores_file)
    if not market_path.is_file():
        raise FileNotFoundError(
            f"Market archive not found: {market_path}. Run collection with DUNKAPI_API_KEY first."
        )
    if not score_path.is_file():
        raise FileNotFoundError(f"Score history not found: {score_path}.")

    market = pd.read_csv(market_path)
    required_market = [
        "game_id", "game_date", "home_team", "away_team", "bookmaker",
        "total_open", "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_home_points", "actual_away_points", "actual_total",
    ]
    missing = [c for c in required_market if c not in market.columns]
    if missing:
        raise ValueError(f"Market archive is missing columns: {missing}")
    market["game_date"] = pd.to_datetime(market["game_date"], errors="coerce").dt.normalize()
    for col in [
        "total_open", "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_home_points", "actual_away_points", "actual_total",
    ]:
        market[col] = pd.to_numeric(market[col], errors="coerce")

    market = market.dropna(subset=[
        "game_id", "game_date", "home_team", "away_team", "bookmaker",
        "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_home_points", "actual_away_points", "actual_total",
    ]).copy()
    market = market[
        (market["total_close"] > 0)
        & (market["total_open"].isna() | (market["total_open"] > 0))
        & (market["over_odds_decimal"] > 1)
        & (market["under_odds_decimal"] > 1)
        & (market["actual_home_points"] > 0)
        & (market["actual_away_points"] > 0)
    ].copy()
    market["game_key"] = [
        game_key(d, h, a)
        for d, h, a in zip(market["game_date"], market["home_team"], market["away_team"])
    ]
    if market.duplicated(["game_key", "bookmaker"]).any():
        raise ValueError("Market archive contains duplicate game/bookmaker rows.")

    history = prepare_score_history(pd.read_csv(score_path))
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
    score_lookup = {row["_game_key"]: row for _, row in history.iterrows()}

    records: list[dict[str, Any]] = []
    unmatched_scores = 0
    mismatched_scores = 0
    for _, game in market.sort_values(["game_date", "game_id"]).iterrows():
        target_date = pd.Timestamp(game["game_date"]).normalize()
        score = score_lookup.get(game["game_key"])
        base = game.to_dict()
        base["prediction_status"] = "scored"
        base["skip_reason"] = ""

        if score is None:
            base["prediction_status"] = "skipped"
            base["skip_reason"] = "No matching final score in local historical history."
            unmatched_scores += 1
            records.append(base)
            continue

        api_total = float(game["actual_total"])
        score_total = float(score["actual_total"])
        if api_total != score_total:
            base["prediction_status"] = "skipped"
            base["skip_reason"] = (
                f"Score mismatch: market provider has {api_total:g}, local history has {score_total:g}."
            )
            mismatched_scores += 1
            records.append(base)
            continue

        prior = history.loc[history["game_date"] < target_date].copy()
        base["prior_history_rows"] = int(len(prior))
        base["prior_max_date"] = (
            prior["game_date"].max().date().isoformat() if not prior.empty else ""
        )
        if len(prior) < args.min_prior_games:
            base["prediction_status"] = "skipped"
            base["skip_reason"] = (
                f"Only {len(prior)} prior score rows; need {args.min_prior_games}."
            )
            records.append(base)
            continue
        if not (prior["game_date"] < target_date).all():
            raise AssertionError("Chronological cutoff admitted target-day/future scores.")

        prediction = predict_game_total(
            home_team=str(game["home_team"]),
            away_team=str(game["away_team"]),
            bookmaker_total=float(game["total_close"]),
            game_date=target_date,
            history_df=prior,
            apply_experimental_adjustments=False,
        )
        if prediction.get("experimental_adjustments_applied", False):
            raise AssertionError("The closing-line backtest must use official baseline projections.")

        projected_total = round(float(prediction["projected_total"]), 2)
        close_line = round(float(game["total_close"]), 2)
        edge = round(projected_total - close_line, 2)
        recommendation, confidence_note = classify_total_edge(edge)
        if recommendation.endswith("Over"):
            side = "Over"
            selected_odds = float(game["over_odds_decimal"])
        elif recommendation.endswith("Under"):
            side = "Under"
            selected_odds = float(game["under_odds_decimal"])
        else:
            side = ""
            selected_odds = np.nan
        outcome, profit, stake_at_risk = settle_total_bet(
            side=side or None,
            actual_total=score_total,
            line=close_line,
            decimal_odds=None if not side else selected_odds,
            stake_units=args.stake_units,
        )

        base.update({
            "points_engine_baseline_total": round(float(prediction["points_engine_total"]), 2),
            "projected_total": projected_total,
            "model_error": round(projected_total - score_total, 2),
            "market_close_error": round(close_line - score_total, 2),
            "edge_vs_close": edge,
            "recommendation": recommendation,
            "confidence_note": confidence_note,
            "bet_side": side,
            "selected_decimal_odds": selected_odds,
            "bet_outcome": outcome,
            "stake_units": stake_at_risk,
            "profit_units": profit,
            "market_line_move_points": (round(close_line - float(game["total_open"]), 2) if pd.notna(game["total_open"]) else np.nan),
            "experimental_adjustments_applied": False,
            "entry_timestamp_utc": "",
            "bet_level_clv_points": np.nan,
        })
        records.append(base)

    results = pd.DataFrame(records)
    scored = results.loc[results["prediction_status"] == "scored"].copy()
    if scored.empty:
        raise ValueError(
            "No market games could be evaluated after score matching and prior-history checks. "
            "Check dates, selected bookmaker and data coverage."
        )
    holdout_start = _reserved_holdout_start(scored, args.holdout_games)
    results.loc[results["prediction_status"] == "scored", "evaluation_split"] = np.where(
        pd.to_datetime(scored["game_date"]).dt.normalize().to_numpy() >= holdout_start.to_datetime64(),
        "reserved_holdout",
        "development",
    )

    scored = results.loc[results["prediction_status"] == "scored"].copy()
    summaries = [
        _score_split(scored, split, strategy)
        for split in ["development", "reserved_holdout", "all_games"]
        for strategy in ["all_qualifying_edges", "strong_only", "lean_only"]
    ]
    summary = pd.DataFrame(summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "totals_closing_line_backtest_results.csv"
    summary_path = output_dir / "totals_closing_line_backtest_summary.csv"
    diagnostics_path = output_dir / "totals_closing_line_backtest_data_quality.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    diagnostics = {
        "status": "success",
        "provider": "DunkAPI",
        "market_file": str(market_path),
        "score_file": str(score_path),
        "market_rows": int(len(market)),
        "games_scored": int((results["prediction_status"] == "scored").sum()),
        "games_skipped": int((results["prediction_status"] != "scored").sum()),
        "unmatched_score_rows": unmatched_scores,
        "score_mismatch_rows": mismatched_scores,
        "selected_bookmakers": sorted(market["bookmaker"].dropna().astype(str).unique().tolist()),
        "reserved_holdout_games_requested": int(args.holdout_games),
        "reserved_holdout_start_date": holdout_start.date().isoformat(),
        "minimum_prior_score_rows": int(args.min_prior_games),
        "entry_time_clv_available": False,
        "methodology": (
            "Strict chronological score history; official Points Engine baseline; "
            "picks compared with the historical closing line and settled at the provider's "
            "closing Over/Under price. Opening-to-closing movement is descriptive only. "
            "This does not establish CLV for an earlier entry price."
        ),
        "outputs": [str(results_path), str(summary_path), str(diagnostics_path)],
    }
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))
    cols = [
        "split", "strategy", "games_scored", "model_mae", "market_close_mae",
        "bets", "wins", "losses", "pushes", "win_rate_pct_excluding_pushes",
        "stake_units_risked", "net_profit_units", "roi_pct", "mean_open_to_close_total_move",
    ]
    print(summary[cols].to_string(index=False))
    print("Closing-line ROI is a historical simulation, not a guarantee of future profit.")
    return results, summary, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronological NBA totals backtest using real historical closing lines.")
    parser.add_argument("--start-date", required=True, help="First Eastern calendar date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Last Eastern calendar date, YYYY-MM-DD inclusive.")
    parser.add_argument("--bookmaker", required=True, help="Exact provider label for one book, e.g. DraftKings.")
    parser.add_argument("--output-file", default=DEFAULT_MARKET_FILE)
    parser.add_argument("--scores-file", default=DEFAULT_SCORE_FILE)
    parser.add_argument("--plan-only", action="store_true", help="Estimate calls/credits only; do not call the API.")
    parser.add_argument("--confirm-api-cost", action="store_true", help="Required before any historical API call.")
    parser.add_argument("--max-api-credits", type=int, default=None, help="Hard maximum credit budget for this run.")
    parser.add_argument("--min-prior-games", type=int, default=100)
    parser.add_argument("--holdout-games", type=int, default=1000)
    parser.add_argument("--stake-units", type=float, default=1.0)
    parser.add_argument("--collect-only", action="store_true", help="Only collect historical closing-line data; do not run the backtest.")
    parser.add_argument("--market-file", default=DEFAULT_MARKET_FILE, help="Input market archive for --backtest-only.")
    parser.add_argument("--backtest-only", action="store_true", help="Run the backtest on an existing market archive without API calls.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.backtest_only:
            run_closing_line_backtest(args)
            return
        if args.plan_only:
            days = parse_date_range(args.start_date, args.end_date)
            credits = estimate_api_credits(len(days))
            print(json.dumps({
                "status": "plan_only",
                "daily_requests": len(days),
                "estimated_credits": credits,
                "bookmaker": args.bookmaker,
                "historical_endpoint_calls_made": 0,
            }, indent=2))
            return
        collect_market_data(args)
        if not args.collect_only:
            # Backtest immediately after the collection so results can be reviewed.
            args.market_file = args.output_file
            run_closing_line_backtest(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
