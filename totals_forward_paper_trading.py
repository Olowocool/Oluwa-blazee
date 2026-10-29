"""Auditable forward paper-trading evaluation for timestamped NBA totals picks.

This module does not train a model, infer probabilities, or enable betting.
It evaluates only completed Over/Under paper picks with verifiable entry-market
snapshots and reports realized flat-stake results plus line movement to the
latest valid pre-tipoff snapshot available from the same bookmaker.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_TOTALS_HISTORY_FILE = Path("totals_history.csv")
DEFAULT_MARKET_SNAPSHOT_FILE = Path("data/live_totals_market_snapshots.csv")
DEFAULT_PICKS_OUTPUT_FILE = Path("outputs/totals_forward_paper_trading_picks.csv")
DEFAULT_SUMMARY_OUTPUT_FILE = Path("outputs/totals_forward_paper_trading_summary.json")
MAX_MARKET_SNAPSHOT_AGE_SECONDS = 300
LINE_MATCH_TOLERANCE = 0.01
TIMESTAMP_MATCH_TOLERANCE_SECONDS = 1.0

PICK_REQUIRED_COLUMNS = {
    "home_team",
    "away_team",
    "pick_type",
    "prediction_time_utc",
    "commence_time_utc",
    "market_snapshot_time_utc",
    "market_last_update_utc",
    "market_snapshot_total_line",
    "over_odds",
    "under_odds",
    "totals_bookmaker",
    "market_snapshot_source",
    "source_event_id",
    "sportsbook_total",
    "actual_total",
    "result",
    "stake",
}
SNAPSHOT_REQUIRED_COLUMNS = {
    "home_team",
    "away_team",
    "totals_bookmaker",
    "source_event_id",
    "snapshot_timestamp_utc",
    "market_last_update_utc",
    "commence_time_utc",
    "total_line",
    "over_odds",
    "under_odds",
    "snapshot_source",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _aware_utc(value: Any) -> datetime | None:
    """Parse an explicit timezone-aware timestamp and normalize it to UTC."""
    text = _clean(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_decimal_odds(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 1.0 else None


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _truthy(value: Any) -> bool:
    return _clean(value).casefold() in {"1", "true", "yes", "y"}


def _same_text(left: Any, right: Any) -> bool:
    return _clean(left).casefold() == _clean(right).casefold()


def _same_number(left: Any, right: Any, tolerance: float = LINE_MATCH_TOLERANCE) -> bool:
    a, b = _number(left), _number(right)
    return a is not None and b is not None and abs(a - b) <= tolerance


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]] | None, str | None]:
    path = Path(path)
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None, f"CSV has no header: {path}"
            return [dict(row) for row in reader], None
    except (OSError, csv.Error, UnicodeError) as exc:
        return None, f"Could not read {path}: {exc}"


def _team_key(value: Any) -> str:
    return " ".join(_clean(value).casefold().split())


def _same_event(pick: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    pick_event_id = _clean(pick.get("source_event_id"))
    snapshot_event_id = _clean(snapshot.get("source_event_id"))
    if pick_event_id and snapshot_event_id:
        if pick_event_id != snapshot_event_id:
            return False
    if _team_key(pick.get("home_team")) != _team_key(snapshot.get("home_team")):
        return False
    if _team_key(pick.get("away_team")) != _team_key(snapshot.get("away_team")):
        return False
    pick_commence = _aware_utc(pick.get("commence_time_utc"))
    snapshot_commence = _aware_utc(snapshot.get("commence_time_utc"))
    if pick_commence is None or snapshot_commence is None:
        return False
    return abs((pick_commence - snapshot_commence).total_seconds()) <= TIMESTAMP_MATCH_TOLERANCE_SECONDS


def _settlement(actual: float, line: float, pick_type: str) -> str:
    if math.isclose(actual, line, rel_tol=0.0, abs_tol=1e-9):
        return "push"
    if pick_type == "over":
        return "win" if actual > line else "loss"
    return "win" if actual < line else "loss"


def _snapshot_is_valid(snapshot: dict[str, Any]) -> tuple[bool, datetime | None]:
    snapshot_time = _aware_utc(snapshot.get("snapshot_timestamp_utc"))
    commence_time = _aware_utc(snapshot.get("commence_time_utc"))
    if snapshot_time is None or commence_time is None or snapshot_time >= commence_time:
        return False, None
    market_update_text = _clean(snapshot.get("market_last_update_utc"))
    if market_update_text:
        market_update = _aware_utc(market_update_text)
        if market_update is None or market_update > snapshot_time or market_update >= commence_time:
            return False, None
    if _positive_number(snapshot.get("total_line")) is None:
        return False, None
    if _valid_decimal_odds(snapshot.get("over_odds")) is None:
        return False, None
    if _valid_decimal_odds(snapshot.get("under_odds")) is None:
        return False, None
    if not _clean(snapshot.get("totals_bookmaker")):
        return False, None
    return True, snapshot_time


def _snapshot_matches_entry(
    pick: dict[str, Any],
    snapshot: dict[str, Any],
    entry_time: datetime,
    line: float,
    over_odds: float,
    under_odds: float,
    bookmaker: str,
    source: str,
) -> bool:
    valid, snapshot_time = _snapshot_is_valid(snapshot)
    if not valid or snapshot_time is None:
        return False
    if abs((snapshot_time - entry_time).total_seconds()) > TIMESTAMP_MATCH_TOLERANCE_SECONDS:
        return False
    if not _same_event(pick, snapshot):
        return False
    if not _same_text(bookmaker, snapshot.get("totals_bookmaker")):
        return False
    if not _same_number(line, snapshot.get("total_line")):
        return False
    if not _same_number(over_odds, snapshot.get("over_odds"), tolerance=0.01):
        return False
    if not _same_number(under_odds, snapshot.get("under_odds"), tolerance=0.01):
        return False
    if source and not _same_text(source, snapshot.get("snapshot_source")):
        return False
    return True


def _exclude(row: dict[str, Any], reason: str) -> dict[str, Any]:
    row["evaluation_status"] = "excluded"
    row["exclusion_reason"] = reason
    return row


def _evaluate_one(
    pick: dict[str, Any],
    index: int,
    snapshots: list[dict[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pick_row": index,
        "game_date": _clean(pick.get("game_date")),
        "home_team": _clean(pick.get("home_team")),
        "away_team": _clean(pick.get("away_team")),
        "pick_type": _clean(pick.get("pick_type")).title(),
        "prediction_time_utc": _clean(pick.get("prediction_time_utc")),
        "commence_time_utc": _clean(pick.get("commence_time_utc")),
        "market_snapshot_time_utc": _clean(pick.get("market_snapshot_time_utc")),
        "totals_bookmaker": _clean(pick.get("totals_bookmaker")),
        "source_event_id": _clean(pick.get("source_event_id")),
        "entry_line": None,
        "entry_odds": None,
        "latest_pre_tip_line": None,
        "latest_pre_tip_snapshot_time_utc": "",
        "line_clv_points": None,
        "computed_result": "",
        "stake": None,
        "net_profit": None,
        "tracker_profit_loss": _number(pick.get("profit_loss")),
        "evaluation_status": "",
        "exclusion_reason": "",
    }

    pick_type = _clean(pick.get("pick_type")).casefold()
    if pick_type not in {"over", "under"}:
        return _exclude(out, "no_bet_or_invalid_pick")
    if _truthy(pick.get("experimental_adjustments_applied")):
        return _exclude(out, "experimental_adjustment_pick")

    prediction_time = _aware_utc(pick.get("prediction_time_utc"))
    entry_time = _aware_utc(pick.get("market_snapshot_time_utc"))
    commence_time = _aware_utc(pick.get("commence_time_utc"))
    if prediction_time is None or entry_time is None or commence_time is None:
        return _exclude(out, "missing_or_timezone_naive_timestamp")
    if prediction_time >= commence_time or entry_time >= commence_time:
        return _exclude(out, "post_tipoff_prediction_or_snapshot")
    if entry_time > prediction_time:
        return _exclude(out, "snapshot_after_prediction")
    age_seconds = (prediction_time - entry_time).total_seconds()
    if age_seconds < 0 or age_seconds > MAX_MARKET_SNAPSHOT_AGE_SECONDS:
        return _exclude(out, "stale_entry_snapshot")

    update_text = _clean(pick.get("market_last_update_utc"))
    if update_text:
        market_update = _aware_utc(update_text)
        if market_update is None:
            return _exclude(out, "invalid_market_update_timestamp")
        if market_update > entry_time or market_update >= commence_time:
            return _exclude(out, "invalid_market_update_time")

    line = _positive_number(pick.get("market_snapshot_total_line"))
    saved_line = _positive_number(pick.get("sportsbook_total"))
    over_odds = _valid_decimal_odds(pick.get("over_odds"))
    under_odds = _valid_decimal_odds(pick.get("under_odds"))
    stake = _positive_number(pick.get("stake"))
    bookmaker = _clean(pick.get("totals_bookmaker"))
    source = _clean(pick.get("market_snapshot_source"))
    if line is None or saved_line is None:
        return _exclude(out, "invalid_entry_line")
    if abs(line - saved_line) > LINE_MATCH_TOLERANCE:
        return _exclude(out, "entry_line_mismatch")
    if over_odds is None or under_odds is None:
        return _exclude(out, "invalid_entry_prices")
    if stake is None:
        return _exclude(out, "invalid_stake")
    if not bookmaker or not source:
        return _exclude(out, "missing_bookmaker_or_source")

    # Verify that the exact entry market in the saved pick exists in the
    # append-only observation ledger; timestamps alone are not sufficient.
    entry_matches = [
        row for row in snapshots
        if _snapshot_matches_entry(
            pick, row, entry_time, line, over_odds, under_odds, bookmaker, source
        )
    ]
    if not entry_matches:
        return _exclude(out, "entry_snapshot_not_found_or_mismatched")

    result_text = _clean(pick.get("result")).casefold()
    actual = _number(pick.get("actual_total"))
    if actual is None or result_text not in {"win", "loss", "push"}:
        return _exclude(out, "unsettled_pick")
    computed_result = _settlement(actual, line, pick_type)
    if computed_result != result_text:
        return _exclude(out, "settlement_label_mismatch")

    entry_odds = over_odds if pick_type == "over" else under_odds
    net_profit = (
        stake * (entry_odds - 1.0) if computed_result == "win"
        else -stake if computed_result == "loss"
        else 0.0
    )

    later_snapshots: list[tuple[datetime, dict[str, str]]] = []
    for snapshot in snapshots:
        valid, snap_time = _snapshot_is_valid(snapshot)
        if not valid or snap_time is None or snap_time <= entry_time:
            continue
        if snap_time >= commence_time or not _same_event(pick, snapshot):
            continue
        if not _same_text(bookmaker, snapshot.get("totals_bookmaker")):
            continue
        later_snapshots.append((snap_time, snapshot))

    latest_line = None
    latest_time = None
    line_clv = None
    if later_snapshots:
        latest_time, latest = max(later_snapshots, key=lambda item: item[0])
        latest_line = _positive_number(latest.get("total_line"))
        if latest_line is not None:
            # Positive CLV means the saved line was better than the later line.
            line_clv = (
                latest_line - line if pick_type == "over"
                else line - latest_line
            )

    out.update({
        "entry_line": round(line, 4),
        "entry_odds": round(entry_odds, 4),
        "latest_pre_tip_line": round(latest_line, 4) if latest_line is not None else None,
        "latest_pre_tip_snapshot_time_utc": _utc_text(latest_time),
        "line_clv_points": round(line_clv, 4) if line_clv is not None else None,
        "computed_result": computed_result,
        "stake": round(stake, 4),
        "net_profit": round(net_profit, 4),
        "evaluation_status": "eligible",
        "exclusion_reason": "",
    })
    return out


def _write_picks(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pick_row", "game_date", "home_team", "away_team", "pick_type",
        "prediction_time_utc", "commence_time_utc", "market_snapshot_time_utc",
        "totals_bookmaker", "source_event_id", "entry_line", "entry_odds",
        "latest_pre_tip_line", "latest_pre_tip_snapshot_time_utc", "line_clv_points",
        "computed_result", "stake", "net_profit", "tracker_profit_loss",
        "evaluation_status", "exclusion_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_forward_paper_trading(
    totals_history_path: str | Path = DEFAULT_TOTALS_HISTORY_FILE,
    market_snapshot_path: str | Path = DEFAULT_MARKET_SNAPSHOT_FILE,
    picks_output_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate completed Over/Under picks against the market captured at entry.

    ROI is calculated from the captured decimal price and recorded stake.
    Line CLV is the move from the entry line to the latest valid same-bookmaker
    snapshot strictly before tip-off; it is not represented as a verified
    sportsbook closing line when no final close observation was captured.
    """
    picks, picks_error = _read_csv(totals_history_path)
    if picks_error:
        return {"status": "error", "message": picks_error, "picks_seen": 0, "eligible_picks": 0}
    assert picks is not None
    if not picks:
        return {"status": "insufficient_data", "message": "No saved totals picks found.", "picks_seen": 0, "eligible_picks": 0}

    pick_fields = set(picks[0].keys())
    missing_pick_fields = sorted(PICK_REQUIRED_COLUMNS - pick_fields)
    if missing_pick_fields:
        return {
            "status": "error",
            "message": "Totals history is missing required columns.",
            "missing_columns": missing_pick_fields,
            "picks_seen": len(picks),
            "eligible_picks": 0,
        }

    snapshots, snapshots_error = _read_csv(market_snapshot_path)
    if snapshots_error:
        evaluated = [
            _exclude({
                "pick_row": i,
                "game_date": _clean(pick.get("game_date")),
                "home_team": _clean(pick.get("home_team")),
                "away_team": _clean(pick.get("away_team")),
                "pick_type": _clean(pick.get("pick_type")).title(),
                "evaluation_status": "",
                "exclusion_reason": "",
            }, "snapshot_ledger_unavailable")
            for i, pick in enumerate(picks, start=2)
        ]
        result = {
            "status": "insufficient_data",
            "message": snapshots_error,
            "picks_seen": len(picks),
            "eligible_picks": 0,
            "excluded_picks": len(picks),
            "snapshot_ledger_available": False,
            "probabilities_used": False,
            "live_betting_enabled": False,
        }
        if picks_output_path is not None:
            _write_picks(picks_output_path, evaluated)
        if summary_output_path is not None:
            _write_json(summary_output_path, result)
        return result
    assert snapshots is not None
    if snapshots:
        missing_snapshot_fields = sorted(SNAPSHOT_REQUIRED_COLUMNS - set(snapshots[0].keys()))
        if missing_snapshot_fields:
            return {
                "status": "error",
                "message": "Market snapshot ledger is missing required columns.",
                "missing_columns": missing_snapshot_fields,
                "picks_seen": len(picks),
                "eligible_picks": 0,
            }
    else:
        missing_snapshot_fields = []

    evaluated = [
        _evaluate_one(pick, index, snapshots)
        for index, pick in enumerate(picks, start=2)
    ]
    eligible = [row for row in evaluated if row.get("evaluation_status") == "eligible"]
    excluded_counts = Counter(
        row.get("exclusion_reason", "") for row in evaluated
        if row.get("evaluation_status") == "excluded"
    )

    wins = sum(row["computed_result"] == "win" for row in eligible)
    losses = sum(row["computed_result"] == "loss" for row in eligible)
    pushes = sum(row["computed_result"] == "push" for row in eligible)
    total_stake = sum(float(row["stake"]) for row in eligible)
    net_profit = sum(float(row["net_profit"]) for row in eligible)
    line_clv_values = [
        float(row["line_clv_points"]) for row in eligible
        if row.get("line_clv_points") not in (None, "")
    ]
    decisions = wins + losses
    positive_clv = sum(value > 0 for value in line_clv_values)
    negative_clv = sum(value < 0 for value in line_clv_values)
    flat_clv = sum(value == 0 for value in line_clv_values)

    result = {
        "status": "success" if eligible else "insufficient_data",
        "picks_seen": len(picks),
        "eligible_picks": len(eligible),
        "excluded_picks": len(evaluated) - len(eligible),
        "excluded_by_reason": dict(sorted(excluded_counts.items())),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_pct_excluding_pushes": round(100.0 * wins / decisions, 4) if decisions else None,
        "stake_risked": round(total_stake, 4),
        "net_profit": round(net_profit, 4),
        "roi_pct": round(100.0 * net_profit / total_stake, 4) if total_stake else None,
        "picks_with_later_valid_pre_tip_snapshot": len(line_clv_values),
        "mean_line_clv_points": round(mean(line_clv_values), 4) if line_clv_values else None,
        "median_line_clv_points": round(median(line_clv_values), 4) if line_clv_values else None,
        "positive_line_clv_pct": round(100.0 * positive_clv / len(line_clv_values), 4) if line_clv_values else None,
        "positive_line_clv_count": positive_clv,
        "negative_line_clv_count": negative_clv,
        "flat_line_clv_count": flat_clv,
        "snapshot_ledger_available": True,
        "snapshot_ledger_rows": len(snapshots),
        "entry_snapshot_max_age_seconds": MAX_MARKET_SNAPSHOT_AGE_SECONDS,
        "line_clv_definition": "Over: later line minus entry line; Under: entry line minus later line; positive is favorable.",
        "pnl_definition": "Flat stake at captured decimal odds; win=stake*(odds-1), loss=-stake, push=0.",
        "probabilities_used": False,
        "calibrated_ev_evaluated": False,
        "live_betting_enabled": False,
    }

    if picks_output_path is not None:
        _write_picks(picks_output_path, evaluated)
    if summary_output_path is not None:
        _write_json(summary_output_path, result)
    return result


def _write_json(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    result = evaluate_forward_paper_trading(
        picks_output_path=DEFAULT_PICKS_OUTPUT_FILE,
        summary_output_path=DEFAULT_SUMMARY_OUTPUT_FILE,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") in {"success", "insufficient_data"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
