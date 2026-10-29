"""Build a leakage-resistant totals learning dataset from graded pick history.

Only saved forecasts with explicit timezone-aware timestamps strictly before
the game's tip-off are eligible. Historical rows without that provenance stay
in the audit ledger but are never silently treated as verified pregame data.
"""
import os
import pandas as pd


TOTALS_HISTORY_FILE = "totals_history.csv"
TOTALS_LEARNING_FILE = "totals_learning_dataset.csv"

PREDICTOR_COLUMNS = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "is_under",
    "is_over",
]
POST_GAME_COLUMNS = {
    "actual_total", "profit_loss", "result", "home_score", "away_score",
    "target",
}
PROVENANCE_COLUMNS = {"prediction_time_utc", "commence_time_utc"}
REQUIRED_HISTORY_COLUMNS = {
    "game_date", "home_team", "away_team", "pick_type",
    "projected_total", "sportsbook_total", "edge",
    "actual_total", "profit_loss", "result",
    "prediction_time_utc", "commence_time_utc",
}


def _parse_aware_utc(value):
    """Return a UTC Timestamp only when the supplied value includes a timezone."""
    if value is None or not str(value).strip():
        return pd.NaT
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if pd.isna(parsed) or parsed.tzinfo is None:
        return pd.NaT
    try:
        return parsed.tz_convert("UTC")
    except (TypeError, ValueError):
        return pd.NaT


def _legacy_synthetic_signature(df):
    """Identify rows matching the old built-in fake-data generator exactly."""
    needed = {
        "game_date", "pick_type", "recommendation", "projected_total",
        "sportsbook_total", "edge", "actual_total", "profit_loss", "result",
    }
    if not needed.issubset(df.columns):
        return pd.Series(False, index=df.index)

    game_date = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pick = df["pick_type"].astype(str).str.strip().str.lower()
    recommendation = df["recommendation"].astype(str).str.strip().str.lower()
    result = df["result"].astype(str).str.strip().str.lower()

    line = pd.to_numeric(df["sportsbook_total"], errors="coerce")
    projection = pd.to_numeric(df["projected_total"], errors="coerce")
    edge = pd.to_numeric(df["edge"], errors="coerce")
    actual = pd.to_numeric(df["actual_total"], errors="coerce")
    pnl = pd.to_numeric(df["profit_loss"], errors="coerce")

    under_win_signature = (
        result.eq("win") & pnl.eq(91) & actual.between(line - 20, line - 5)
    )
    under_loss_signature = (
        result.eq("loss") & pnl.eq(-100) & actual.between(line + 5, line + 20)
    )

    return (
        game_date.eq("2026-06-14")
        & pick.eq("under")
        & recommendation.eq("strong under")
        & edge.eq(-10)
        & (projection.round(6) == (line - 10).round(6))
        & (under_win_signature | under_loss_signature)
    ).fillna(False)


def _error(message, **details):
    return {"status": "error", "message": message, **details}


def build_totals_learning_dataset():
    if not os.path.exists(TOTALS_HISTORY_FILE):
        return _error("totals_history.csv not found")

    df = pd.read_csv(TOTALS_HISTORY_FILE)
    if df.empty:
        return _error("No totals picks found")

    original_rows = len(df)
    missing = sorted(REQUIRED_HISTORY_COLUMNS.difference(df.columns))
    if missing:
        provenance_missing = sorted(PROVENANCE_COLUMNS.difference(df.columns))
        if provenance_missing:
            return _error(
                "Totals learning blocked: saved picks lack verified pregame "
                "timestamp fields. New picks must record timezone-aware "
                "prediction_time_utc and commence_time_utc; old rows cannot be "
                "assumed to have been saved before tip-off.",
                rows=0,
                original_rows=original_rows,
                missing_columns=missing,
                provenance_columns_missing=provenance_missing,
                training_allowed=False,
            )
        return _error(
            f"Totals history is missing required columns: {missing}",
            rows=0,
            original_rows=original_rows,
            missing_columns=missing,
            training_allowed=False,
        )

    # Exclude new synthetic sources, explicit flags, and legacy known test rows.
    if "source" in df.columns:
        synthetic_source = df["source"].astype(str).str.lower().str.contains(
            r"synthetic|test", regex=True, na=False
        )
    else:
        synthetic_source = pd.Series(False, index=df.index)

    if "is_synthetic" in df.columns:
        synthetic_flag = df["is_synthetic"].astype(str).str.lower().isin(
            ["1", "true", "yes", "y"]
        )
    else:
        synthetic_flag = pd.Series(False, index=df.index)

    legacy_synthetic = _legacy_synthetic_signature(df)
    excluded_synthetic = synthetic_source | synthetic_flag | legacy_synthetic
    df = df.loc[~excluded_synthetic].copy()
    synthetic_rows_excluded = int(excluded_synthetic.sum())

    # Require parseable, timezone-aware timestamps, then verify strict pre-tipoff order.
    prediction_ts = df["prediction_time_utc"].apply(_parse_aware_utc)
    commence_ts = df["commence_time_utc"].apply(_parse_aware_utc)
    valid_timestamps = prediction_ts.notna() & commence_ts.notna()
    timestamp_invalid_rows_excluded = int((~valid_timestamps).sum())
    post_tipoff = valid_timestamps & (prediction_ts >= commence_ts)
    post_tipoff_rows_excluded = int(post_tipoff.sum())
    pregame = valid_timestamps & (prediction_ts < commence_ts)
    df = df.loc[pregame].copy()
    df["prediction_time_utc"] = prediction_ts.loc[pregame].map(
        lambda ts: ts.isoformat().replace("+00:00", "Z")
    )
    df["commence_time_utc"] = commence_ts.loc[pregame].map(
        lambda ts: ts.isoformat().replace("+00:00", "Z")
    )

    if df.empty:
        return _error(
            "No eligible pre-tipoff totals rows remain. Timestamp-missing, "
            "timezone-naive, or post-tipoff records are excluded from training.",
            rows=0,
            original_rows=original_rows,
            synthetic_rows_excluded=synthetic_rows_excluded,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            training_allowed=False,
        )

    # Only completed, settled Over/Under picks can become binary labels.
    df["result"] = df["result"].astype(str).str.strip().str.lower()
    df["pick_type"] = df["pick_type"].astype(str).str.strip().str.lower()
    df = df.loc[df["result"].isin(["win", "loss"])].copy()
    df = df.loc[df["pick_type"].isin(["over", "under"])].copy()
    if df.empty:
        return _error(
            "No eligible real, graded Over/Under picks found after provenance checks.",
            rows=0,
            original_rows=original_rows,
            synthetic_rows_excluded=synthetic_rows_excluded,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            training_allowed=False,
        )

    numeric_columns = [
        "projected_total", "sportsbook_total", "edge", "actual_total", "profit_loss"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    numeric_valid = df[numeric_columns].notna().all(axis=1)
    df = df.loc[numeric_valid].copy()

    # Reject inconsistent edges and result labels; pushes are not binary training outcomes.
    expected_edge = (df["projected_total"] - df["sportsbook_total"]).round(2)
    edge_valid = (df["edge"].round(2) - expected_edge).abs() <= 0.02
    actual = df["actual_total"]
    line = df["sportsbook_total"]
    over = df["pick_type"].eq("over")
    under = df["pick_type"].eq("under")
    settled = actual.ne(line)
    expected_win = (over & (actual > line)) | (under & (actual < line))
    label_valid = df["result"].eq("win").eq(expected_win)
    consistent = edge_valid & settled & label_valid
    inconsistent_rows_excluded = int((~consistent).sum())
    df = df.loc[consistent].copy()

    if df.empty:
        return _error(
            "No eligible rows remain after edge/result settlement checks.",
            rows=0,
            original_rows=original_rows,
            synthetic_rows_excluded=synthetic_rows_excluded,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            inconsistent_rows_excluded=inconsistent_rows_excluded,
            training_allowed=False,
        )

    df["is_under"] = (df["pick_type"] == "under").astype(int)
    df["is_over"] = (df["pick_type"] == "over").astype(int)
    df["target"] = expected_win.loc[df.index].astype(int)

    # These outcomes are retained for audit/label checks; only PREDICTOR_COLUMNS
    # are admitted into train_totals_model.FEATURES.
    columns = [
        "game_date", "prediction_time_utc", "commence_time_utc",
        "home_team", "away_team", "pick_type", *PREDICTOR_COLUMNS,
        "actual_total", "profit_loss", "result", "target",
    ]
    learning_df = df[columns].copy()
    os.makedirs(os.path.dirname(TOTALS_LEARNING_FILE) or ".", exist_ok=True)
    learning_df.to_csv(TOTALS_LEARNING_FILE, index=False)

    return {
        "status": "success",
        "rows": len(learning_df),
        "original_rows": original_rows,
        "synthetic_rows_excluded": synthetic_rows_excluded,
        "timestamp_invalid_rows_excluded": timestamp_invalid_rows_excluded,
        "post_tipoff_rows_excluded": post_tipoff_rows_excluded,
        "inconsistent_rows_excluded": inconsistent_rows_excluded,
        "ungraded_or_non_over_under_rows_excluded": (
            original_rows - synthetic_rows_excluded
            - timestamp_invalid_rows_excluded - post_tipoff_rows_excluded
            - len(learning_df) - inconsistent_rows_excluded
        ),
        "file": TOTALS_LEARNING_FILE,
        "predictor_columns": PREDICTOR_COLUMNS,
        "post_game_fields_excluded_from_predictors": sorted(POST_GAME_COLUMNS),
        "training_allowed": True,
        "provenance_rule": "prediction_time_utc must be timezone-aware and strictly earlier than commence_time_utc",
    }


if __name__ == "__main__":
    print(build_totals_learning_dataset())
