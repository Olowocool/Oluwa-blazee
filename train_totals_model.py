"""Train the totals outcome classifier using verified pre-tipoff data only."""
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_FILE = "totals_learning_dataset.csv"
MODEL_FILE = "models/totals_model_v2.joblib"

# Outcome and provenance fields are never predictors.
FEATURES = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "is_under",
    "is_over",
]
PROVENANCE_COLUMNS = [
    "prediction_time_utc",
    "market_snapshot_time_utc",
    "commence_time_utc",
]
MARKET_COLUMNS = [
    "market_snapshot_total_line",
    "over_odds",
    "under_odds",
    "totals_bookmaker",
    "market_snapshot_source",
    "market_last_update_utc",
]
REQUIRED_COLUMNS = [
    *FEATURES,
    "target",
    "game_date",
    "actual_total",
    "pick_type",
    *PROVENANCE_COLUMNS,
    *MARKET_COLUMNS,
]


def _parse_aware_utc(value):
    """Parse only timezone-aware timestamps; a naive timestamp is not auditable."""
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


def _result(status, message, **details):
    return {"status": status, "message": message, **details}


def train_totals_model():
    if not os.path.exists(DATA_FILE):
        return _result("error", "totals_learning_dataset.csv not found", training_allowed=False)

    df = pd.read_csv(DATA_FILE)
    if len(df) < 20:
        return _result(
            "error",
            f"Need at least 20 graded totals rows. Current: {len(df)}",
            training_allowed=False,
        )

    original_rows = len(df)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return _result(
            "error",
            "Training blocked: dataset lacks verified pregame market-snapshot fields "
            "(prediction timestamp, observed snapshot timestamp, tip-off, bookmaker, "
            f"paired Over/Under prices, and same-snapshot line). Missing: {missing}",
            missing_columns=missing,
            training_allowed=False,
        )

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in FEATURES + ["target", "actual_total", "market_snapshot_total_line", "over_odds", "under_odds"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[FEATURES + ["target", "actual_total", "market_snapshot_total_line", "over_odds", "under_odds"]] = df[
        FEATURES + ["target", "actual_total", "market_snapshot_total_line", "over_odds", "under_odds"]
    ].replace([float("inf"), float("-inf")], float("nan"))

    # Check the actual observation time of the market line used by the prediction.
    prediction_ts = df["prediction_time_utc"].apply(_parse_aware_utc)
    snapshot_ts = df["market_snapshot_time_utc"].apply(_parse_aware_utc)
    commence_ts = df["commence_time_utc"].apply(_parse_aware_utc)
    valid_timestamps = prediction_ts.notna() & snapshot_ts.notna() & commence_ts.notna()
    timestamp_invalid_rows_excluded = int((~valid_timestamps).sum())
    df = df.loc[valid_timestamps].copy()
    prediction_ts = prediction_ts.loc[df.index]
    snapshot_ts = snapshot_ts.loc[df.index]
    commence_ts = commence_ts.loc[df.index]

    market_update_raw = df["market_last_update_utc"].fillna("").astype(str).str.strip()
    market_update_ts = df["market_last_update_utc"].apply(_parse_aware_utc).loc[df.index]
    bad_market_update = (
        market_update_raw.ne("") & market_update_ts.isna()
    ) | (market_update_ts.notna() & (market_update_ts > snapshot_ts))
    invalid_market_update_rows_excluded = int(bad_market_update.sum())
    df = df.loc[~bad_market_update].copy()
    prediction_ts = prediction_ts.loc[df.index]
    snapshot_ts = snapshot_ts.loc[df.index]
    commence_ts = commence_ts.loc[df.index]

    post_tipoff = (prediction_ts >= commence_ts) | (snapshot_ts >= commence_ts)
    post_tipoff_rows_excluded = int(post_tipoff.sum())
    df = df.loc[~post_tipoff].copy()
    prediction_ts = prediction_ts.loc[df.index]
    snapshot_ts = snapshot_ts.loc[df.index]
    commence_ts = commence_ts.loc[df.index]

    snapshot_after_prediction = snapshot_ts > prediction_ts
    market_snapshot_after_prediction_rows_excluded = int(snapshot_after_prediction.sum())
    df = df.loc[~snapshot_after_prediction].copy()

    if df.empty:
        return _result(
            "error",
            "Training blocked: no records have verifiable market snapshots at or before "
            "forecast generation and both timestamps strictly before scheduled tip-off.",
            original_rows=original_rows,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            invalid_market_update_rows_excluded=invalid_market_update_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            market_snapshot_after_prediction_rows_excluded=market_snapshot_after_prediction_rows_excluded,
            training_allowed=False,
        )

    # Revalidate line/odds/bookmaker provenance even if a CSV was edited manually.
    picks = df["pick_type"].astype(str).str.strip().str.casefold()
    is_over = picks.eq("over")
    is_under = picks.eq("under")
    line = df["sportsbook_total"]
    actual = df["actual_total"]
    snap_line = df["market_snapshot_total_line"]
    edge = df["edge"]
    expected_edge = (df["projected_total"] - line).round(2)
    edge_valid = (edge.round(2) - expected_edge).abs() <= 0.02
    line_match = (snap_line - line).abs() <= 0.01
    odds_valid = (df["over_odds"] > 1.0) & (df["under_odds"] > 1.0)
    bookmaker_valid = df["totals_bookmaker"].astype(str).str.strip().ne("")
    source_valid = df["market_snapshot_source"].astype(str).str.strip().ne("")
    direction_valid = (
        ((df["is_over"] == 1) & (df["is_under"] == 0) & is_over)
        | ((df["is_under"] == 1) & (df["is_over"] == 0) & is_under)
    )
    settled = actual.notna() & line.notna() & actual.ne(line)
    expected_win = (is_over & (actual > line)) | (is_under & (actual < line))
    target_valid = df["target"].isin([0, 1]) & (df["target"].astype("Int64") == expected_win.astype(int))
    numeric_valid = df[FEATURES + ["target", "actual_total", "market_snapshot_total_line", "over_odds", "under_odds"]].notna().all(axis=1)
    valid = numeric_valid & edge_valid & line_match & odds_valid & bookmaker_valid & source_valid & direction_valid & settled & target_valid
    market_or_label_rows_excluded = int((~valid).sum())
    df = df.loc[valid].copy()

    if len(df) < 20:
        return _result(
            "error",
            f"Need at least 20 verified pre-tipoff graded rows after market/label validation. Current: {len(df)}",
            rows=len(df),
            original_rows=original_rows,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            invalid_market_update_rows_excluded=invalid_market_update_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            market_snapshot_after_prediction_rows_excluded=market_snapshot_after_prediction_rows_excluded,
            market_or_label_rows_excluded=market_or_label_rows_excluded,
            training_allowed=False,
        )

    df["target"] = df["target"].astype(int)
    df = df.sort_values(["game_date"]).reset_index(drop=True)

    unique_dates = sorted(df["game_date"].dt.normalize().unique())
    if len(unique_dates) < 2:
        return _result(
            "error",
            "Chronological validation requires at least two distinct game dates.",
            rows=len(df),
            training_allowed=False,
        )

    split_at = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.70)))
    train_dates = set(unique_dates[:split_at])
    train_mask = df["game_date"].dt.normalize().isin(train_dates)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    if train_df["target"].nunique() != 2:
        return _result(
            "error",
            "Chronological training window must contain both wins and losses.",
            rows=len(df),
            training_rows=len(train_df),
            holdout_rows=len(test_df),
            training_allowed=False,
        )
    if test_df.empty or test_df["target"].nunique() != 2:
        return _result(
            "error",
            "Chronological holdout must contain both wins and losses; collect more "
            "verified pre-tipoff graded dates.",
            rows=len(df),
            training_rows=len(train_df),
            holdout_rows=len(test_df),
            training_allowed=False,
        )

    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df["target"]
    X_test = test_df[FEATURES].fillna(0)
    y_test = test_df["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    os.makedirs(os.path.dirname(MODEL_FILE) or ".", exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    return {
        "status": "success",
        "accuracy": round(accuracy, 2),
        "rows": len(df),
        "original_rows": original_rows,
        "training_rows": len(train_df),
        "holdout_rows": len(test_df),
        "training_through": train_df["game_date"].max().date().isoformat(),
        "holdout_from": test_df["game_date"].min().date().isoformat(),
        "timestamp_invalid_rows_excluded": timestamp_invalid_rows_excluded,
        "invalid_market_update_rows_excluded": invalid_market_update_rows_excluded,
        "post_tipoff_rows_excluded": post_tipoff_rows_excluded,
        "market_snapshot_after_prediction_rows_excluded": market_snapshot_after_prediction_rows_excluded,
        "market_or_label_rows_excluded": market_or_label_rows_excluded,
        "model_file": MODEL_FILE,
        "features_used": FEATURES,
        "calibration_status": "not_calibrated",
        "training_allowed": True,
        "note": (
            "Chronological holdout accuracy is not evidence of calibration or "
            "betting profitability. Only verified pre-tipoff market snapshots "
            "with matching line/odds provenance were eligible."
        ),
    }


if __name__ == "__main__":
    print(train_totals_model())
