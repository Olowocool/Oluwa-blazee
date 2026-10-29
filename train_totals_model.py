"""Train the totals outcome classifier using verified pre-tipoff rows only."""
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_FILE = "totals_learning_dataset.csv"
MODEL_FILE = "models/totals_model_v2.joblib"

# Only information available before tip-off is permitted as a predictor.
FEATURES = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "is_under",
    "is_over",
]
PROVENANCE_COLUMNS = ["prediction_time_utc", "commence_time_utc"]
REQUIRED_COLUMNS = [
    *FEATURES, "target", "game_date", "actual_total", "pick_type",
    *PROVENANCE_COLUMNS,
]


def _parse_aware_utc(value):
    """Parse a timestamp only if the original string includes a timezone."""
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
        return _result("error", "totals_learning_dataset.csv not found")

    df = pd.read_csv(DATA_FILE)
    if len(df) < 20:
        return _result(
            "error",
            f"Need at least 20 graded totals rows. Current: {len(df)}",
            training_allowed=False,
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return _result(
            "error",
            "Training blocked: dataset is missing required fields, including "
            "pregame timestamp provenance. Rebuild it from verified pre-tipoff "
            f"records. Missing: {missing}",
            missing_columns=missing,
            training_allowed=False,
        )

    original_rows = len(df)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in FEATURES + ["target", "actual_total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[FEATURES + ["target", "actual_total"]] = df[
        FEATURES + ["target", "actual_total"]
    ].replace([float("inf"), float("-inf")], float("nan"))

    prediction_ts = df["prediction_time_utc"].apply(_parse_aware_utc)
    commence_ts = df["commence_time_utc"].apply(_parse_aware_utc)
    valid_timestamps = prediction_ts.notna() & commence_ts.notna()
    timestamp_invalid_rows_excluded = int((~valid_timestamps).sum())
    post_tipoff = valid_timestamps & (prediction_ts >= commence_ts)
    post_tipoff_rows_excluded = int(post_tipoff.sum())
    pregame = valid_timestamps & (prediction_ts < commence_ts)
    df = df.loc[pregame].copy()

    if df.empty:
        return _result(
            "error",
            "Training blocked: no rows have valid timezone-aware timestamps "
            "strictly before tip-off.",
            original_rows=original_rows,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            training_allowed=False,
        )

    # Defense in depth: the label and direction must agree with the settled total.
    picks = df["pick_type"].astype(str).str.strip().str.casefold()
    is_over = picks.eq("over")
    is_under = picks.eq("under")
    valid_direction = (
        ((df["is_over"] == 1) & (df["is_under"] == 0) & is_over)
        | ((df["is_under"] == 1) & (df["is_over"] == 0) & is_under)
    )
    line = df["sportsbook_total"]
    actual = df["actual_total"]
    settled = actual.notna() & line.notna() & actual.ne(line)
    expected_win = (is_over & (actual > line)) | (is_under & (actual < line))
    target_valid = df["target"].isin([0, 1]) & (df["target"].astype("Int64") == expected_win.astype(int))
    expected_edge = (df["projected_total"] - line).round(2)
    edge_valid = (df["edge"].round(2) - expected_edge).abs() <= 0.02
    valid = (
        df["game_date"].notna()
        & df[FEATURES + ["target", "actual_total"]].notna().all(axis=1)
        & valid_direction
        & settled
        & target_valid
        & edge_valid
    )
    inconsistent_rows_excluded = int((~valid).sum())
    df = df.loc[valid].copy()

    if len(df) < 20:
        return _result(
            "error",
            f"Need at least 20 valid pre-tipoff graded rows after provenance and "
            f"label checks. Current: {len(df)}",
            rows=len(df),
            original_rows=original_rows,
            timestamp_invalid_rows_excluded=timestamp_invalid_rows_excluded,
            post_tipoff_rows_excluded=post_tipoff_rows_excluded,
            inconsistent_rows_excluded=inconsistent_rows_excluded,
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
        "post_tipoff_rows_excluded": post_tipoff_rows_excluded,
        "inconsistent_rows_excluded": inconsistent_rows_excluded,
        "model_file": MODEL_FILE,
        "features_used": FEATURES,
        "calibration_status": "not_calibrated",
        "training_allowed": True,
        "note": (
            "Chronological holdout accuracy is not evidence of calibration or "
            "betting profitability. Only timestamp-verified pre-tipoff records "
            "were eligible for training."
        ),
    }


if __name__ == "__main__":
    print(train_totals_model())
