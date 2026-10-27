import os
import pandas as pd


TOTALS_HISTORY_FILE = "totals_history.csv"
TOTALS_LEARNING_FILE = "totals_learning_dataset.csv"

# Outcome fields are retained for auditing/labels only, never as predictors.
PREDICTOR_COLUMNS = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "is_under",
    "is_over",
]
POST_GAME_COLUMNS = {"actual_total", "profit_loss", "result", "home_score", "away_score"}


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


def build_totals_learning_dataset():
    if not os.path.exists(TOTALS_HISTORY_FILE):
        return {
            "status": "error",
            "message": "totals_history.csv not found",
        }

    df = pd.read_csv(TOTALS_HISTORY_FILE)
    if df.empty:
        return {
            "status": "error",
            "message": "No totals picks found",
        }

    original_rows = len(df)

    # New synthetic data is stored separately. Also exclude explicitly tagged
    # synthetic rows and rows matching the exact legacy test-data signature.
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

    df = df[df["result"].astype(str).str.lower().isin(["win", "loss"])].copy()
    if df.empty:
        return {
            "status": "error",
            "message": (
                "No eligible real graded totals bets found after excluding "
                "synthetic/test records."
            ),
        }

    required = ["game_date", "home_team", "away_team", "pick_type",
                "projected_total", "sportsbook_total", "edge",
                "actual_total", "profit_loss", "result"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "status": "error",
            "message": f"Totals history is missing required columns: {missing}",
        }

    for column in [
        "projected_total", "sportsbook_total", "edge", "actual_total", "profit_loss"
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["is_under"] = (df["pick_type"].astype(str).str.lower() == "under").astype(int)
    df["is_over"] = (df["pick_type"].astype(str).str.lower() == "over").astype(int)
    df["target"] = (df["result"].astype(str).str.lower() == "win").astype(int)

    # Sanity guard: post-game fields can be labels/audit columns, but never
    # predictors. Keep them in the dataset only for traceability.
    learning_df = df[
        [
            "game_date", "home_team", "away_team", "pick_type",
            "projected_total", "sportsbook_total", "edge",
            "actual_total", "profit_loss", "is_under", "is_over", "target",
        ]
    ].copy()

    os.makedirs(os.path.dirname(TOTALS_LEARNING_FILE) or ".", exist_ok=True)
    learning_df.to_csv(TOTALS_LEARNING_FILE, index=False)

    return {
        "status": "success",
        "rows": len(learning_df),
        "synthetic_rows_excluded": int(excluded_synthetic.sum()),
        "ungraded_rows_excluded": original_rows - int(excluded_synthetic.sum()) - len(learning_df),
        "file": TOTALS_LEARNING_FILE,
        "predictor_columns": PREDICTOR_COLUMNS,
        "post_game_fields_excluded_from_predictors": sorted(POST_GAME_COLUMNS),
    }


if __name__ == "__main__":
    print(build_totals_learning_dataset())
