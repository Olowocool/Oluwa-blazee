"""Robustness and paired uncertainty analysis for raw-total candidates.

Run from the project root:
    python raw_total_candidate_robustness.py

Reads outputs/raw_total_candidate_predictions.csv only. No model is retrained
and no prediction/production file is changed.

Reports chronological block performance and date-cluster bootstrap confidence
intervals for paired MAE/RMSE differences versus V3.
"""
import os
import numpy as np
import pandas as pd

INPUT = "outputs/raw_total_candidate_predictions.csv"
OUT_DIR = "outputs"
SUMMARY_OUT = os.path.join(OUT_DIR, "raw_total_candidate_robustness_summary.csv")
MONTHLY_OUT = os.path.join(OUT_DIR, "raw_total_candidate_monthly.csv")
BLOCKS_OUT = os.path.join(OUT_DIR, "raw_total_candidate_chronological_blocks.csv")
BOOTSTRAP_OUT = os.path.join(OUT_DIR, "raw_total_candidate_holdout_bootstrap.csv")

MODEL_COLUMNS = [
    ("V2 raw", "v2_raw_total"),
    ("V3 candidate", "v3_raw_total"),
    ("V4 90-day", "V4_HL90_P8"),
    ("V4 180-day", "V4_HL180_P8"),
    ("V4 365-day", "V4_HL365_P8"),
    ("V4 365-day stronger shrinkage", "V4_HL365_P20"),
    ("V4 730-day", "V4_HL730_P8"),
]
BOOTSTRAP_REPLICATES = 3000
RANDOM_SEED = 20261031


def score_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "games": int(len(error)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_pred_minus_actual": float(error.mean()),
        "within_5_pct": float((absolute <= 5).mean() * 100),
        "within_10_pct": float((absolute <= 10).mean() * 100),
    }


def load_predictions():
    if not os.path.isfile(INPUT):
        raise FileNotFoundError(
            f"Missing {INPUT}. Run totals_backtest_raw_candidates.py first."
        )
    df = pd.read_csv(INPUT)
    required = ["game_date", "actual_total", "period"] + [
        column for _, column in MODEL_COLUMNS
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in ["actual_total"] + [column for _, column in MODEL_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["game_date", "actual_total", "period"])
    df = df.dropna(subset=[column for _, column in MODEL_COLUMNS])
    if df.empty:
        raise ValueError("No complete candidate prediction rows remain.")
    return df.sort_values(["game_date", "home_team", "away_team"] if
                          {"home_team", "away_team"}.issubset(df.columns)
                          else ["game_date"]).reset_index(drop=True)


def metric_table(df, group_columns=None):
    group_columns = group_columns or []
    rows = []
    grouped = [((), df)] if not group_columns else df.groupby(
        group_columns, dropna=False, sort=True
    )
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        labels = dict(zip(group_columns, key))
        for model_name, column in MODEL_COLUMNS:
            row = {"model": model_name, **labels}
            row.update(score_metrics(group["actual_total"], group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def chronological_blocks(dev):
    dates = np.array(sorted(dev["game_date"].unique()))
    if len(dates) < 8:
        return pd.DataFrame()
    # Four equal-count contiguous date blocks within development only.
    chunks = [chunk for chunk in np.array_split(dates, 4) if len(chunk)]
    rows = []
    for fold, chunk in enumerate(chunks, start=1):
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        group = dev[dev["game_date"].isin(chunk)]
        for model_name, column in MODEL_COLUMNS:
            row = {
                "development_block": fold,
                "block_start": start.date().isoformat(),
                "block_end": end.date().isoformat(),
                "model": model_name,
            }
            row.update(score_metrics(group["actual_total"], group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_holdout(holdout):
    if holdout.empty:
        return pd.DataFrame()

    models_to_compare = [
        ("V2 raw", "v2_raw_total"),
        ("V4 90-day", "V4_HL90_P8"),
        ("V4 180-day", "V4_HL180_P8"),
        ("V4 365-day", "V4_HL365_P8"),
        ("V4 365-day stronger shrinkage", "V4_HL365_P20"),
        ("V4 730-day", "V4_HL730_P8"),
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(holdout["game_date"].unique()))
    groups = [holdout.index[holdout["game_date"] == date].to_numpy() for date in dates]
    index_positions = {idx: pos for pos, idx in enumerate(holdout.index)}
    date_positions = [
        np.array([index_positions[idx] for idx in group_indices], dtype=int)
        for group_indices in groups
    ]

    actual = holdout["actual_total"].to_numpy(dtype=float)
    v3 = holdout["v3_raw_total"].to_numpy(dtype=float)
    v3_abs = np.abs(v3 - actual)
    v3_sq = (v3 - actual) ** 2

    rows = []
    for model_name, column in models_to_compare:
        candidate = holdout[column].to_numpy(dtype=float)
        candidate_abs = np.abs(candidate - actual)
        candidate_sq = (candidate - actual) ** 2

        delta_mae = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        delta_rmse = np.empty(BOOTSTRAP_REPLICATES, dtype=float)

        for b in range(BOOTSTRAP_REPLICATES):
            sampled_dates = rng.integers(0, len(date_positions), size=len(date_positions))
            sampled_rows = np.concatenate([date_positions[j] for j in sampled_dates])
            delta_mae[b] = (
                candidate_abs[sampled_rows].mean() - v3_abs[sampled_rows].mean()
            )
            delta_rmse[b] = (
                np.sqrt(candidate_sq[sampled_rows].mean())
                - np.sqrt(v3_sq[sampled_rows].mean())
            )

        point_mae = float(candidate_abs.mean() - v3_abs.mean())
        point_rmse = float(np.sqrt(candidate_sq.mean()) - np.sqrt(v3_sq.mean()))
        rows.append({
            "comparison": f"{model_name} minus V3",
            "holdout_games": int(len(holdout)),
            "holdout_dates": int(len(dates)),
            "delta_mae_candidate_minus_v3": point_mae,
            "delta_mae_95ci_low": float(np.quantile(delta_mae, 0.025)),
            "delta_mae_95ci_high": float(np.quantile(delta_mae, 0.975)),
            "bootstrap_probability_candidate_lower_mae": float(np.mean(delta_mae < 0)),
            "delta_rmse_candidate_minus_v3": point_rmse,
            "delta_rmse_95ci_low": float(np.quantile(delta_rmse, 0.025)),
            "delta_rmse_95ci_high": float(np.quantile(delta_rmse, 0.975)),
            "bootstrap_probability_candidate_lower_rmse": float(np.mean(delta_rmse < 0)),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_predictions()
    dev = df[df["period"].astype(str).str.casefold() == "development"].copy()
    holdout = df[df["period"].astype(str).str.casefold() == "holdout"].copy()
    if dev.empty or holdout.empty:
        raise ValueError("Both development and holdout periods are required.")

    summary = metric_table(df, ["period"])
    overall = metric_table(df)
    overall.insert(0, "period", "all")
    summary = pd.concat([overall, summary], ignore_index=True)

    # Add season metrics where the existing prediction file contains the key.
    if "season_start_year" in df.columns:
        seasonal = metric_table(df, ["season_start_year"])
        seasonal.insert(0, "period", "season")
        summary = pd.concat([summary, seasonal], ignore_index=True)

    monthly_df = df.copy()
    monthly_df["month"] = monthly_df["game_date"].dt.to_period("M").astype(str)
    monthly = metric_table(monthly_df, ["month"])
    monthly = monthly[monthly["games"] >= 20].copy()

    blocks = chronological_blocks(dev)
    bootstrap = cluster_bootstrap_holdout(holdout)

    summary.to_csv(SUMMARY_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    blocks.to_csv(BLOCKS_OUT, index=False)
    bootstrap.to_csv(BOOTSTRAP_OUT, index=False)

    print("\n===== RAW-TOTAL CANDIDATE ROBUSTNESS =====")
    print(f"Total games: {len(df)} | Development: {len(dev)} | Holdout: {len(holdout)}")
    print(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    print("\nMODEL SUMMARY BY PERIOD")
    print(summary.to_string(index=False))

    print("\nMONTHLY RESULTS (months with at least 20 games)")
    print(monthly.to_string(index=False))

    print("\nFOUR CONTIGUOUS DEVELOPMENT BLOCKS")
    print(blocks.to_string(index=False))

    print("\nPAIRED DATE-CLUSTER BOOTSTRAP: HOLDOUT DIFFERENCES VERSUS V3")
    print("Negative MAE/RMSE differences favour the comparison model.")
    print(bootstrap.to_string(index=False))

    print("\nSaved:")
    print(SUMMARY_OUT)
    print(MONTHLY_OUT)
    print(BLOCKS_OUT)
    print(BOOTSTRAP_OUT)
    print("\nThe bootstrap resamples whole game dates, preserving same-date clustering.")
    print("Research diagnostics only; no model or production prediction path was changed.")


if __name__ == "__main__":
    main()
