"""Final validation of V3/V4 raw-total blend using multi-season predictions.

Run from the project root:
    python raw_total_multiseason_validation.py

Reads outputs/raw_total_multiseason_predictions.csv only. It:
1) searches V3 blend weights in 5% steps using development seasons only;
2) freezes the selected weight and reports later-season validation;
3) compares paired errors against V2, V3, V4-90 and V4-180;
4) uses season-stratified, game-date-cluster bootstrap intervals on the
   later-season validation period.

No engine is retrained, and no production files are changed.
"""
import os
import numpy as np
import pandas as pd

INPUT = "outputs/raw_total_multiseason_predictions.csv"
OUT_DIR = "outputs"
WEIGHT_GRID_OUT = os.path.join(OUT_DIR, "raw_total_blend_weight_grid.csv")
VALIDATION_OUT = os.path.join(OUT_DIR, "raw_total_blend_validation_summary.csv")
SEASON_OUT = os.path.join(OUT_DIR, "raw_total_blend_season_results.csv")
BOOTSTRAP_OUT = os.path.join(OUT_DIR, "raw_total_blend_paired_bootstrap.csv")

BASE_COLUMNS = {
    "V2 raw": "v2_raw_total",
    "V3 candidate": "v3_raw_total",
    "V4 90-day": "v4_90_total",
    "V4 180-day": "v4_180_total",
}
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20261031


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "games": int(len(error)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias_pred_minus_actual": float(np.mean(error)),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
    }


def load_data():
    if not os.path.isfile(INPUT):
        raise FileNotFoundError(
            f"Missing {INPUT}. Run totals_backtest_raw_multiseason.py first."
        )
    df = pd.read_csv(INPUT)
    required = [
        "game_date", "season_start_year", "period", "actual_total",
        *BASE_COLUMNS.values(),
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for col in ["season_start_year", "actual_total", *BASE_COLUMNS.values()]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["season_start_year"] = df["season_start_year"].astype(int)
    df["period"] = df["period"].astype(str).str.casefold()
    if df.empty:
        raise ValueError("No valid prediction rows.")
    return df.sort_values(["season_start_year", "game_date"]).reset_index(drop=True)


def add_blend(df, v3_weight):
    return (
        float(v3_weight) * df["v3_raw_total"].to_numpy(dtype=float)
        + (1.0 - float(v3_weight)) * df["v4_90_total"].to_numpy(dtype=float)
    )


def weight_grid(development, validation):
    rows = []
    for weight in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        for period_name, subset in [
            ("development", development),
            ("later_validation", validation),
        ]:
            prediction = add_blend(subset, weight)
            row = {
                "v3_weight": float(weight),
                "v4_90_weight": float(1.0 - weight),
                "period": period_name,
            }
            row.update(metrics(subset["actual_total"], prediction))
            rows.append(row)
    result = pd.DataFrame(rows)
    dev = result[result["period"] == "development"].sort_values(
        ["mae", "rmse", "v3_weight"]
    )
    selected_weight = float(dev.iloc[0]["v3_weight"])
    return result, selected_weight


def summary_rows(development, validation, selected_weight):
    rows = []
    periods = [
        ("development", development),
        ("later_validation", validation),
        ("all", pd.concat([development, validation], ignore_index=True)),
    ]
    for period_name, subset in periods:
        for model_name, column in BASE_COLUMNS.items():
            row = {"period": period_name, "model": model_name}
            row.update(metrics(subset["actual_total"], subset[column]))
            rows.append(row)
        blend = add_blend(subset, selected_weight)
        row = {
            "period": period_name,
            "model": f"Selected blend V3 {selected_weight:.0%} / V4-90 {1-selected_weight:.0%}",
        }
        row.update(metrics(subset["actual_total"], blend))
        rows.append(row)
    return pd.DataFrame(rows)


def season_rows(df, selected_weight):
    rows = []
    for season, group in df.groupby("season_start_year", sort=True):
        for model_name, column in BASE_COLUMNS.items():
            row = {"season_start_year": int(season), "model": model_name}
            row.update(metrics(group["actual_total"], group[column]))
            rows.append(row)
        row = {
            "season_start_year": int(season),
            "model": f"Selected blend V3 {selected_weight:.0%} / V4-90 {1-selected_weight:.0%}",
        }
        row.update(metrics(group["actual_total"], add_blend(group, selected_weight)))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(validation, selected_weight):
    """Resample game-date clusters independently within each validation season."""
    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = [
        ("V2 raw", "v2_raw_total"),
        ("V3 candidate", "v3_raw_total"),
        ("V4 90-day", "v4_90_total"),
        ("V4 180-day", "v4_180_total"),
    ]

    season_groups = []
    for season, season_df in validation.groupby("season_start_year", sort=True):
        date_groups = [
            group.index.to_numpy()
            for _, group in season_df.groupby("game_date", sort=True)
        ]
        if not date_groups:
            continue
        season_groups.append((int(season), date_groups))

    if len(season_groups) < 1:
        raise ValueError("No season/date clusters found for bootstrap.")

    # Convert labels to fixed positional row indexes for efficient resampling.
    index_to_position = {idx: pos for pos, idx in enumerate(validation.index)}
    positional_groups = []
    for season, date_groups in season_groups:
        positional_groups.append((
            season,
            [
                np.array([index_to_position[idx] for idx in group], dtype=int)
                for group in date_groups
            ],
        ))

    actual = validation["actual_total"].to_numpy(dtype=float)
    blend = add_blend(validation, selected_weight)
    candidate_arrays = {
        name: validation[column].to_numpy(dtype=float)
        for name, column in comparisons
    }
    candidate_arrays["selected blend"] = blend

    rows = []
    for comparison_name, candidate in candidate_arrays.items():
        if comparison_name == "selected blend":
            continue
        cand_abs = np.abs(candidate - actual)
        cand_sq = (candidate - actual) ** 2
        blend_abs = np.abs(blend - actual)
        blend_sq = (blend - actual) ** 2

        delta_mae = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        delta_rmse = np.empty(BOOTSTRAP_REPLICATES, dtype=float)

        for b in range(BOOTSTRAP_REPLICATES):
            sampled = []
            for _, date_groups in positional_groups:
                picks = rng.integers(0, len(date_groups), size=len(date_groups))
                sampled.extend(date_groups[pick] for pick in picks)
            positions = np.concatenate(sampled)

            # Positive values mean selected blend has lower error.
            delta_mae[b] = cand_abs[positions].mean() - blend_abs[positions].mean()
            delta_rmse[b] = (
                np.sqrt(cand_sq[positions].mean())
                - np.sqrt(blend_sq[positions].mean())
            )

        point_mae = float(cand_abs.mean() - blend_abs.mean())
        point_rmse = float(np.sqrt(cand_sq.mean()) - np.sqrt(blend_sq.mean()))
        rows.append({
            "comparison": f"{comparison_name} minus selected blend",
            "validation_games": int(len(validation)),
            "validation_dates": int(validation["game_date"].nunique()),
            "delta_mae_comparator_minus_blend": point_mae,
            "delta_mae_95ci_low": float(np.quantile(delta_mae, 0.025)),
            "delta_mae_95ci_high": float(np.quantile(delta_mae, 0.975)),
            "probability_blend_lower_mae": float(np.mean(delta_mae > 0)),
            "delta_rmse_comparator_minus_blend": point_rmse,
            "delta_rmse_95ci_low": float(np.quantile(delta_rmse, 0.025)),
            "delta_rmse_95ci_high": float(np.quantile(delta_rmse, 0.975)),
            "probability_blend_lower_rmse": float(np.mean(delta_rmse > 0)),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    development = df[df["period"] == "development"].copy()
    validation = df[df["period"] == "later_seasons"].copy()
    if development.empty or validation.empty:
        raise ValueError("Both development and later_seasons periods are required.")

    grid, selected_weight = weight_grid(development, validation)
    summary = summary_rows(development, validation, selected_weight)
    seasons = season_rows(df, selected_weight)
    bootstrap = paired_cluster_bootstrap(validation, selected_weight)

    grid.to_csv(WEIGHT_GRID_OUT, index=False)
    summary.to_csv(VALIDATION_OUT, index=False)
    seasons.to_csv(SEASON_OUT, index=False)
    bootstrap.to_csv(BOOTSTRAP_OUT, index=False)

    print("\n===== RAW-TOTAL BLEND FINAL VALIDATION =====")
    print(f"Development games: {len(development):,}")
    print(f"Later validation games: {len(validation):,}")
    print(
        f"Development dates: {development['game_date'].min().date()} to "
        f"{development['game_date'].max().date()}"
    )
    print(
        f"Validation dates: {validation['game_date'].min().date()} to "
        f"{validation['game_date'].max().date()}"
    )
    print("\nBLEND WEIGHT GRID (weight selected on development only)")
    print(grid[grid["period"] == "development"].sort_values(
        ["mae", "rmse"]
    ).to_string(index=False))
    print(f"\nSelected V3 weight: {selected_weight:.0%}")
    print(f"Selected V4-90 weight: {1-selected_weight:.0%}")

    print("\nMODEL PERFORMANCE")
    print(summary.to_string(index=False))

    print("\nSEASON-BY-SEASON PERFORMANCE")
    print(seasons.to_string(index=False))

    print("\nPAIRED DATE-CLUSTER BOOTSTRAP: COMPARATOR MINUS SELECTED BLEND")
    print("Positive differences favour the selected blend.")
    print(bootstrap.to_string(index=False))

    print("\nSaved:")
    print(WEIGHT_GRID_OUT)
    print(VALIDATION_OUT)
    print(SEASON_OUT)
    print(BOOTSTRAP_OUT)
    print(
        "\nBootstrap resamples whole game dates separately within each "
        "validation season."
    )
    print(
        "The later validation seasons were not used to select the blend weight. "
        "Research-only; no production model was changed."
    )


if __name__ == "__main__":
    main()
