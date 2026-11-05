"""Out-of-time calibration investigation for V2 and V3 raw total predictions.

Run from the project root:
    python points_engine_v3_calibration.py

Reads outputs/points_engine_v3_comparison.csv only. It does not modify either
prediction engine or any production file.

The calibration check uses the first 70% of unique game dates to fit a simple
linear mapping (actual = intercept + slope * predicted), then evaluates that
mapping only on the chronologically later 30%.
"""
import os
import numpy as np
import pandas as pd

INPUT = "outputs/points_engine_v3_comparison.csv"
OUT_DIR = "outputs"
RANGE_OUT = os.path.join(OUT_DIR, "points_engine_v3_prediction_ranges.csv")
DECILE_OUT = os.path.join(OUT_DIR, "points_engine_v3_prediction_deciles.csv")
HOLDOUT_OUT = os.path.join(OUT_DIR, "points_engine_v3_calibration_holdout.csv")
HOLDOUT_BANDS_OUT = os.path.join(OUT_DIR, "points_engine_v3_calibration_holdout_bands.csv")

MODELS = {
    "V2 raw": "v2_raw_total",
    "V3 candidate": "v3_raw_total",
}


def regression_stats(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2 or np.std(predicted) == 0:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan,
                "calibration_r2": np.nan}
    slope, intercept = np.polyfit(predicted, actual, 1)
    fitted = intercept + slope * predicted
    ss_res = float(np.sum((actual - fitted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_r2": float(r2),
    }


def performance(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "games": int(len(error)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "bias_pred_minus_actual": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
    }


def load_comparison():
    if not os.path.isfile(INPUT):
        raise FileNotFoundError(
            f"Missing {INPUT}. Run totals_backtest_v3.py and "
            "points_engine_v3_robustness.py first."
        )
    df = pd.read_csv(INPUT)
    required = ["game_date", "actual_total", "v2_raw_total", "v3_raw_total"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {missing}")
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in ["actual_total", "v2_raw_total", "v3_raw_total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).sort_values("game_date").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} valid games; at least 100 are required.")
    return df


def make_prediction_range_tables(df):
    range_rows = []
    decile_rows = []
    fixed_bins = [-np.inf, 220.0, 240.0, np.inf]
    fixed_labels = ["predicted <220", "predicted 220-239.99", "predicted >=240"]

    for model, column in MODELS.items():
        pred = df[column].to_numpy(dtype=float)
        actual = df["actual_total"].to_numpy(dtype=float)
        stats = performance(actual, pred)
        reg = regression_stats(actual, pred)
        actual_sd = float(np.std(actual, ddof=1))
        pred_sd = float(np.std(pred, ddof=1))
        range_rows.append({
            "model": model,
            **stats,
            "predicted_sd": pred_sd,
            "actual_sd": actual_sd,
            "prediction_to_actual_sd_ratio": pred_sd / actual_sd if actual_sd else np.nan,
            "correlation": float(np.corrcoef(pred, actual)[0, 1]),
            **reg,
        })

        temp = pd.DataFrame({"predicted": pred, "actual": actual})
        temp["predicted_band"] = pd.cut(
            temp["predicted"], bins=fixed_bins, labels=fixed_labels, right=False
        )
        for band, group in temp.groupby("predicted_band", observed=False):
            if group.empty:
                continue
            row = {
                "model": model,
                "predicted_band": str(band),
                "predicted_min": float(group["predicted"].min()),
                "predicted_max": float(group["predicted"].max()),
                "actual_min": float(group["actual"].min()),
                "actual_max": float(group["actual"].max()),
            }
            row.update(performance(group["actual"], group["predicted"]))
            range_rows.append({"model": model, "row_type": "band", **row})

        # Equal-count prediction deciles expose range compression without
        # selecting cut-points after looking at the outcomes.
        temp["predicted_decile"] = (
            pd.qcut(temp["predicted"].rank(method="first"), q=10, labels=False) + 1
        )
        for decile, group in temp.groupby("predicted_decile", sort=True):
            decile_rows.append({
                "model": model,
                "predicted_decile": int(decile),
                "games": int(len(group)),
                "predicted_min": float(group["predicted"].min()),
                "predicted_max": float(group["predicted"].max()),
                "mean_predicted": float(group["predicted"].mean()),
                "mean_actual": float(group["actual"].mean()),
                "bias_pred_minus_actual": float((group["predicted"] - group["actual"]).mean()),
                "mae": float((group["predicted"] - group["actual"]).abs().mean()),
                "actual_sd": float(group["actual"].std(ddof=1)) if len(group) > 1 else np.nan,
            })

    ranges = pd.DataFrame(range_rows)
    # Add a clear row-type marker to overall summaries.
    if "row_type" not in ranges.columns:
        ranges["row_type"] = "overall"
    else:
        ranges["row_type"] = ranges["row_type"].fillna("overall")
    return ranges, pd.DataFrame(decile_rows)


def chronological_holdout(df):
    unique_dates = np.array(sorted(df["game_date"].dropna().unique()))
    split_index = int(len(unique_dates) * 0.70)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_index])
    train = df[df["game_date"] < split_date].copy()
    test = df[df["game_date"] >= split_date].copy()
    if len(train) < 50 or len(test) < 30:
        raise ValueError(
            f"Insufficient chronological split: train={len(train)}, test={len(test)}."
        )

    rows = []
    band_rows = []
    for model, column in MODELS.items():
        x_train = train[column].to_numpy(dtype=float)
        y_train = train["actual_total"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_train, y_train, 1)

        x_test = test[column].to_numpy(dtype=float)
        y_test = test["actual_total"].to_numpy(dtype=float)
        calibrated = intercept + slope * x_test

        base = performance(y_test, x_test)
        corrected = performance(y_test, calibrated)
        rows.append({
            "model": model,
            "train_games": int(len(train)),
            "holdout_games": int(len(test)),
            "train_start": train["game_date"].min().date().isoformat(),
            "train_end": train["game_date"].max().date().isoformat(),
            "holdout_start": test["game_date"].min().date().isoformat(),
            "holdout_end": test["game_date"].max().date().isoformat(),
            "calibration_intercept_fit_on_train": float(intercept),
            "calibration_slope_fit_on_train": float(slope),
            "uncalibrated_mae": base["mae"],
            "calibrated_mae": corrected["mae"],
            "mae_change_calibrated_minus_uncalibrated": corrected["mae"] - base["mae"],
            "uncalibrated_rmse": base["rmse"],
            "calibrated_rmse": corrected["rmse"],
            "rmse_change_calibrated_minus_uncalibrated": corrected["rmse"] - base["rmse"],
            "uncalibrated_bias_pred_minus_actual": base["bias_pred_minus_actual"],
            "calibrated_bias_pred_minus_actual": corrected["bias_pred_minus_actual"],
            "uncalibrated_within_5_pct": base["within_5_pct"],
            "calibrated_within_5_pct": corrected["within_5_pct"],
            "uncalibrated_within_10_pct": base["within_10_pct"],
            "calibrated_within_10_pct": corrected["within_10_pct"],
        })

        hold = pd.DataFrame({
            "actual": y_test,
            "uncalibrated": x_test,
            "calibrated": calibrated,
        })
        hold["predicted_band_before_calibration"] = pd.cut(
            hold["uncalibrated"],
            bins=[-np.inf, 220.0, 240.0, np.inf],
            labels=["predicted <220", "predicted 220-239.99", "predicted >=240"],
            right=False,
        )
        for band, group in hold.groupby("predicted_band_before_calibration", observed=False):
            if group.empty:
                continue
            for label, prediction_col in [
                ("uncalibrated", "uncalibrated"),
                ("calibrated", "calibrated"),
            ]:
                item = {
                    "model": model,
                    "predicted_band_before_calibration": str(band),
                    "prediction_version": label,
                }
                item.update(performance(group["actual"], group[prediction_col]))
                band_rows.append(item)

    return pd.DataFrame(rows), pd.DataFrame(band_rows), train, test, split_date


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_comparison()
    ranges, deciles = make_prediction_range_tables(df)
    holdout, holdout_bands, train, test, split_date = chronological_holdout(df)

    ranges.to_csv(RANGE_OUT, index=False)
    deciles.to_csv(DECILE_OUT, index=False)
    holdout.to_csv(HOLDOUT_OUT, index=False)
    holdout_bands.to_csv(HOLDOUT_BANDS_OUT, index=False)

    print("\n===== PREDICTED-TOTAL RANGE AND CALIBRATION =====")
    print(f"Games: {len(df)} | Dates: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print("\nOVERALL DISTRIBUTION AND CALIBRATION REGRESSION")
    print(ranges[ranges["row_type"] == "overall"].to_string(index=False))
    print("\nFIXED PREDICTED-TOTAL BANDS")
    print(ranges[ranges["row_type"] == "band"].to_string(index=False))
    print("\nPREDICTION DECILES (1=lowest predicted totals, 10=highest)")
    print(deciles.to_string(index=False))
    print("\nCHRONOLOGICAL CALIBRATION HOLDOUT")
    print(f"Calibration fit: {len(train)} games through {train['game_date'].max().date()}")
    print(f"Untouched later holdout: {len(test)} games from {split_date.date()} onward")
    print(holdout.to_string(index=False))
    print("\nHOLDOUT PERFORMANCE BY ORIGINAL PREDICTED-TOTAL BAND")
    print(holdout_bands.to_string(index=False))
    print("\nSaved:")
    print(RANGE_OUT)
    print(DECILE_OUT)
    print(HOLDOUT_OUT)
    print(HOLDOUT_BANDS_OUT)
    print("\nImportant: calibration coefficients are fitted only on the earlier dates. "
          "The later holdout is not used to fit them.")
    print("Research-only diagnostics. No production model or prediction path was changed.")


if __name__ == "__main__":
    main()
