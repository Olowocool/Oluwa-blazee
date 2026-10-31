"""Robustness diagnostics for the existing V2/V3 raw-total comparison.

Run from the project root:
    python points_engine_v3_robustness.py

Reads outputs/points_engine_v3_comparison.csv only. It does not retrain,
recalculate predictions, or modify production code.
"""
import os
import numpy as np
import pandas as pd

INPUT = "outputs/points_engine_v3_comparison.csv"
OUT_DIR = "outputs"
MONTHLY_OUT = os.path.join(OUT_DIR, "points_engine_v3_monthly.csv")
TEAM_OUT = os.path.join(OUT_DIR, "points_engine_v3_team_splits.csv")
BANDS_OUT = os.path.join(OUT_DIR, "points_engine_v3_total_bands.csv")


def metric_row(group, model, error_col):
    e = pd.to_numeric(group[error_col], errors="coerce").dropna().to_numpy(dtype=float)
    if not len(e):
        return None
    ae = np.abs(e)
    return {
        "model": model,
        "games": int(len(e)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(e.mean()),
        "within_5_pct": float((ae <= 5).mean() * 100),
        "within_10_pct": float((ae <= 10).mean() * 100),
    }


def build_group_table(df, group_col, scope_name, min_games=1):
    rows = []
    for key, group in df.groupby(group_col, dropna=False):
        if len(group) < min_games:
            continue
        for model, error_col in [("V2 raw", "v2_error"), ("V3 candidate", "v3_error")]:
            row = metric_row(group, model, error_col)
            if row:
                row[scope_name] = key
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    if not os.path.isfile(INPUT):
        raise FileNotFoundError(
            f"Missing {INPUT}. Run totals_backtest_v3.py first."
        )
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT)
    required = [
        "game_date", "home_team", "away_team", "actual_total",
        "v2_raw_total", "v3_raw_total", "v2_error", "v3_error",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Comparison file is missing columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for col in ["actual_total", "v2_raw_total", "v3_raw_total", "v2_error", "v3_error"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["game_date", "actual_total", "v2_error", "v3_error"])
    if df.empty:
        raise ValueError("No valid comparison rows found.")

    # Calendar-month stability.
    df["month"] = df["game_date"].dt.to_period("M").astype(str)
    monthly = build_group_table(df, "month", "month", min_games=20)
    monthly.to_csv(MONTHLY_OUT, index=False)

    # Error performance when a team appears as the home or away team.
    home = build_group_table(df, "home_team", "team", min_games=20)
    if not home.empty:
        home.insert(0, "role", "home")
    away = build_group_table(df, "away_team", "team", min_games=20)
    if not away.empty:
        away.insert(0, "role", "away")
    teams = pd.concat([home, away], ignore_index=True)
    if not teams.empty:
        teams.to_csv(TEAM_OUT, index=False)
    else:
        pd.DataFrame(columns=["role", "team", "model", "games", "mae", "rmse", "bias",
                              "within_5_pct", "within_10_pct"]).to_csv(TEAM_OUT, index=False)

    # Fixed actual-total bands, chosen before inspecting the split results.
    bins = [-np.inf, 219.999, 239.999, np.inf]
    labels = ["low (<220)", "middle (220-239)", "high (240+)"]
    df["actual_total_band"] = pd.cut(df["actual_total"], bins=bins, labels=labels)
    bands = build_group_table(df, "actual_total_band", "actual_total_band")
    bands.to_csv(BANDS_OUT, index=False)

    print("\n===== V3 RAW-TOTAL ROBUSTNESS CHECK =====")
    print(f"Valid games: {len(df)}")
    print(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print("\nMONTHLY (months with at least 20 games)")
    print(monthly.to_string(index=False) if not monthly.empty else "No qualifying months.")
    print("\nTEAM SPLITS (team-role combinations with at least 20 games)")
    print(teams.to_string(index=False) if not teams.empty else "No qualifying team splits.")
    print("\nACTUAL TOTAL BANDS")
    print(bands.to_string(index=False) if not bands.empty else "No band results.")
    print("\nSaved:")
    print(MONTHLY_OUT)
    print(TEAM_OUT)
    print(BANDS_OUT)
    print("\nInterpretation note: team splits group total-prediction errors by the home-team "
          "or away-team identity; they do not measure team-point prediction accuracy.")
    print("Research diagnostics only. No production files or predictions were changed.")


if __name__ == "__main__":
    main()
