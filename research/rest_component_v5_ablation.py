#!/usr/bin/env python3
"""Point-in-time V5 rest and schedule-density ablation (research only).

The current production rule is reproduced exactly:
    0.50 * (home_rest_days + away_rest_days - 2)
where rest days are calendar days since the last completed game minus one,
clipped to [0, 3]. Rest history is staged by date, so no same-day result can
leak into another prediction on that date.

Development: 2021-2023. Later validation: 2024-2025.
No production prediction path is modified.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd

from points_engine_v5 import calculate_matchup_points_v5

BASE = "https://basketball-data.billmill.org"
FIRST_SEASON, LAST_SEASON = 2010, 2026
TARGET_START_YEARS = {2021, 2022, 2023, 2024, 2025}
DEV_YEARS = {2021, 2022, 2023}
BOOTSTRAPS = 5000
SEED = 20261031
OUT = ROOT / "outputs/research/rest_component_v5"


def load_source():
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as response:
        meta = json.loads(response.read().decode())["datasets"]
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))
    for table in ("team_game_logs", "games"):
        info = meta[table]
        if int(info["first_season"]) > FIRST_SEASON or int(info["last_season"]) < LAST_SEASON:
            raise RuntimeError(f"{table} metadata does not cover required seasons: {info}")
        urls = ", ".join(
            "'" + f"{BASE}/nba/stats/{table}/season={season}/data.parquet" + "'"
            for season in seasons
        )
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
    raw = con.execute("""
        SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
               t.home, t.pts, g.game_type, g.neutral_site
        FROM team_game_logs t
        JOIN games g USING (season, game_id)
        WHERE t.season BETWEEN ? AND ? AND g.game_type = 'regular_season'
        ORDER BY t.game_date, t.season, t.game_id, t.team_id
    """, [FIRST_SEASON, LAST_SEASON]).fetchdf()
    con.close()
    return raw, meta


def prepare_source(raw):
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["pts"] = pd.to_numeric(d["pts"], errors="coerce")
    d["team_id"] = d["team_id"].astype(str)
    d["opp_team_id"] = d["opp_team_id"].astype(str)
    d = d.dropna(subset=["game_date", "season", "pts", "home"])
    d = d[d["pts"] > 0]
    d = d[~d["neutral_site"].fillna(False).astype(bool)]
    if d.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate (season, game_id, team_id) team rows.")

    by_date = defaultdict(list)
    score_rows = []
    for (season, game_id), group in d.groupby(["season", "game_id"], sort=False):
        if len(group) != 2 or group["team_id"].nunique() != 2:
            raise RuntimeError(f"Invalid team-game pairing for game {game_id}.")
        records = group.to_dict("records")
        homes = [row for row in records if bool(row["home"])]
        aways = [row for row in records if not bool(row["home"])]
        if len(homes) != 1 or len(aways) != 1:
            raise RuntimeError(f"Invalid home/away flags for game {game_id}.")
        home, away = homes[0], aways[0]
        if home["opp_team_id"] != away["team_id"] or away["opp_team_id"] != home["team_id"]:
            raise RuntimeError(f"Broken opponent link for game {game_id}.")
        date = pd.Timestamp(home["game_date"]).normalize()
        season = int(season)
        home_id, away_id = str(home["team_id"]), str(away["team_id"])
        home_pts, away_pts = float(home["pts"]), float(away["pts"])
        total = home_pts + away_pts
        gid = str(game_id)
        by_date[date].append(
            (season, gid, home_id, away_id, home_pts, away_pts, total)
        )
        score_rows.append({
            "game_date": date, "home_team": home_id, "away_team": away_id,
            "home_score": home_pts, "away_score": away_pts, "total_score": total,
        })

    scores = pd.DataFrame(score_rows).sort_values("game_date").reset_index(drop=True)
    return d, by_date, scores


def team_schedule_features(team_id, target_date, team_dates):
    """Features from completed games strictly before target_date only."""
    prior = team_dates.get(team_id, [])
    if not prior:
        # Mirrors calculate_team_rest_days()'s no-history fallback.
        return {
            "rest_days": 1, "last_game_date": None,
            "games_prev_3d": 0, "games_prev_5d": 0, "games_prev_7d": 0,
            "games_prev_14d": 0,
        }
    latest = prior[-1]
    if not latest < target_date:
        raise RuntimeError("Rest feature contains a same-day or future team game.")
    rest = max(0, min(3, int((target_date - latest).days) - 1))
    def count_days(n):
        cutoff = target_date - pd.Timedelta(days=n)
        return sum(cutoff <= day < target_date for day in prior)
    return {
        "rest_days": rest,
        "last_game_date": latest,
        "games_prev_3d": count_days(3),
        "games_prev_5d": count_days(5),
        "games_prev_7d": count_days(7),
        "games_prev_14d": count_days(14),
    }


def metrics(frame, column):
    err = frame[column].to_numpy(dtype=float) - frame["actual_total"].to_numpy(dtype=float)
    absolute = np.abs(err)
    return {
        "games": int(len(err)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "within_5_pct": float((absolute <= 5).mean() * 100),
        "within_10_pct": float((absolute <= 10).mean() * 100),
    }


def fit_scalar_beta(frame, signal):
    x = frame[signal].to_numpy(dtype=float)
    y = (frame["actual_total"] - frame["v5_raw_total"]).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    denom = float(x[valid] @ x[valid])
    return float(x[valid] @ y[valid] / denom) if denom > 1e-12 else 0.0


def fit_ridge(frame, feature_cols, alpha):
    x = frame[feature_cols].to_numpy(dtype=float)
    y = (frame["actual_total"] - frame["v5_raw_total"]).to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite values in development rest features.")
    gram = x.T @ x
    penalty = float(alpha) * np.eye(x.shape[1])
    coef = np.linalg.solve(gram + penalty, x.T @ y)
    return coef


def paired_date_bootstrap(frame, candidate_col, seed):
    rng = np.random.default_rng(seed)
    date_groups = [
        group.index.to_numpy()
        for _, group in frame.groupby("game_date", sort=True)
    ]
    positions = {index: pos for pos, index in enumerate(frame.index)}
    groups = [
        np.asarray([positions[index] for index in group], dtype=int)
        for group in date_groups
    ]
    actual = frame["actual_total"].to_numpy(dtype=float)
    baseline = frame["v5_raw_total"].to_numpy(dtype=float)
    candidate = frame[candidate_col].to_numpy(dtype=float)
    base_abs = np.abs(baseline - actual)
    cand_abs = np.abs(candidate - actual)
    base_sq = (baseline - actual) ** 2
    cand_sq = (candidate - actual) ** 2
    delta_mae = np.empty(BOOTSTRAPS)
    delta_rmse = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[pick] for pick in picks])
        delta_mae[i] = cand_abs[sample].mean() - base_abs[sample].mean()
        delta_rmse[i] = np.sqrt(cand_sq[sample].mean()) - np.sqrt(base_sq[sample].mean())
    return {
        "mae_difference_candidate_minus_v5": float(cand_abs.mean() - base_abs.mean()),
        "mae_95ci": [float(np.quantile(delta_mae, .025)), float(np.quantile(delta_mae, .975))],
        "probability_candidate_lower_mae": float(np.mean(delta_mae < 0)),
        "rmse_difference_candidate_minus_v5": float(
            np.sqrt(cand_sq.mean()) - np.sqrt(base_sq.mean())
        ),
        "rmse_95ci": [float(np.quantile(delta_rmse, .025)), float(np.quantile(delta_rmse, .975))],
        "probability_candidate_lower_rmse": float(np.mean(delta_rmse < 0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, metadata = load_source()
    d, by_date, score_history = prepare_source(raw)
    print(
        f"Loaded {len(d):,} regular-season non-neutral team-game rows; "
        f"{len(score_history):,} games."
    )

    team_dates = defaultdict(list)
    cache = {"v3": {}, "v4": {}}
    rows = []
    target_dates = sorted(by_date)

    for date_index, date in enumerate(target_dates, start=1):
        todays = by_date[date]
        cache["v3"].clear()
        for key in list(cache["v4"].keys()):
            if key != "__v4_clean_history__":
                del cache["v4"][key]

        pending_team_updates = []
        for season, game_id, home_id, away_id, home_score, away_score, actual_total in todays:
            season_start = season - 1
            if season_start in TARGET_START_YEARS:
                home_rest = team_schedule_features(home_id, date, team_dates)
                away_rest = team_schedule_features(away_id, date, team_dates)

                v5 = calculate_matchup_points_v5(
                    home_id, away_id, score_history, game_date=date,
                    _prepared_cache=cache,
                )
                raw_total = float(v5["raw_total"])

                h_prev3 = int(home_rest["games_prev_3d"])
                a_prev3 = int(away_rest["games_prev_3d"])
                h_prev5 = int(home_rest["games_prev_5d"])
                a_prev5 = int(away_rest["games_prev_5d"])
                h_prev7 = int(home_rest["games_prev_7d"])
                a_prev7 = int(away_rest["games_prev_7d"])

                row = {
                    "game_date": date,
                    "game_id": game_id,
                    "season_start_year": season_start,
                    "period": "development" if season_start in DEV_YEARS else "holdout",
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "actual_total": actual_total,
                    "v5_raw_total": raw_total,
                    "home_last_game_date": (
                        home_rest["last_game_date"].date().isoformat()
                        if home_rest["last_game_date"] is not None else None
                    ),
                    "away_last_game_date": (
                        away_rest["last_game_date"].date().isoformat()
                        if away_rest["last_game_date"] is not None else None
                    ),
                    "home_rest_days": int(home_rest["rest_days"]),
                    "away_rest_days": int(away_rest["rest_days"]),
                    "home_games_prev_3d": h_prev3,
                    "away_games_prev_3d": a_prev3,
                    "home_games_prev_5d": h_prev5,
                    "away_games_prev_5d": a_prev5,
                    "home_games_prev_7d": h_prev7,
                    "away_games_prev_7d": a_prev7,
                    "home_games_prev_14d": int(home_rest["games_prev_14d"]),
                    "away_games_prev_14d": int(away_rest["games_prev_14d"]),
                    "home_back_to_back": int(home_rest["rest_days"] == 0),
                    "away_back_to_back": int(away_rest["rest_days"] == 0),
                    # Target game would make a 3rd game in four calendar days,
                    # or a 4th in six, when these prior-game counts cross limits.
                    "home_3in4": int(h_prev3 >= 2),
                    "away_3in4": int(a_prev3 >= 2),
                    "home_4in6": int(h_prev5 >= 3),
                    "away_4in6": int(a_prev5 >= 3),
                }
                rows.append(row)

                for team_id, feature in ((home_id, home_rest), (away_id, away_rest)):
                    last_date = feature["last_game_date"]
                    if last_date is not None and not last_date < date:
                        raise RuntimeError("Rest cutoff audit failed.")
            pending_team_updates.extend([(home_id, date), (away_id, date)])

        # Stage all games for the date only after every same-date prediction.
        seen_today = set()
        for team_id, game_date in pending_team_updates:
            if team_id in seen_today:
                raise RuntimeError(f"Team {team_id} appears in multiple regular-season games on {date}.")
            seen_today.add(team_id)
            team_dates[team_id].append(game_date)

        if date_index % 100 == 0 or date_index == len(target_dates):
            print(f"Processed {date_index}/{len(target_dates)} dates; target predictions={len(rows)}")

    features = pd.DataFrame(rows)
    if len(features) != 6140:
        raise RuntimeError(f"Expected 6,140 target predictions, produced {len(features)}.")
    if features.duplicated(["game_date", "game_id"]).any():
        raise RuntimeError("Duplicate target-game predictions.")

    for col in ("home_last_game_date", "away_last_game_date"):
        prior = pd.to_datetime(features[col], errors="coerce")
        target = pd.to_datetime(features["game_date"])
        if bool((prior.notna() & (prior >= target)).any()):
            raise RuntimeError(f"Same-day or future history found in {col}.")

    dev = features[features["period"] == "development"].copy()
    hold = features[features["period"] == "holdout"].copy()
    if len(dev) != 3690 or len(hold) != 2450:
        raise RuntimeError(f"Unexpected date split: development={len(dev)}, holdout={len(hold)}.")

    # Reproduce the live 0.50 rule exactly.
    features["rest_legacy_fixed_050"] = features["v5_raw_total"] + (
        0.50 * (features["home_rest_days"] + features["away_rest_days"] - 2)
    )

    features["combined_rest_delta"] = (
        features["home_rest_days"] + features["away_rest_days"] - 2
    )
    beta = fit_scalar_beta(dev, "combined_rest_delta")
    features["rest_learned_shared_beta"] = (
        features["v5_raw_total"] + beta * features["combined_rest_delta"]
    )

    # Separate home/away rest coefficients; all coefficients are fit on 2021-23 only.
    features["home_rest_delta"] = features["home_rest_days"] - 1
    features["away_rest_delta"] = features["away_rest_days"] - 1
    rest_features = ["home_rest_delta", "away_rest_delta"]
    candidates = {
        "V5 raw": "v5_raw_total",
        "Legacy fixed 0.50 combined rest": "rest_legacy_fixed_050",
        f"Learned shared rest beta={beta:.4f}": "rest_learned_shared_beta",
    }
    coefficients = [{
        "candidate": "learned_shared_rest",
        "feature": "home_rest_delta + away_rest_delta",
        "coefficient": beta,
        "ridge_alpha": 0,
    }]

    for alpha in (0.0, 100.0, 500.0):
        coef = fit_ridge(dev, rest_features, alpha)
        col = f"rest_separate_ridge_{int(alpha)}"
        features[col] = features["v5_raw_total"] + (
            features[rest_features].to_numpy(dtype=float) @ coef
        )
        candidates[f"Separate home/away rest ridge alpha={int(alpha)}"] = col
        for name, value in zip(rest_features, coef):
            coefficients.append({
                "candidate": col, "feature": name,
                "coefficient": float(value), "ridge_alpha": alpha,
            })

    # Schedule-density candidate: rest-day deltas plus 3-in-4 and 4-in-6 flags.
    schedule_features = [
        "home_rest_delta", "away_rest_delta",
        "home_3in4", "away_3in4", "home_4in6", "away_4in6",
    ]
    for alpha in (100.0, 1000.0):
        coef = fit_ridge(dev, schedule_features, alpha)
        col = f"rest_schedule_ridge_{int(alpha)}"
        features[col] = features["v5_raw_total"] + (
            features[schedule_features].to_numpy(dtype=float) @ coef
        )
        candidates[f"Rest plus schedule-density ridge alpha={int(alpha)}"] = col
        for name, value in zip(schedule_features, coef):
            coefficients.append({
                "candidate": col, "feature": name,
                "coefficient": float(value), "ridge_alpha": alpha,
            })

    comparison_rows = []
    for period in ("development", "holdout", "all"):
        part = features if period == "all" else features[features["period"] == period]
        for name, column in candidates.items():
            comparison_rows.append({
                "period": period, "model": name, **metrics(part, column)
            })
    for season, part in features.groupby("season_start_year", sort=True):
        for name, column in candidates.items():
            comparison_rows.append({
                "period": "season", "season_start_year": int(season),
                "model": name, **metrics(part, column),
            })
    comparison = pd.DataFrame(comparison_rows)

    dev_comparison = comparison[
        comparison["period"] == "development"
    ].sort_values(["mae", "rmse"])
    selected = dev_comparison.iloc[0]
    selected_name = str(selected["model"])
    selected_col = candidates[selected_name]
    hold_row = comparison[
        (comparison["period"] == "holdout")
        & (comparison["model"] == selected_name)
    ].iloc[0]

    legacy_bootstrap = paired_date_bootstrap(
        hold.reset_index(drop=True).assign(
            rest_legacy_fixed_050=features.loc[hold.index, "rest_legacy_fixed_050"].to_numpy()
        ),
        "rest_legacy_fixed_050", SEED + 1,
    )
    selected_bootstrap = paired_date_bootstrap(
        hold.reset_index(drop=True).assign(
            **{selected_col: features.loc[hold.index, selected_col].to_numpy()}
        ),
        selected_col, SEED + 2,
    )

    rest_distribution = {}
    for team_side in ("home", "away"):
        vals = features[f"{team_side}_rest_days"]
        rest_distribution[team_side] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "counts_0_1_2_3": {
                str(i): int((vals == i).sum()) for i in range(4)
            },
            "back_to_back_games": int((vals == 0).sum()),
            "three_in_four_games": int(features[f"{team_side}_3in4"].sum()),
            "four_in_six_games": int(features[f"{team_side}_4in6"].sum()),
        }

    latest_history = pd.concat([
        pd.to_datetime(features["home_last_game_date"], errors="coerce"),
        pd.to_datetime(features["away_last_game_date"], errors="coerce"),
    ]).dropna()
    target_dates_for_audit = pd.concat([
        pd.to_datetime(features["game_date"]),
        pd.to_datetime(features["game_date"]),
    ]).reset_index(drop=True)
    if len(latest_history) != len(target_dates_for_audit.dropna()):
        # Two prior dates are expected for every prediction in this sample.
        raise RuntimeError("Some target team schedules have no prior game; investigate before scoring.")
    cutoff_ok = bool(
        (pd.to_datetime(features["home_last_game_date"]) < pd.to_datetime(features["game_date"])).all()
        and (pd.to_datetime(features["away_last_game_date"]) < pd.to_datetime(features["game_date"])).all()
    )
    if not cutoff_ok:
        raise RuntimeError("Point-in-time rest cutoff audit failed.")

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NBA Stats team_game_logs; llimllib/nba_data v2",
        "team_game_rows": int(len(d)),
        "regular_season_non_neutral_games": int(len(score_history)),
        "target_games": int(len(features)),
        "development_games": int(len(dev)),
        "holdout_games": int(len(hold)),
        "target_date_start": str(pd.Timestamp(features.game_date.min()).date()),
        "target_date_end": str(pd.Timestamp(features.game_date.max()).date()),
        "same_day_results_staged_until_after_predictions": True,
        "strict_prior_date_rest_cutoff_passed": cutoff_ok,
        "rest_definition": "clip((target_date - latest_prior_team_game_date).days - 1, 0, 3)",
        "no_prior_history_fallback": 1,
        "rest_distribution": rest_distribution,
        "legacy_rule": "raw_total + 0.50 * (home_rest_days + away_rest_days - 2)",
        "development_fitted_shared_rest_beta": beta,
        "coefficients_fit_on_development_only": coefficients,
        "selected_on_development_only": selected_name,
        "selected_holdout_metrics": hold_row.to_dict(),
        "legacy_fixed_050_bootstrap_vs_v5": legacy_bootstrap,
        "selected_candidate_bootstrap_vs_v5": selected_bootstrap,
        "source_updated_utc": {
            name: metadata[name]["updated"]
            for name in ("team_game_logs", "games")
        },
        "production_modified": False,
    }

    features.to_csv(OUT / "rest_component_v5_predictions.csv", index=False)
    comparison.to_csv(OUT / "rest_component_v5_comparison.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "rest_component_v5_coefficients.csv", index=False)
    (OUT / "rest_component_v5_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\n===== V5 REST COMPONENT ABLATION =====")
    print(f"Development games: {len(dev)} | Later validation games: {len(hold)}")
    print("\nREST DISTRIBUTION")
    print(json.dumps(rest_distribution, indent=2))
    print("\nTOTAL PREDICTION RESULTS")
    print(comparison.to_string(index=False))
    print("\nDEVELOPMENT-FITTED COEFFICIENTS")
    print(pd.DataFrame(coefficients).to_string(index=False))
    print(f"\nSelected on development only: {selected_name}")
    print(
        f"Later validation: MAE={hold_row.mae:.4f}, "
        f"RMSE={hold_row.rmse:.4f}, bias={hold_row.bias:.4f}"
    )
    print("\nLegacy fixed 0.50 rule: paired date-cluster bootstrap vs V5")
    print(json.dumps(legacy_bootstrap, indent=2))
    print("\nDevelopment-selected candidate: paired date-cluster bootstrap vs V5")
    print(json.dumps(selected_bootstrap, indent=2))
    print(f"\nOutputs: {OUT}")
    print("Rest remains research-only; no production integration.")


if __name__ == "__main__":
    main()
