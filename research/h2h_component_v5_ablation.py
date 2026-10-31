#!/usr/bin/env python3
"""Point-in-time V5 H2H adjustment ablation (research only).

Compares the existing fixed 15% H2H rule with smaller, development-fitted
corrections using 2021-2023 for development and 2024-2025 as later validation.
All target-date features use meetings and league results strictly before date.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

from points_engine_v5 import calculate_matchup_points_v5

BASE = "https://basketball-data.billmill.org"
FIRST_SEASON, LAST_SEASON = 2010, 2026
TARGET_START_YEARS = [2021, 2022, 2023, 2024, 2025]
DEV_YEARS = {2021, 2022, 2023}
WINDOWS = (5, 10)
SHRINK_K = 5.0
BOOTSTRAPS = 5000
SEED = 20261031
OUT = Path("outputs/research/h2h_component_v5")


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

    dates = defaultdict(list)
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
        total = float(home["pts"]) + float(away["pts"])
        gid = str(game_id)
        home_id, away_id = str(home["team_id"]), str(away["team_id"])
        dates[date].append((season, gid, home_id, away_id, float(home["pts"]), float(away["pts"]), total))
        score_rows.append({
            "game_date": date, "home_team": home_id, "away_team": away_id,
            "home_score": float(home["pts"]), "away_score": float(away["pts"]),
            "total_score": total,
        })
    scores = pd.DataFrame(score_rows).sort_values("game_date").reset_index(drop=True)
    return d, dates, scores


def fit_beta(frame, signal_col):
    x = frame[signal_col].to_numpy(dtype=float)
    y = (frame["actual_total"] - frame["v5_raw_total"]).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    denom = float(x @ x)
    return float(x @ y / denom) if denom > 1e-12 else 0.0


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


def paired_date_bootstrap(frame, candidate_col):
    rng = np.random.default_rng(SEED)
    groups = [g.index.to_numpy() for _, g in frame.groupby("game_date", sort=True)]
    positions = {index: pos for pos, index in enumerate(frame.index)}
    groups = [np.asarray([positions[index] for index in group]) for group in groups]
    actual = frame["actual_total"].to_numpy(dtype=float)
    baseline = frame["v5_raw_total"].to_numpy(dtype=float)
    candidate = frame[candidate_col].to_numpy(dtype=float)
    base_abs, cand_abs = np.abs(baseline - actual), np.abs(candidate - actual)
    base_sq, cand_sq = (baseline - actual) ** 2, (candidate - actual) ** 2
    delta_mae, delta_rmse = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[pick] for pick in picks])
        delta_mae[i] = cand_abs[sample].mean() - base_abs[sample].mean()
        delta_rmse[i] = np.sqrt(cand_sq[sample].mean()) - np.sqrt(base_sq[sample].mean())
    return {
        "mae_difference_candidate_minus_v5": float(cand_abs.mean() - base_abs.mean()),
        "mae_95ci": [float(np.quantile(delta_mae, .025)), float(np.quantile(delta_mae, .975))],
        "probability_candidate_lower_mae": float(np.mean(delta_mae < 0)),
        "rmse_difference_candidate_minus_v5": float(np.sqrt(cand_sq.mean()) - np.sqrt(base_sq.mean())),
        "rmse_95ci": [float(np.quantile(delta_rmse, .025)), float(np.quantile(delta_rmse, .975))],
        "probability_candidate_lower_rmse": float(np.mean(delta_rmse < 0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, metadata = load_source()
    d, dates, score_history = prepare_source(raw)
    print(f"Loaded {len(d):,} regular-season team-game rows; {len(score_history):,} games.")

    pair_history = defaultdict(list)
    season_totals = defaultdict(list)
    full_scores = score_history.copy()
    cache = {"v3": {}, "v4": {}}
    rows = []
    target_dates = sorted(dates)

    for date_index, date in enumerate(target_dates, start=1):
        todays = dates[date]
        # V3 fits are date-specific. Preserve only V4's immutable cleaned history.
        cache["v3"].clear()
        for key in list(cache["v4"].keys()):
            if key != "__v4_clean_history__":
                del cache["v4"][key]

        pending_pair_updates = []
        pending_season_updates = []
        for season, game_id, home_id, away_id, home_score, away_score, actual_total in todays:
            season_start = season - 1
            if season_start not in TARGET_START_YEARS:
                pass
            else:
                pair_key = tuple(sorted((home_id, away_id)))
                meetings = pair_history.get(pair_key, [])
                target_lg = season_totals.get(season, [])
                current_league_avg = float(np.mean(target_lg)) if target_lg else 225.0

                v5 = calculate_matchup_points_v5(
                    home_id, away_id, full_scores, game_date=date,
                    _prepared_cache=cache,
                )
                raw_total = float(v5["raw_total"])

                # Same H2H definition as the old engine: any prior regular-season
                # meeting between the teams, regardless of home/away order.
                # It is limited to strictly prior dates by staging updates by date.
                recent = meetings[-10:]
                n10 = len(recent)
                recent_total_mean = float(np.mean([item["total"] for item in recent])) if recent else raw_total
                legacy_signal = recent_total_mean - raw_total

                # Era adjustment: put each meeting's total onto the target date's
                # league scoring environment using the target-date-known average
                # for that meeting's season. This removes across-season scoring shifts.
                era_values = []
                for meeting in recent:
                    historical_season_avg_values = season_totals.get(meeting["season"], [])
                    historical_season_avg = (
                        float(np.mean(historical_season_avg_values))
                        if historical_season_avg_values else current_league_avg
                    )
                    era_values.append(meeting["total"] + current_league_avg - historical_season_avg)

                max_h2h_date = recent[-1]["date"] if recent else None
                if max_h2h_date is not None and not max_h2h_date < date:
                    raise RuntimeError("H2H history contains same-day or future meetings.")

                record = {
                    "game_date": date,
                    "h2h_max_history_date": (
                        max_h2h_date.date().isoformat() if max_h2h_date is not None else None
                    ),
                    "game_id": game_id,
                    "season_start_year": season_start,
                    "period": "development" if season_start in DEV_YEARS else "holdout",
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "actual_total": actual_total,
                    "v5_raw_total": raw_total,
                    "h2h_games_last10": n10,
                    "h2h_avg_total_last10": recent_total_mean,
                    "h2h_legacy_signal": legacy_signal,
                    "current_league_avg_total": current_league_avg,
                    "h2h_avg_era_adjusted_last10": float(np.mean(era_values)) if era_values else raw_total,
                    "h2h_era_signal_last10": (
                        float(np.mean(era_values)) - raw_total if era_values else 0.0
                    ),
                }

                for window in WINDOWS:
                    window_games = meetings[-window:]
                    count = len(window_games)
                    mean_total = (
                        float(np.mean([item["total"] for item in window_games]))
                        if window_games else raw_total
                    )
                    normalized = []
                    for meeting in window_games:
                        hist_values = season_totals.get(meeting["season"], [])
                        hist_avg = float(np.mean(hist_values)) if hist_values else current_league_avg
                        normalized.append(meeting["total"] + current_league_avg - hist_avg)
                    era_mean = float(np.mean(normalized)) if normalized else raw_total
                    shrink = count / (count + SHRINK_K) if count else 0.0
                    record[f"h2h_signal_last{window}_shrunk"] = (mean_total - raw_total) * shrink
                    record[f"h2h_era_signal_last{window}_shrunk"] = (era_mean - raw_total) * shrink
                    record[f"h2h_count_last{window}"] = count

                rows.append(record)

            # Stage pair and league updates; none enter a same-date target's features.
            pending_pair_updates.append((
                tuple(sorted((home_id, away_id))),
                {"date": date, "season": season, "game_id": game_id, "total": actual_total},
            ))
            pending_season_updates.append((season, actual_total))

        for key, update in pending_pair_updates:
            pair_history[key].append(update)
        for season, total in pending_season_updates:
            season_totals[season].append(total)

        if date_index % 100 == 0 or date_index == len(target_dates):
            print(f"Processed {date_index}/{len(target_dates)} dates; target predictions={len(rows)}")

    features = pd.DataFrame(rows)
    if len(features) != 6140:
        raise RuntimeError(f"Expected 6,140 target predictions, produced {len(features)}.")
    if features.duplicated(["game_date", "game_id"]).any():
        raise RuntimeError("Duplicate target-game predictions.")
    h2h_cutoff_ok = features["h2h_max_history_date"].isna() | (
        pd.to_datetime(features["h2h_max_history_date"], errors="coerce")
        < pd.to_datetime(features["game_date"])
    )
    if not bool(h2h_cutoff_ok.all()):
        raise RuntimeError("H2H feature contains a same-day or future meeting.")

    # Development-only coefficient fitting. The existing rule uses a fixed 0.15.
    dev = features[features["period"] == "development"].copy()
    hold = features[features["period"] == "holdout"].copy()
    candidates = {"V5 raw": "v5_raw_total"}
    coefficients = []

    features["legacy_h2h_15pct"] = (
        features["v5_raw_total"] + 0.15 * features["h2h_legacy_signal"]
    )
    candidates["Legacy H2H 15% last-10"] = "legacy_h2h_15pct"

    for window in WINDOWS:
        for signal_kind, signal_col in (
            ("raw", f"h2h_signal_last{window}_shrunk"),
            ("era_adjusted", f"h2h_era_signal_last{window}_shrunk"),
        ):
            beta = fit_beta(dev, signal_col)
            prediction_col = f"v5_h2h_{signal_kind}_last{window}"
            features[prediction_col] = features["v5_raw_total"] + beta * features[signal_col]
            candidates[f"Learned H2H {signal_kind} last-{window} (beta={beta:.4f})"] = prediction_col
            coefficients.append({
                "window": window, "signal": signal_kind,
                "shrink_k": SHRINK_K, "beta": beta, "development_games": len(dev),
            })

    comparison_rows = []
    for period in ("development", "holdout", "all"):
        part = features if period == "all" else features[features["period"] == period]
        for name, column in candidates.items():
            comparison_rows.append({"period": period, "model": name, **metrics(part, column)})

    for season, part in features.groupby("season_start_year", sort=True):
        for name, column in candidates.items():
            comparison_rows.append({
                "period": "season", "season_start_year": int(season),
                "model": name, **metrics(part, column),
            })

    comparison = pd.DataFrame(comparison_rows)
    dev_comparison = comparison[comparison["period"] == "development"].sort_values(
        ["mae", "rmse"]
    )
    selected = dev_comparison.iloc[0]
    selected_name = str(selected["model"])
    selected_col = candidates[selected_name]
    hold_row = comparison[
        (comparison["period"] == "holdout") & (comparison["model"] == selected_name)
    ].iloc[0]
    # Candidate columns are added after the initial split, so rebuild the
    # holdout frame before paired bootstrap comparison.
    hold = features[features["period"] == "holdout"].copy()
    bootstrap = paired_date_bootstrap(hold.reset_index(drop=True), selected_col)

    h2h_counts = features["h2h_games_last10"]
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NBA Stats team_game_logs; llimllib/nba_data v2",
        "team_game_rows": int(len(d)),
        "target_games": int(len(features)),
        "development_games": int(len(dev)),
        "holdout_games": int(len(hold)),
        "target_date_start": str(pd.Timestamp(features.game_date.min()).date()),
        "target_date_end": str(pd.Timestamp(features.game_date.max()).date()),
        "h2h_regular_season_non_neutral_only": True,
        "same_day_results_excluded": True,
        "same_day_h2h_cutoff_passed": bool(h2h_cutoff_ok.all()),
        "h2h_history_count": {
            "mean": float(h2h_counts.mean()),
            "median": float(h2h_counts.median()),
            "zero_count": int((h2h_counts == 0).sum()),
            "at_least_3": int((h2h_counts >= 3).sum()),
            "at_least_5": int((h2h_counts >= 5).sum()),
            "at_least_10": int((h2h_counts >= 10).sum()),
        },
        "coefficients_fit_on_development_only": coefficients,
        "selected_on_development_only": selected_name,
        "selected_holdout_metrics": hold_row.to_dict(),
        "paired_date_cluster_bootstrap_vs_v5": bootstrap,
        "source_updated_utc": {
            name: metadata[name]["updated"] for name in ("team_game_logs", "games")
        },
        "production_modified": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT / "h2h_component_v5_predictions.csv", index=False)
    comparison.to_csv(OUT / "h2h_component_v5_comparison.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "h2h_component_v5_coefficients.csv", index=False)
    (OUT / "h2h_component_v5_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n===== V5 H2H COMPONENT ABLATION =====")
    print(f"Development games: {len(dev)} | Later validation games: {len(hold)}")
    print("\nH2H SAMPLE COUNTS")
    print(json.dumps(summary["h2h_history_count"], indent=2))
    print("\nTOTAL PREDICTION RESULTS")
    print(comparison.to_string(index=False))
    print("\nDEVELOPMENT-FITTED COEFFICIENTS")
    print(pd.DataFrame(coefficients).to_string(index=False))
    print(f"\nSelected on development only: {selected_name}")
    print(
        f"Later validation: MAE={hold_row.mae:.4f}, "
        f"RMSE={hold_row.rmse:.4f}, bias={hold_row.bias:.4f}"
    )
    print("\nPaired date-cluster bootstrap vs V5 raw")
    print(json.dumps(bootstrap, indent=2))
    print(f"\nOutputs: {OUT}")
    print("H2H remains research-only; no production integration.")


if __name__ == "__main__":
    main()
