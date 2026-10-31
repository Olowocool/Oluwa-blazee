#!/usr/bin/env python3
"""Point-in-time V5 pace-component ablation (research only).

Run from repo root:
    python research/pace_component_v5_ablation.py

Uses NBA Stats team_game_logs via llimllib/nba_data v2. The source play-by-play
poss field is the primary possession measure; FGA + 0.44*FTA - OREB + TOV is
audited against it rather than assumed exact. No production files are changed.
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
TARGET_START_YEARS = [2021, 2022, 2023, 2024, 2025]
DEV_YEARS = {2021, 2022, 2023}
PRIOR_GAMES = 20.0
MIN_TEAM_GAMES = 5
BOOTSTRAPS = 5000
SEED = 20261031
OUT = Path("outputs/research/pace_component_v5")


def wmean(values, half_life=5.0, default=np.nan):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float(default)
    age = np.arange(len(a) - 1, -1, -1, dtype=float)
    w = np.exp(-np.log(2.0) * age / float(half_life))
    return float(np.average(a, weights=w))


def league_stats(rows, fallback=None):
    fallback = fallback or {"pace": 100.0, "ortg": 110.0, "drtg": 110.0}
    if not rows:
        return dict(fallback)
    poss = np.asarray([x["poss"] for x in rows], dtype=float)
    pts = np.asarray([x["pts"] for x in rows], dtype=float)
    pace = np.asarray([x["pace"] for x in rows], dtype=float)
    drtg = np.asarray([x["def"] for x in rows], dtype=float)
    ok = np.isfinite(poss) & (poss > 0) & np.isfinite(pts)
    ortg = 100.0 * pts[ok].sum() / poss[ok].sum() if ok.any() else fallback["ortg"]
    return {
        "pace": float(np.nanmean(pace)) if np.isfinite(pace).any() else fallback["pace"],
        "ortg": float(ortg),
        "drtg": float(np.nanmean(drtg)) if np.isfinite(drtg).any() else fallback["drtg"],
    }


def load_source():
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as r:
        meta = json.loads(r.read().decode())["datasets"]
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))
    for table in ("team_game_logs", "games"):
        m = meta[table]
        if int(m["first_season"]) > FIRST_SEASON or int(m["last_season"]) < LAST_SEASON:
            raise RuntimeError(f"{table} metadata does not cover required seasons: {m}")
        urls = ", ".join(
            "'" + f"{BASE}/nba/stats/{table}/season={s}/data.parquet" + "'"
            for s in seasons
        )
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
    raw = con.execute("""
        SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
               t.home, t.pts, t.poss, t.pace, t.off_rating, t.def_rating,
               t.fga, t.fta, t.oreb, t.tov, t.team_oreb,
               g.game_type, g.neutral_site
        FROM team_game_logs t JOIN games g USING (season, game_id)
        WHERE t.season BETWEEN ? AND ? AND g.game_type = 'regular_season'
        ORDER BY t.game_date, t.season, t.game_id, t.team_id
    """, [FIRST_SEASON, LAST_SEASON]).fetchdf()
    con.close()
    return raw, meta


def pair_source(raw):
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    for c in ("season", "pts", "poss", "pace", "off_rating", "def_rating",
              "fga", "fta", "oreb", "tov", "team_oreb"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["team_id"] = d.team_id.astype(str)
    d["opp_team_id"] = d.opp_team_id.astype(str)
    d = d.dropna(subset=["game_date", "season", "pts", "poss", "pace",
                         "off_rating", "def_rating", "fga", "fta", "oreb", "tov", "home"])
    d = d[(d.poss > 0) & (d.pace > 0) & (~d.neutral_site.fillna(False).astype(bool))]
    if d.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate (season, game_id, team_id) keys.")
    d["formula_poss"] = d.fga + 0.44 * d.fta - d.oreb + d.tov

    dates = defaultdict(list)
    score_rows, audit_rows = [], []
    for (season, gid), g in d.groupby(["season", "game_id"], sort=False):
        if len(g) != 2 or g.team_id.nunique() != 2:
            raise RuntimeError(f"Invalid team-row pairing for game {gid}.")
        rows = g.to_dict("records")
        hr = [r for r in rows if bool(r["home"])]
        ar = [r for r in rows if not bool(r["home"])]
        if len(hr) != 1 or len(ar) != 1:
            raise RuntimeError(f"Invalid home/away flags for game {gid}.")
        h, a = hr[0], ar[0]
        if h["opp_team_id"] != a["team_id"] or a["opp_team_id"] != h["team_id"]:
            raise RuntimeError(f"Broken reciprocal opponent link for game {gid}.")
        date = pd.Timestamp(h["game_date"]).normalize()
        season = int(season)
        gid = str(gid)
        dates[date].append((season, gid, h, a))
        score_rows.append({
            "game_date": date, "home_team": h["team_id"], "away_team": a["team_id"],
            "home_score": float(h["pts"]), "away_score": float(a["pts"]),
            "total_score": float(h["pts"]) + float(a["pts"]),
        })
        for r in (h, a):
            delta = float(r["formula_poss"]) - float(r["poss"])
            audit_rows.append({
                "season": season, "source_poss": float(r["poss"]),
                "formula_poss": float(r["formula_poss"]),
                "abs_formula_diff": abs(delta),
                "team_oreb_available": bool(pd.notna(r["team_oreb"])),
            })
    scores = pd.DataFrame(score_rows).sort_values("game_date").reset_index(drop=True)
    return d, dates, scores, pd.DataFrame(audit_rows)


def team_form(team_id, season, team_history, lg, prev_lg):
    history = team_history.get(team_id, [])
    cur = [r for r in history if r["season"] == season]
    prev = [r for r in history if r["season"] == season - 1]
    n = len(cur)
    prior = {
        "pace": wmean([r["pace"] for r in prev[-25:]], 12, prev_lg["pace"]),
        "off": wmean([r["off"] for r in prev[-25:]], 12, prev_lg["ortg"]),
        "def": wmean([r["def"] for r in prev[-25:]], 12, prev_lg["drtg"]),
    }
    current = {
        "pace10": wmean([r["pace"] for r in cur[-10:]], 5, lg["pace"]),
        "pace20": wmean([r["pace"] for r in cur[-20:]], 10, lg["pace"]),
        "pace_season": float(np.mean([r["pace"] for r in cur])) if cur else lg["pace"],
        "off10": wmean([r["off"] for r in cur[-10:]], 5, lg["ortg"]),
        "off_season": float(np.mean([r["off"] for r in cur])) if cur else lg["ortg"],
        "def10": wmean([r["def"] for r in cur[-10:]], 5, lg["drtg"]),
        "def_season": float(np.mean([r["def"] for r in cur])) if cur else lg["drtg"],
    }
    shrink = n / (n + PRIOR_GAMES) if n else 0.0
    pace10 = shrink * current["pace10"] + (1 - shrink) * prior["pace"]
    pace20 = shrink * current["pace20"] + (1 - shrink) * prior["pace"]
    pace_blend_current = 0.5 * current["pace10"] + 0.3 * current["pace20"] + 0.2 * current["pace_season"]
    pace_blend = shrink * pace_blend_current + (1 - shrink) * prior["pace"]
    off_cur = 0.6 * current["off10"] + 0.4 * current["off_season"]
    def_cur = 0.6 * current["def10"] + 0.4 * current["def_season"]
    return {
        "n": n,
        "pace10": pace10,
        "pace20": pace20,
        "pace_blend": pace_blend,
        "off": shrink * off_cur + (1 - shrink) * prior["off"],
        "def": shrink * def_cur + (1 - shrink) * prior["def"],
    }


def legacy_proxy(team_id, team_history, target_home):
    """Recreate legacy score-proxy pace using only this team's prior games."""
    hist = team_history.get(team_id, [])
    if not hist:
        return 100.0
    totals = [x["game_total"] for x in hist]
    venue = [x["game_total"] for x in hist if bool(x["home"]) == bool(target_home)]
    vals = [
        _mean(totals[-10:], 225.0),
        _mean(totals[-20:], 225.0),
        _mean(totals, 225.0),
        _mean(venue[-20:], _mean(totals, 225.0)),
    ]
    proxy = [100.0 + ((v - 225.0) / 225.0) * 20.0 for v in vals]
    return 0.40 * proxy[0] + 0.25 * proxy[1] + 0.20 * proxy[2] + 0.15 * proxy[3]


def _mean(values, default):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float(default)


def fit_beta(dev, signal):
    x = dev[signal].to_numpy(float)
    y = (dev.actual_total - dev.v5_raw_total).to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    denom = float(x @ x)
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def score_metrics(frame, col):
    e = frame[col].to_numpy(float) - frame.actual_total.to_numpy(float)
    a = np.abs(e)
    return {
        "games": int(len(e)), "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(e.mean()),
        "within_5_pct": float((a <= 5).mean() * 100),
        "within_10_pct": float((a <= 10).mean() * 100),
    }


def bootstrap_delta(frame, candidate_col):
    rng = np.random.default_rng(SEED)
    groups = [g.index.to_numpy() for _, g in frame.groupby("game_date", sort=True)]
    positions = {idx: i for i, idx in enumerate(frame.index)}
    groups = [np.asarray([positions[idx] for idx in group]) for group in groups]
    actual = frame.actual_total.to_numpy(float)
    base = frame.v5_raw_total.to_numpy(float)
    cand = frame[candidate_col].to_numpy(float)
    base_abs, cand_abs = np.abs(base - actual), np.abs(cand - actual)
    base_sq, cand_sq = (base - actual) ** 2, (cand - actual) ** 2
    d_mae, d_rmse = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        selected = rng.integers(0, len(groups), size=len(groups))
        pos = np.concatenate([groups[j] for j in selected])
        d_mae[i] = cand_abs[pos].mean() - base_abs[pos].mean()
        d_rmse[i] = np.sqrt(cand_sq[pos].mean()) - np.sqrt(base_sq[pos].mean())
    return {
        "mae_difference_candidate_minus_v5": float(cand_abs.mean() - base_abs.mean()),
        "mae_95ci": [float(np.quantile(d_mae, .025)), float(np.quantile(d_mae, .975))],
        "probability_candidate_lower_mae": float(np.mean(d_mae < 0)),
        "rmse_difference_candidate_minus_v5": float(np.sqrt(cand_sq.mean()) - np.sqrt(base_sq.mean())),
        "rmse_95ci": [float(np.quantile(d_rmse, .025)), float(np.quantile(d_rmse, .975))],
        "probability_candidate_lower_rmse": float(np.mean(d_rmse < 0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, metadata = load_source()
    d, dates, scores, poss_audit = pair_source(raw)
    print(f"Loaded {len(d):,} regular-season team-game rows.")

    # The score history is complete and static; V3/V4 internally enforce the
    # strict target-date cutoff, and the cache stores date-specific fits.
    history_scores = scores.copy()
    team_hist, league_hist = defaultdict(list), defaultdict(list)
    rows = []
    v5_cache = {}

    for date_idx, date in enumerate(sorted(dates), start=1):
        today_rows = dates[date]
        pending_team_updates = []
        for season, gid, h, a in today_rows:
            start_year = season - 1
            if start_year not in TARGET_START_YEARS:
                pass
            else:
                lg = league_stats(league_hist[season],
                                  league_stats(league_hist[season - 1]) if league_hist[season - 1] else None)
                prev_lg = league_stats(league_hist[season - 1]) if league_hist[season - 1] else lg
                hid, aid = str(h["team_id"]), str(a["team_id"])
                hf = team_form(hid, season, team_hist, lg, prev_lg)
                af = team_form(aid, season, team_hist, lg, prev_lg)
                if min(hf["n"], af["n"]) >= MIN_TEAM_GAMES:
                    v5 = calculate_matchup_points_v5(
                        hid, aid, history_scores, game_date=date,
                        _prepared_cache=v5_cache,
                    )
                    expected = {
                        "l10": 0.5 * (hf["pace10"] + af["pace10"]),
                        "l20": 0.5 * (hf["pace20"] + af["pace20"]),
                        "blend": 0.5 * (hf["pace_blend"] + af["pace_blend"]),
                    }
                    gaps = {k: float(v - lg["pace"]) for k, v in expected.items()}
                    home_eff = lg["ortg"] + 0.5 * (hf["off"] - lg["ortg"]) + 0.5 * (af["def"] - lg["drtg"])
                    away_eff = lg["ortg"] + 0.5 * (af["off"] - lg["ortg"]) + 0.5 * (hf["def"] - lg["drtg"])
                    conversion = (home_eff + away_eff) / 100.0
                    actual_pace = 0.5 * (float(h["pace"]) + float(a["pace"]))
                    latest = [x["date"] for x in (team_hist[hid] + team_hist[aid])]
                    max_history_date = max(latest) if latest else pd.NaT
                    if pd.notna(max_history_date) and not max_history_date < date:
                        raise RuntimeError("Pace history contains same-day/future data.")

                    row = {
                        "game_date": date, "game_id": gid, "season_end_year": season,
                        "season_start_year": start_year, "period": "development" if start_year in DEV_YEARS else "holdout",
                        "home_team_id": hid, "away_team_id": aid,
                        "actual_total": float(h["pts"]) + float(a["pts"]),
                        "actual_pace": actual_pace,
                        "actual_possessions": 0.5 * (float(h["poss"]) + float(a["poss"])),
                        "v5_raw_total": float(v5["raw_total"]),
                        "league_pace_prior": lg["pace"], "league_ortg_prior": lg["ortg"],
                        "pace_expected_l10": expected["l10"], "pace_expected_l20": expected["l20"],
                        "pace_expected_blend": expected["blend"],
                        "pace_gap_l10": gaps["l10"], "pace_gap_l20": gaps["l20"], "pace_gap_blend": gaps["blend"],
                        "matchup_home_ortg": home_eff, "matchup_away_ortg": away_eff,
                        "points_per_pace_unit": conversion,
                        "physical_adj_l10": gaps["l10"] * conversion,
                        "physical_adj_l20": gaps["l20"] * conversion,
                        "physical_adj_blend": gaps["blend"] * conversion,
                        "legacy_score_pace_adj": (
                            (0.5 * (legacy_proxy(hid, team_hist, True) + legacy_proxy(aid, team_hist, False)) - 100.0) * 2.5
                        ),
                        "max_history_date": max_history_date,
                    }
                    rows.append(row)

            # Stage today's team records; only append after every game date is predicted.
            for r, opp in ((h, a), (a, h)):
                pending_team_updates.append((str(r["team_id"]), {
                    "season": season, "date": date, "game_id": gid,
                    "pace": float(r["pace"]), "poss": float(r["poss"]),
                    "off": float(r["off_rating"]), "def": float(r["def_rating"]),
                    "pts": float(r["pts"]), "game_total": float(h["pts"]) + float(a["pts"]),
                    "home": bool(r["home"]),
                }))

        for team_id, update in pending_team_updates:
            team_hist[team_id].append(update)
            league_hist[update["season"]].append(update)

        if date_idx % 100 == 0 or date_idx == len(dates):
            print(f"Processed {date_idx}/{len(dates)} dates; targets={len(rows)}")

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No target predictions created.")
    if features.duplicated(["game_date", "game_id"]).any():
        raise RuntimeError("Duplicate game predictions.")
    if not (pd.to_datetime(features.max_history_date) < pd.to_datetime(features.game_date)).all():
        raise RuntimeError("Same-day/future pace features detected.")

    dev = features[features.period == "development"].copy()
    hold = features[features.period == "holdout"].copy()
    if dev.empty or hold.empty:
        raise RuntimeError("Missing development or holdout rows.")

    candidates = {"V5 raw": "v5_raw_total"}
    coefficients = []
    for variant in ("l10", "l20", "blend"):
        signal = f"pace_gap_{variant}"
        beta = fit_beta(dev, signal)
        learned = f"v5_learned_{variant}"
        physical = f"v5_physical_{variant}"
        features[learned] = features.v5_raw_total + beta * features[signal]
        features[physical] = features.v5_raw_total + features[f"physical_adj_{variant}"]
        candidates[f"V5 + learned pace {variant} (beta={beta:.3f})"] = learned
        candidates[f"V5 + physical efficiency pace {variant}"] = physical
        coefficients.append({"variant": variant, "beta_points_per_pace_point": beta,
                             "fit_games": len(dev)})

    features["v5_legacy_score_proxy"] = features.v5_raw_total + features.legacy_score_pace_adj
    candidates["V5 + legacy score proxy"] = "v5_legacy_score_proxy"

    comparison_rows, pace_rows = [], []
    for period in ("development", "holdout", "all"):
        part = features if period == "all" else features[features.period == period]
        for name, col in candidates.items():
            comparison_rows.append({"period": period, "model": name, **score_metrics(part, col)})

        for variant in ("l10", "l20", "blend"):
            p = part[f"pace_expected_{variant}"].to_numpy(float)
            a = part.actual_pace.to_numpy(float)
            e = p - a
            pace_rows.append({
                "period": period, "pace_variant": variant, "games": len(part),
                "pace_mae": float(np.mean(np.abs(e))),
                "pace_rmse": float(np.sqrt(np.mean(e ** 2))),
                "pace_bias": float(e.mean()),
                "pace_correlation": float(np.corrcoef(p, a)[0, 1]),
            })

    comparison = pd.DataFrame(comparison_rows)
    pace_metrics = pd.DataFrame(pace_rows)
    # Select pace candidate using development MAE only.
    dev_comparison = comparison[comparison.period == "development"].sort_values(["mae", "rmse"])
    winner = dev_comparison.iloc[0]
    winner_name, winner_col = str(winner.model), candidates[str(winner.model)]
    holdout_row = comparison[(comparison.period == "holdout") & (comparison.model == winner_name)].iloc[0]

    # Paired bootstrap against V5 raw on later validation dates only.
    rng = np.random.default_rng(SEED)
    date_groups = [g.index.to_numpy() for _, g in hold.groupby("game_date", sort=True)]
    positions = {idx: i for i, idx in enumerate(hold.index)}
    groups = [np.asarray([positions[idx] for idx in group]) for group in date_groups]
    actual = hold.actual_total.to_numpy(float)
    base = hold.v5_raw_total.to_numpy(float)
    cand = hold[winner_col].to_numpy(float)
    base_abs, cand_abs = np.abs(base - actual), np.abs(cand - actual)
    base_sq, cand_sq = (base - actual) ** 2, (cand - actual) ** 2
    d_mae, d_rmse = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        pick = rng.integers(0, len(groups), size=len(groups))
        pos = np.concatenate([groups[j] for j in pick])
        d_mae[i] = cand_abs[pos].mean() - base_abs[pos].mean()
        d_rmse[i] = np.sqrt(cand_sq[pos].mean()) - np.sqrt(base_sq[pos].mean())
    bootstrap = {
        "mae_difference_candidate_minus_v5": float(cand_abs.mean() - base_abs.mean()),
        "mae_95ci": [float(np.quantile(d_mae, .025)), float(np.quantile(d_mae, .975))],
        "probability_candidate_lower_mae": float(np.mean(d_mae < 0)),
        "rmse_difference_candidate_minus_v5": float(np.sqrt(cand_sq.mean()) - np.sqrt(base_sq.mean())),
        "rmse_95ci": [float(np.quantile(d_rmse, .025)), float(np.quantile(d_rmse, .975))],
        "probability_candidate_lower_rmse": float(np.mean(d_rmse < 0)),
    }

    poss_delta = (poss_audit.formula_poss - poss_audit.poss).abs()
    poss_summary = {
        "rows": int(len(poss_audit)),
        "formula_mae_vs_play_by_play_poss": float(poss_delta.mean()),
        "formula_median_abs_diff": float(poss_delta.median()),
        "formula_p95_abs_diff": float(poss_delta.quantile(.95)),
        "exact_within_0_01_pct": float((poss_delta <= .01).mean() * 100),
    }
    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "target_start_years": TARGET_START_YEARS,
        "development_start_years": sorted(DEV_YEARS),
        "development_games": int(len(dev)),
        "holdout_games": int(len(hold)),
        "all_games": int(len(features)),
        "possession_audit": poss_summary,
        "selected_on_development_only": winner_name,
        "selected_holdout_metrics": holdout_row.to_dict(),
        "paired_date_cluster_bootstrap_vs_v5": bootstrap,
        "same_day_results_excluded": True,
        "production_modified": False,
        "source_updated_utc": {
            x: metadata[x]["updated"] for x in ("team_game_logs", "games")
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT / "pace_component_v5_predictions.csv", index=False)
    comparison.to_csv(OUT / "pace_component_v5_comparison.csv", index=False)
    pace_metrics.to_csv(OUT / "pace_component_v5_pace_accuracy.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "pace_component_v5_coefficients.csv", index=False)
    (OUT / "pace_component_v5_summary.json").write_text(json.dumps(payload, indent=2, default=str))

    print("\n===== V5 POSSESSION/PACE COMPONENT ABLATION =====")
    print(f"Development games: {len(dev)} | Later validation games: {len(hold)}")
    print("\nPOSSESSION FORMULA AUDIT")
    print(json.dumps(poss_summary, indent=2))
    print("\nTOTAL PREDICTION RESULTS")
    print(comparison.to_string(index=False))
    print("\nDIRECT PACE FORECAST RESULTS")
    print(pace_metrics.to_string(index=False))
    print("\nDEVELOPMENT-FITTED PACE COEFFICIENTS")
    print(pd.DataFrame(coefficients).to_string(index=False))
    print(f"\nSelected on development only: {winner_name}")
    print("Later validation result: "
          f"MAE={holdout_row.mae:.4f}, RMSE={holdout_row.rmse:.4f}, "
          f"bias={holdout_row.bias:.4f}")
    print("Paired date-cluster bootstrap vs V5 raw:")
    print(json.dumps(bootstrap, indent=2))
    print(f"\nOutputs: {OUT}")
    print("No pace adjustment is promoted automatically. Research only.")


def score_metrics(frame, col):
    e = frame[col].to_numpy(float) - frame.actual_total.to_numpy(float)
    a = np.abs(e)
    return {
        "games": int(len(e)), "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(e.mean()),
        "within_5_pct": float((a <= 5).mean() * 100),
        "within_10_pct": float((a <= 10).mean() * 100),
    }


if __name__ == "__main__":
    main()
