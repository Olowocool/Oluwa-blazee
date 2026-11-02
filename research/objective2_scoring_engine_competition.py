#!/usr/bin/env python3
"""Fair chronological scoring-engine competition on archived V4 Phase 2 games."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from live_market_testbench_engine import load_score_history
from points_engine import calculate_matchup_points
from points_engine_v5 import calculate_matchup_points_v5
from team_name_utils import normalize_team_name

BASE = "https://basketball-data.billmill.org"
DEFAULT_INJURY = Path("outputs/research/v4_phase2_reference/v4_phase2_forward_predictions.csv")
OUT = Path("outputs/research/objective2_scoring_engine_competition")
COMPONENTS = ("v5_raw", "v2", "v4_phase1", "injury_rapm")
BOOTSTRAP_REPS, BOOTSTRAP_SEED, BLEND_STEP = 5000, 20261102, 0.05

def metrics(actual, prediction):
    y, p = np.asarray(actual, float), np.asarray(prediction, float)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require equal-length, nonempty finite values.")
    e, a = p-y, np.abs(p-y)
    return {"games": int(len(e)), "mae": float(a.mean()),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "bias_prediction_minus_actual": float(e.mean()),
            "within_5_points_pct": float(np.mean(a <= 5)*100),
            "within_10_points_pct": float(np.mean(a <= 10)*100)}

def simplex_weights(step=BLEND_STEP):
    units = int(round(1/step))
    if not 0 < step <= 1 or abs(units*step-1) > 1e-9:
        raise ValueError("Blend step must divide one exactly.")
    for a in range(units+1):
        for b in range(units-a+1):
            for c in range(units-a-b+1):
                yield np.asarray([a,b,c,units-a-b-c], float)/units

def fit_simplex_blend(frame):
    actual = frame.actual_total.to_numpy(float)
    matrix = frame[list(COMPONENTS)].to_numpy(float)
    if frame.empty or not np.isfinite(matrix).all() or not np.isfinite(actual).all():
        raise ValueError("Blend inputs must be finite and nonempty.")
    best = None
    for weights in simplex_weights():
        score = float(np.mean(np.abs(matrix @ weights-actual)))
        key = (score, -weights[0], -weights[1], -weights[2])
        if best is None or key < best[0]:
            best = (key, weights.copy())
    w = best[1]
    return {"weights": dict(zip(COMPONENTS, map(float,w))),
            "validation_games": int(len(frame)), "validation_mae": float(best[0][0]),
            "grid_step": BLEND_STEP, "constraint": "nonnegative convex weights summing to one"}

def apply_blend(frame, fit):
    w = np.asarray([fit["weights"][x] for x in COMPONENTS], float)
    out = frame.copy()
    out["simplex_blend"] = out[list(COMPONENTS)].to_numpy(float) @ w
    return out

def date_cluster_bootstrap(frame, candidate, baseline="v5_raw", reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    d = frame.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    d["_diff"] = (d[candidate]-d.actual_total).abs()-(d[baseline]-d.actual_total).abs()
    g = d.groupby("game_date", sort=True)["_diff"].agg(["sum","count"])
    if len(g) < 2: raise ValueError("At least two game dates are required.")
    sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng, samples = np.random.default_rng(seed), np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0,len(g),size=len(g))
        samples[i] = sums[ix].sum()/counts[ix].sum()
    low, high = np.quantile(samples,[.025,.975])
    return {"candidate_minus_baseline_mae": float(sums.sum()/counts.sum()),
            "bootstrap_95pct_interval":[float(low),float(high)],
            "bootstrap_replicates":int(reps),"game_dates":int(len(g)),"games":int(len(d))}

def read_reference(path):
    d = pd.read_csv(path)
    req = {"season","game_id","game_date","home_team_id","away_team_id","v4_phase1_total",
           "actual_total","rapm_snapshot_date","injury_adjusted_total","injury_model_train_games"}
    missing = sorted(req-set(d.columns))
    if missing: raise ValueError(f"Phase 2 artifact missing columns: {missing}")
    d["season"] = pd.to_numeric(d.season,errors="raise").astype(int)
    d["game_id"] = d.game_id.astype(str)
    for c in ("game_date","rapm_snapshot_date"):
        d[c] = pd.to_datetime(d[c],errors="coerce").dt.normalize()
    for c in ("v4_phase1_total","actual_total","injury_adjusted_total","injury_model_train_games"):
        d[c] = pd.to_numeric(d[c],errors="coerce")
    cols = ["game_date","rapm_snapshot_date","v4_phase1_total","actual_total","injury_adjusted_total","injury_model_train_games"]
    if d[cols].isna().any().any(): raise ValueError("Phase 2 artifact contains missing dates/model values.")
    if not (d.rapm_snapshot_date < d.game_date).all(): raise ValueError("RAPM snapshot is not strictly pregame.")
    if (d.injury_model_train_games <= 0).any(): raise ValueError("Injury model has no prior training games.")
    if d.duplicated(["season","game_id"]).any(): raise ValueError("Duplicate Phase 2 game IDs.")
    d["home_team_id"], d["away_team_id"] = d.home_team_id.astype(str), d.away_team_id.astype(str)
    d["v4_phase1"], d["injury_rapm"] = d.v4_phase1_total, d.injury_adjusted_total
    return d

def load_game_context():
    con = duckdb.connect()
    con.execute("INSTALL httpfs"); con.execute("LOAD httpfs")
    urls = "['"+BASE+"/nba/stats/games/season=2025/data.parquet','"+BASE+"/nba/stats/games/season=2026/data.parquet']"
    games = con.execute(
        f"SELECT season, CAST(game_id AS VARCHAR) game_id, neutral_site FROM read_parquet({urls}, "
        "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true) "
        "WHERE game_type='regular_season'").fetchdf()
    teams = con.execute(
        f"SELECT team_id, full_name FROM read_parquet('{BASE}/nba/stats/team_seasons/season=2026/data.parquet', "
        "hive_partitioning=true, union_by_name=true) WHERE is_nba=TRUE").fetchdf()
    con.close()
    games["season"] = pd.to_numeric(games.season,errors="raise").astype(int)
    games["game_id"] = games.game_id.astype(str)
    games["neutral_site"] = games.neutral_site.fillna(False).astype(bool)
    if games.duplicated(["season","game_id"]).any(): raise ValueError("Duplicate NBA Stats game IDs.")
    teams["team_id"] = teams.team_id.astype(str)
    teams["full_name"] = teams.full_name.astype(str).str.strip()
    if teams.empty or teams.team_id.duplicated().any(): raise ValueError("Ambiguous team-name map.")
    return games, dict(zip(teams.team_id,teams.full_name))

def score_key(row):
    return (pd.Timestamp(row.game_date).normalize(),
            normalize_team_name(row.home_team).casefold(),
            normalize_team_name(row.away_team).casefold())

def reconcile_scores(frame, history):
    lookup = {}
    for r in history.itertuples(index=False):
        k = score_key(r)
        if k in lookup: raise ValueError(f"Duplicate score key: {k}")
        lookup[k] = float(r.home_score)+float(r.away_score)
    missing, mismatch = [], []
    for r in frame.itertuples(index=False):
        k = (pd.Timestamp(r.game_date).normalize(),
             normalize_team_name(r.home_team).casefold(),
             normalize_team_name(r.away_team).casefold())
        actual = lookup.get(k)
        if actual is None: missing.append(str(r.game_id))
        elif abs(actual-float(r.actual_total)) > .01: mismatch.append(str(r.game_id))
    if missing or mismatch:
        raise ValueError(f"Score reconciliation failed: missing={len(missing)}, mismatch={len(mismatch)}")
    return {"score_rows_reconciled":len(frame),"score_mismatches":0}

def v2_subset(history, day, home, away):
    keys = {normalize_team_name(home).casefold(),normalize_team_name(away).casefold()}
    dates = pd.to_datetime(history.game_date,errors="coerce")
    mask = dates.lt(pd.Timestamp(day).normalize()) & (
        history._home_key.isin(keys) | history._away_key.isin(keys))
    return history.loc[mask].drop(columns=["_home_key","_away_key"]).copy()

def score_engines(frame, score_history):
    h = score_history.copy()
    h["game_date"] = pd.to_datetime(h.game_date,errors="coerce").dt.normalize()
    h["_home_key"] = h.home_team.map(lambda x:normalize_team_name(x).casefold())
    h["_away_key"] = h.away_team.map(lambda x:normalize_team_name(x).casefold())
    cache = {"v3":{},"v4":{}}
    v2s, v5s, v2max = [],[],[]
    for r in frame.itertuples(index=False):
        day = pd.Timestamp(r.game_date).normalize()
        prior = v2_subset(h,day,r.home_team,r.away_team)
        if not prior.empty and not (pd.to_datetime(prior.game_date)<day).all():
            raise AssertionError("V2 received same-day/future history.")
        p2 = calculate_matchup_points(r.home_team,r.away_team,history_df=prior,game_date=day)
        p5 = calculate_matchup_points_v5(r.home_team,r.away_team,history_df=h,game_date=day,_prepared_cache=cache)
        v2s.append(float(p2["projected_total"]))
        v5s.append(float(p5["raw_total"]))
        v2max.append(pd.to_datetime(prior.game_date).max() if not prior.empty else pd.NaT)
    out=frame.copy(); out["v2"]=v2s; out["v5_raw"]=v5s; out["v2_max_history_date"]=v2max
    if out[["v2","v5_raw"]].isna().any().any() or not np.isfinite(out[["v2","v5_raw"]].to_numpy(float)).all():
        raise ValueError("Scoring engine returned missing/non-finite predictions.")
    for c in ("v2_max_history_date","rapm_snapshot_date"):
        if out[c].isna().any() or not (pd.to_datetime(out[c])<pd.to_datetime(out.game_date)).all():
            raise AssertionError(f"Point-in-time cutoff failed: {c}")
    return out

def metric_rows(frame, split):
    return [{"split":split,"model":name,**metrics(frame.actual_total,frame[name])}
            for name in COMPONENTS+("simplex_blend",)]

def run(injury_path=DEFAULT_INJURY, output_dir=OUT):
    injury_path, output_dir = Path(injury_path), Path(output_dir)
    if not injury_path.is_file(): raise FileNotFoundError(f"Missing Phase 2 evidence artifact: {injury_path}")
    output_dir.mkdir(parents=True,exist_ok=True)
    ref=read_reference(injury_path)
    counts=ref.groupby("season").size().to_dict()
    if set(counts)!={2025,2026} or counts!={2025:1148,2026:1130}:
        raise ValueError(f"Unexpected Phase 2 artifact counts: {counts}")
    games, team_map=load_game_context()
    ref=ref.merge(games,on=["season","game_id"],how="left",validate="one_to_one")
    if ref.neutral_site.isna().any(): raise ValueError("Missing neutral-site status.")
    neutral_excluded=int(ref.neutral_site.sum())
    ref=ref.loc[~ref.neutral_site].copy()
    ref["home_team"]=ref.home_team_id.map(team_map); ref["away_team"]=ref.away_team_id.map(team_map)
    if ref[["home_team","away_team"]].isna().any().any(): raise ValueError("Unmapped NBA team ID.")
    ref=ref.sort_values(["game_date","game_id"]).reset_index(drop=True)
    history=load_score_history()
    history["game_date"]=pd.to_datetime(history.game_date,errors="coerce").dt.normalize()
    reconciliation=reconcile_scores(ref,history)
    scored=score_engines(ref,history)
    val=scored.loc[scored.season.eq(2025)].copy()
    hold=scored.loc[scored.season.eq(2026)].copy()
    if val.empty or hold.empty or val.game_date.max()>=hold.game_date.min():
        raise ValueError("Validation/holdout chronology is invalid.")
    fit=fit_simplex_blend(val)
    val=apply_blend(val,fit); hold=apply_blend(hold,fit)
    val_metrics=pd.DataFrame(metric_rows(val,"validation_2024_25"))
    hold_metrics=pd.DataFrame(metric_rows(hold,"previously_examined_holdout_2025_26"))
    season_rows=[]
    for season,part in ((2025,val),(2026,hold)):
        for name in COMPONENTS+("simplex_blend",):
            season_rows.append({"season":season,"model":name,**metrics(part.actual_total,part[name])})
    boot={}
    for i,candidate in enumerate(("v2","v4_phase1","injury_rapm","simplex_blend"),1):
        boot[candidate]=date_cluster_bootstrap(hold,candidate,seed=BOOTSTRAP_SEED+i)
    boot["injury_rapm_vs_v4_phase1"]=date_cluster_bootstrap(
        hold,"injury_rapm",baseline="v4_phase1",seed=BOOTSTRAP_SEED+10)
    scored.to_csv(output_dir/"paired_predictions.csv",index=False)
    val_metrics.to_csv(output_dir/"validation_metrics.csv",index=False)
    hold_metrics.to_csv(output_dir/"holdout_metrics.csv",index=False)
    pd.DataFrame(season_rows).to_csv(output_dir/"season_metrics.csv",index=False)
    summary={
        "status":"completed",
        "source_artifact":str(injury_path),
        "source_artifact_rows":int(len(ref)+neutral_excluded),
        "source_artifact_season_counts":{str(k):int(v) for k,v in counts.items()},
        "neutral_site_games_excluded":neutral_excluded,
        "common_games_after_neutral_exclusion":int(len(scored)),
        "validation_games_2024_25":int(len(val)),
        "holdout_games_2025_26":int(len(hold)),
        "validation_date_range":[str(val.game_date.min().date()),str(val.game_date.max().date())],
        "holdout_date_range":[str(hold.game_date.min().date()),str(hold.game_date.max().date())],
        "score_history_rows":int(len(history)),
        "score_history_latest_date":str(history.game_date.max().date()),
        "score_reconciliation":reconciliation,
        "blend_fit_on_validation_only":fit,
        "validation_metrics":val_metrics.to_dict(orient="records"),
        "holdout_metrics":hold_metrics.to_dict(orient="records"),
        "season_metrics":season_rows,
        "paired_date_cluster_bootstrap":boot,
        "point_in_time":{"v2_history_strictly_prior":True,"v5_history_strictly_prior":True,
                         "rapm_snapshot_strictly_prior":True,"same_date_games_batch_isolated":True,
                         "holdout_used_to_fit_blend":False},
        "limitations":[
            "The 2025-26 holdout was previously examined in separate V4 and V5 research; this is a paired re-evaluation, not a fresh confirmatory holdout.",
            "Phase 2 injury/RAPM predictions are reused from the completed 2026-10-30 workflow artifact; they are not retrained.",
            "Historical NBA Stats may contain retrospective corrections.",
            "This is score-prediction accuracy research only; no market entry prices or ROI are evaluated."
        ],
        "production_modified":False,"raw_market_data_used":False,"paid_api_credits_used":0}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    (output_dir/"report.md").write_text(render_report(summary),encoding="utf-8")
    print((output_dir/"report.md").read_text(encoding="utf-8"))
    return summary

def render_report(s):
    lines=[
        "# Objective 2 — fair scoring-engine competition","",
        "**Status:** completed historical paired evaluation; prospective confirmation remains open.","",
        "## Population and protocol","",
        f"- Validation: {s['validation_games_2024_25']:,} games, {s['validation_date_range'][0]} to {s['validation_date_range'][1]}.",
        f"- Previously examined holdout: {s['holdout_games_2025_26']:,} games, {s['holdout_date_range'][0]} to {s['holdout_date_range'][1]}.",
        f"- Identical target games for V2, V5, V4 Phase 1 and injury/RAPM; neutral-site exclusions: {s['neutral_site_games_excluded']}.",
        f"- Score reconciliation: {s['score_reconciliation']['score_rows_reconciled']:,} matches; {s['score_reconciliation']['score_mismatches']} mismatches.",
        "- V2/V5 histories and RAPM snapshots are strictly pregame; same-day games are batch-isolated.","",
        "## Validation results — 2024-25","",
        "| Model | Games | MAE | RMSE | Bias | Within 5 | Within 10 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in s["validation_metrics"]:
        lines.append(f"| {r['model']} | {r['games']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['bias_prediction_minus_actual']:.4f} | {r['within_5_points_pct']:.2f}% | {r['within_10_points_pct']:.2f}% |")
    lines += ["","## Previously examined holdout — 2025-26","",
        "| Model | Games | MAE | RMSE | Bias | Within 5 | Within 10 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in s["holdout_metrics"]:
        lines.append(f"| {r['model']} | {r['games']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['bias_prediction_minus_actual']:.4f} | {r['within_5_points_pct']:.2f}% | {r['within_10_points_pct']:.2f}% |")
    lines += ["","## Paired date-cluster bootstrap — holdout","",
        "Negative candidate-minus-V5 MAE favours the candidate; intervals crossing zero do not establish a reliable improvement."]
    for name in ("v2","v4_phase1","injury_rapm","simplex_blend"):
        b=s["paired_date_cluster_bootstrap"][name]
        lines.append(f"- **{name} vs V5:** {b['candidate_minus_baseline_mae']:+.4f}; 95% interval [{b['bootstrap_95pct_interval'][0]:+.4f}, {b['bootstrap_95pct_interval'][1]:+.4f}].")
    b=s["paired_date_cluster_bootstrap"]["injury_rapm_vs_v4_phase1"]
    lines.append(f"- **Injury/RAPM vs V4 Phase 1:** {b['candidate_minus_baseline_mae']:+.4f}; 95% interval [{b['bootstrap_95pct_interval'][0]:+.4f}, {b['bootstrap_95pct_interval'][1]:+.4f}].")
    lines += ["","## Combined challenger","",
        "Four-component convex blend fitted on 2024-25 validation only: " + json.dumps(s["blend_fit_on_validation_only"]["weights"],sort_keys=True) + ".",
        f"- Validation MAE: {s['blend_fit_on_validation_only']['validation_mae']:.4f}; weights frozen before 2025-26 scoring.","",
        "## Decision and limitations","",
        "- Retain V5 as the research benchmark unless a challenger shows a repeatable, uncertainty-supported advantage.",
        "- The 2025-26 holdout has been examined previously; this is a fair paired alignment audit, not a fresh confirmatory test.",
        "- Injury/RAPM predictions are reused from the completed 2026-10-30 experiment, not retrained.",
        "- No market prices, ROI, production model, live app or paper ledger were changed.",
        "- Prospective confirmation on 2026-27 is still required once a sufficiently large sample accumulates.",""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injury-predictions",default=str(DEFAULT_INJURY))
    p.add_argument("--output-dir",default=str(OUT))
    a=p.parse_args()
    run(a.injury_predictions,a.output_dir)

if __name__=="__main__":
    main()
