#!/usr/bin/env python3
"""ORBIT-10 Stage 3: preregistered, one-pattern-at-a-time residual tests.

Uses the exact Stage 2 source loaders and frozen baseline predictions.
No market data, no paid APIs, no feature interactions, no holdout tuning.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/research/orbit10_stage3")
BASELINE = Path("outputs/research/orbit10_stage2/orbit10_baseline_predictions.csv")
STAGE2 = Path("research/orbit10_stage2_baseline.py")
BOOTSTRAPS = 1000
SEED = 20261114
MIN_TRAIN_GROUP = 100

def load_stage2():
    spec = importlib.util.spec_from_file_location("orbit10_stage2", STAGE2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    legacy, _, _ = mod.load_legacy()
    modern, _, _ = mod.load_modern()
    spine = pd.concat([legacy, modern], ignore_index=True)
    spine = spine[spine.season_end_year.between(2001, 2026)].copy()
    spine = spine.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if spine.game_id.duplicated().any():
        raise RuntimeError("Stage 3 source spine contains duplicate game IDs.")
    return spine

def build_features(spine):
    histories = {}
    h2h = {}
    rows = []
    for day, games in spine.groupby("game_date", sort=True):
        pace_values = {}
        for team, hist in histories.items():
            if len(hist) >= 5:
                pace_values[team] = float(np.mean([x["pace"] for x in hist[-25:]]))
        ranked = sorted(pace_values.items(), key=lambda z: z[1])
        bottom = {t for t, _ in ranked[:3]} if len(ranked) >= 20 else set()
        top = {t for t, _ in ranked[-3:]} if len(ranked) >= 20 else set()
        for g in games.to_dict("records"):
            ht, at = str(g["home_team_id"]), str(g["away_team_id"])
            hh, ah = histories.get(ht, []), histories.get(at, [])
            def rest(hist):
                return int((day - hist[-1]["date"]).days - 1) if hist else None
            hr, ar = rest(hh), rest(ah)
            hlast, alast = hh[-25:], ah[-25:]
            hppg = float(np.mean([x["pts"] for x in hlast])) if hlast else np.nan
            # Defensive rating: opponent points per 100 possessions faced.
            hdef = float(100*np.sum([x["opp_pts"] for x in hlast])/np.sum([x["opp_poss"] for x in hlast])) if hlast and sum(x["opp_poss"] for x in hlast)>0 else np.nan
            adef = float(100*np.sum([x["opp_pts"] for x in alast])/np.sum([x["opp_poss"] for x in alast])) if alast and sum(x["opp_poss"] for x in alast)>0 else np.nan
            h2h_rows = [x for x in h2h.get(tuple(sorted((ht,at))), []) if 0 < (day-x["date"]).days <= 42]
            home_held = len(h2h_rows)>3 and all((x["away_pts"] if x["home_team"]==ht else x["home_pts"]) < 100 for x in h2h_rows)
            away_held = len(h2h_rows)>3 and all((x["away_pts"] if x["home_team"]==at else x["home_pts"]) < 100 for x in h2h_rows)
            rows.append({
                "game_id":g["game_id"],
                "p1_home_high_ppg_weak_def": bool(np.isfinite(hppg) and np.isfinite(adef) and hppg>110 and adef>105),
                "p4_either_team_b2b": bool(hr==0 or ar==0),
                "p5_rest_disadvantage": bool((hr is not None and ar is not None) and ((hr<=1 and ar>=3) or (ar<=1 and hr>=3))),
                "p6_recent_h2h_def_dominance": bool(home_held or away_held),
                "p7_top3_vs_bottom3_pace": bool((ht in top and at in bottom) or (at in top and ht in bottom)),
                "p9_first_after_extended_break": bool((hr is not None and hr>=7) or (ar is not None and ar>=7)),
                "home_rest_days":hr, "away_rest_days":ar,
                "h2h_games_last_42d":len(h2h_rows),
                "home_recent_ppg":hppg, "away_recent_def_rating":adef,
            })
        # Same-day results enter only after all games on the date were featured.
        for g in games.to_dict("records"):
            ht, at = str(g["home_team_id"]), str(g["away_team_id"])
            hp, ap = float(g["poss_home"]), float(g["poss_away"])
            pace=(hp+ap)/2
            histories.setdefault(ht,[]).append({"date":day,"pts":float(g["pts_home"]),"opp_pts":float(g["pts_away"]),"opp_poss":ap,"pace":pace})
            histories.setdefault(at,[]).append({"date":day,"pts":float(g["pts_away"]),"opp_pts":float(g["pts_home"]),"opp_poss":hp,"pace":pace})
            key=tuple(sorted((ht,at)))
            h2h.setdefault(key,[]).append({"date":day,"home_team":ht,"away_team":at,"home_pts":float(g["pts_home"]),"away_pts":float(g["pts_away"])})
    return pd.DataFrame(rows)

def metric(y, pred):
    e=np.asarray(pred)-np.asarray(y)
    return {"games":int(len(e)),"mae":round(float(np.mean(np.abs(e))),4),"rmse":round(float(np.sqrt(np.mean(e**2))),4),"bias":round(float(np.mean(e)),4)}

def boot_ci(values, rng):
    x=np.asarray(values,dtype=float)
    if len(x)<2: return [None,None]
    idx=rng.integers(0,len(x),size=(BOOTSTRAPS,len(x)))
    means=x[idx].mean(axis=1)
    return [round(float(np.quantile(means,.025)),4),round(float(np.quantile(means,.975)),4)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not BASELINE.exists(): raise FileNotFoundError("Stage 2 baseline artifact is required.")
    pred=pd.read_csv(BASELINE,parse_dates=["game_date"])
    spine=load_stage2()
    feat=build_features(spine)
    df=pred.merge(feat,on="game_id",how="left",validate="one_to_one")
    if df.filter(regex=r"^p[145679]_").isna().any().any(): raise RuntimeError("Missing Stage 3 feature rows.")
    df["residual"]=df.predicted_total-df.actual_total
    patterns={
      1:("p1_home_high_ppg_weak_def","Home high-scoring profile vs weak opponent defence"),
      4:("p4_either_team_b2b","Either team on second night of back-to-back"),
      5:("p5_rest_disadvantage","One team has <=1 rest day; opponent has >=3"),
      6:("p6_recent_h2h_def_dominance","Recent 42-day H2H: >3 meetings with repeated sub-100 opponent scoring"),
      7:("p7_top3_vs_bottom3_pace","Top-three vs bottom-three pregame pace (minimum 20 teams ranked)"),
      9:("p9_first_after_extended_break","At least one team returning after >=7 calendar days off"),
    }
    rng=np.random.default_rng(SEED)
    reports=[]
    adjusted_frames=[]
    for n,(col,label) in patterns.items():
        tr=df[(df.split=="train") & df[col].notna()]
        grp=tr[tr[col].astype(bool)].residual.to_numpy()
        ctl=tr[~tr[col].astype(bool)].residual.to_numpy()
        result={"pattern":n,"name":label,"feature":col,"train_qualifying_games":int(len(grp)),"train_control_games":int(len(ctl))}
        if len(grp)<MIN_TRAIN_GROUP or len(ctl)<MIN_TRAIN_GROUP:
            result.update({"status":"INSUFFICIENT EVIDENCE","reason":"Fewer than 100 training games in feature or control group."})
            reports.append(result); continue
        effect=float(grp.mean()-ctl.mean())
        # Bootstrap group mean residual difference, sampling each group independently.
        diffs=[]
        for _ in range(BOOTSTRAPS):
            diffs.append(float(rng.choice(grp,len(grp),replace=True).mean()-rng.choice(ctl,len(ctl),replace=True).mean()))
        ci=[round(float(np.quantile(diffs,.025)),4),round(float(np.quantile(diffs,.975)),4)]
        result.update({"status":"TESTED","train_residual_effect_feature_minus_control":round(effect,4),"train_effect_bootstrap_95ci":ci,"train_feature_residual_mean":round(float(grp.mean()),4),"train_control_residual_mean":round(float(ctl.mean()),4)})
        for split in ("validation","holdout"):
            part=df[df.split==split].copy()
            yes=part[part[col].astype(bool)].residual
            no=part[~part[col].astype(bool)].residual
            part_effect=float(yes.mean()-no.mean()) if len(yes) and len(no) else None
            part["adjusted_prediction"]=part.predicted_total + part[col].astype(float)*effect
            result[f"{split}_qualifying_games"]=int(len(yes))
            result[f"{split}_feature_minus_control_residual"]=round(part_effect,4) if part_effect is not None else None
            result[f"{split}_baseline_metrics"]=metric(part.actual_total,part.predicted_total)
            result[f"{split}_pattern_adjusted_metrics"]=metric(part.actual_total,part.adjusted_prediction)
            result[f"{split}_mae_change_adjusted_minus_baseline"]=round(result[f"{split}_pattern_adjusted_metrics"]["mae"]-result[f"{split}_baseline_metrics"]["mae"],4)
            result[f"{split}_rmse_change_adjusted_minus_baseline"]=round(result[f"{split}_pattern_adjusted_metrics"]["rmse"]-result[f"{split}_baseline_metrics"]["rmse"],4)
            adjusted_frames.append(part[["game_id","split","actual_total","predicted_total","adjusted_prediction"]].assign(pattern=n))
        reports.append(result)
    unavailable=[
      {"pattern":2,"status":"NOT TESTABLE — SOURCE GAP","reason":"No verified historical coach/team effective-date and coaching-style source."},
      {"pattern":3,"status":"PARTIAL — SEPARATE ERA STUDY REQUIRED","reason":"Injury reports only documented from 2018-12-17; date-level completeness and point-in-time report versions must be validated before a fair injury experiment."},
      {"pattern":8,"status":"NOT TESTABLE — SOURCE GAP","reason":"No verified historical game-venue/arena coordinates and travel itinerary source."},
      {"pattern":10,"status":"NOT TESTED — PROXY NOT EQUIVALENT","reason":"Standings can be reconstructed, but exact pregame clinching/elimination and motivation states are not verified; a win-percentage proxy must not be labelled motivation."},
    ]
    pd.DataFrame(reports).to_csv(OUT/"orbit10_stage3_pattern_results.csv",index=False)
    if adjusted_frames: pd.concat(adjusted_frames,ignore_index=True).to_csv(OUT/"orbit10_stage3_adjusted_predictions.csv",index=False)
    summary={
      "stage":"ORBIT-10 Stage 3 — individual pattern tests",
      "baseline_artifact":"ORBIT-10 Stage 2 frozen predictions",
      "prediction_rows":int(len(df)),
      "feature_rows":int(len(feat)),
      "tested_patterns":[1,4,5,6,7,9],
      "unavailable_or_partial_patterns":unavailable,
      "pattern_results":reports,
      "protocol":{"effect":"training-only difference in baseline residual means, feature group minus control group","adjustment":"training effect applied unchanged to validation and holdout; no holdout fitting","bootstrap_replicates":BOOTSTRAPS,"seed":SEED,"minimum_train_group_n":MIN_TRAIN_GROUP,"break_threshold_days":7,"h2h_window_days":42,"h2h_min_meetings_exclusive":3,"pace_rank_threshold":3},
      "limitations":["These are univariate association/ablation screens, not causal estimates.","Pattern 1 operationalizes the stated home-scoring/weak-defence hypothesis with pregame trailing-25 team points and defensive efficiency.","Patterns 4, 5, 6, 7 and 9 are schedule/history proxies, not direct physiological or causal measurements.","Patterns 2, 3, 8 and 10 are not falsely backfilled into the full-era common sample."],
      "holdout_policy":"Holdout metrics are reported once as an untouched descriptive check; no tuning or feature selection is permitted from them.",
    }
    (OUT/"orbit10_stage3_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
