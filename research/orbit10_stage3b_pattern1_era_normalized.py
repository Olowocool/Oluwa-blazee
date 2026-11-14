#!/usr/bin/env python3
"""ORBIT-10 Pattern 1 era-normalized replication.

Uses pregame, venue-specific trailing-25 rates and same-date league context.
Thresholds are estimated on TRAIN only (75th percentile of each z-score),
then frozen for VALIDATION. The final HOLDOUT is deliberately excluded.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path("outputs/research/orbit10_stage3b_pattern1")
BASELINE=Path("outputs/research/orbit10_stage2/orbit10_baseline_predictions.csv")
STAGE2=Path("research/orbit10_stage2_baseline.py")
SEED=20261114
BOOTSTRAPS=2000
MIN_GROUP=100
ROLLING=25
MIN_VENUE_GAMES=5
TRAIN_Q=0.75

def load_spine():
    spec=importlib.util.spec_from_file_location("orbit10_stage2",STAGE2)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    legacy,_,_=mod.load_legacy(); modern,_,_=mod.load_modern()
    s=pd.concat([legacy,modern],ignore_index=True)
    s=s[s.season_end_year.between(2001,2026)].copy()
    s=s.sort_values(["game_date","game_id"]).reset_index(drop=True)
    if s.game_id.duplicated().any(): raise RuntimeError("Duplicate source game IDs")
    return s

def build_features(spine):
    histories={}
    out=[]
    for day,games in spine.groupby("game_date",sort=True):
        venue_stats={}
        for team,hist in histories.items():
            home=[x for x in hist if x["is_home"]][-ROLLING:]
            road=[x for x in hist if not x["is_home"]][-ROLLING:]
            if len(home)>=MIN_VENUE_GAMES:
                venue_stats[(team,"home")]=float(np.mean([x["pts"] for x in home]))
            if len(road)>=MIN_VENUE_GAMES and sum(x["opp_poss"] for x in road)>0:
                venue_stats[(team,"road")]=float(100*sum(x["opp_pts"] for x in road)/sum(x["opp_poss"] for x in road))
        home_vals=np.array([v for (t,k),v in venue_stats.items() if k=="home"],dtype=float)
        road_vals=np.array([v for (t,k),v in venue_stats.items() if k=="road"],dtype=float)
        hm=float(home_vals.mean()) if len(home_vals) else np.nan
        hs=float(home_vals.std(ddof=0)) if len(home_vals)>1 else np.nan
        rm=float(road_vals.mean()) if len(road_vals) else np.nan
        rs=float(road_vals.std(ddof=0)) if len(road_vals)>1 else np.nan
        for g in games.to_dict("records"):
            ht,at=str(g["home_team_id"]),str(g["away_team_id"])
            hppg=venue_stats.get((ht,"home"),np.nan)
            adef=venue_stats.get((at,"road"),np.nan)
            zh=(hppg-hm)/hs if np.isfinite(hppg) and np.isfinite(hm) and hs>0 else np.nan
            za=(adef-rm)/rs if np.isfinite(adef) and np.isfinite(rm) and rs>0 else np.nan
            out.append({"game_id":g["game_id"],"game_date":day,
                "home_home_ppg":hppg,"away_road_def_rating":adef,
                "league_home_ppg_mean":hm,"league_home_ppg_sd":hs,
                "league_road_def_mean":rm,"league_road_def_sd":rs,
                "home_ppg_z":zh,"away_def_z":za})
        # Update histories only after every game on the date has been featured.
        for g in games.to_dict("records"):
            ht,at=str(g["home_team_id"]),str(g["away_team_id"])
            hp,ap=float(g["poss_home"]),float(g["poss_away"])
            histories.setdefault(ht,[]).append({"date":day,"pts":float(g["pts_home"]),"opp_pts":float(g["pts_away"]),"opp_poss":ap,"is_home":True})
            histories.setdefault(at,[]).append({"date":day,"pts":float(g["pts_away"]),"opp_pts":float(g["pts_home"]),"opp_poss":hp,"is_home":False})
    return pd.DataFrame(out)

def metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {"n":int(len(e)),"mae":round(float(np.mean(np.abs(e))),4),"rmse":round(float(np.sqrt(np.mean(e*e))),4),"bias":round(float(e.mean()),4)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not BASELINE.exists(): raise FileNotFoundError("Frozen Stage 2 baseline missing")
    baseline=pd.read_csv(BASELINE,parse_dates=["game_date"])
    # Deliberately never merge, inspect or score holdout labels/predictions.
    baseline=baseline[baseline.split.isin(["train","validation"])].copy()
    spine=load_spine()
    features=build_features(spine)
    df_all=baseline.merge(features,on=["game_id","game_date"],how="left",validate="one_to_one")
    feature_cols=["home_ppg_z","away_def_z"]
    coverage={}
    for split in ["train","validation"]:
        part=df_all[df_all.split==split]
        coverage[split]={"baseline_games":int(len(part)),"eligible_games":int(part[feature_cols].notna().all(axis=1).sum()),"excluded_missing_normalized_feature":int((~part[feature_cols].notna().all(axis=1)).sum()),"coverage_rate":round(float(part[feature_cols].notna().all(axis=1).mean()),6) if len(part) else None}
    # Complete-case eligibility is explicit; never impute missing league/venue history.
    df=df_all[df_all[feature_cols].notna().all(axis=1)].copy()
    if df.empty: raise RuntimeError("No games have complete normalized feature history.")
    train=df[df.split=="train"]
    # Both cutoffs are learned only from the training era distribution.
    h_cut=float(train.home_ppg_z.quantile(TRAIN_Q))
    d_cut=float(train.away_def_z.quantile(TRAIN_Q))
    df["p1_era_normalized"]=(df.home_ppg_z>=h_cut)&(df.away_def_z>=d_cut)
    tr=df[df.split=="train"]
    yes=tr[tr.p1_era_normalized].predicted_total-tr[tr.p1_era_normalized].actual_total
    no=tr[~tr.p1_era_normalized].predicted_total-tr[~tr.p1_era_normalized].actual_total
    if len(yes)<MIN_GROUP or len(no)<MIN_GROUP:
        raise RuntimeError(f"Insufficient train groups: feature={len(yes)}, control={len(no)}")
    effect=float(yes.mean()-no.mean())
    rng=np.random.default_rng(SEED)
    diffs=np.empty(BOOTSTRAPS)
    yv=yes.to_numpy(); nv=no.to_numpy()
    for i in range(BOOTSTRAPS):
        diffs[i]=rng.choice(yv,len(yv),replace=True).mean()-rng.choice(nv,len(nv),replace=True).mean()
    ci=[round(float(np.quantile(diffs,.025)),4),round(float(np.quantile(diffs,.975)),4)]
    reports={"coverage":coverage,"thresholds":{"home_ppg_z_train_75th_percentile":round(h_cut,6),"away_road_def_rating_z_train_75th_percentile":round(d_cut,6)},
      "train":{"qualifying":len(yes),"control":len(no),"residual_effect_feature_minus_control":round(effect,4),"bootstrap_95ci":ci},
      "validation":{}}
    adjusted=[]
    for split in ["train","validation"]:
        part=df[df.split==split].copy()
        part["residual"]=part.predicted_total-part.actual_total
        grp=part[part.p1_era_normalized].residual
        ctl=part[~part.p1_era_normalized].residual
        part["adjusted_prediction"]=part.predicted_total-part.p1_era_normalized.astype(float)*effect
        reports[split]={
          "games":len(part),"qualifying":len(grp),"control":len(ctl),
          "feature_minus_control_residual":round(float(grp.mean()-ctl.mean()),4) if len(grp) and len(ctl) else None,
          "baseline_metrics":metrics(part.actual_total,part.predicted_total),
          "adjusted_metrics":metrics(part.actual_total,part.adjusted_prediction),
          "mae_change_adjusted_minus_baseline":round(metrics(part.actual_total,part.adjusted_prediction)["mae"]-metrics(part.actual_total,part.predicted_total)["mae"],4),
          "rmse_change_adjusted_minus_baseline":round(metrics(part.actual_total,part.adjusted_prediction)["rmse"]-metrics(part.actual_total,part.predicted_total)["rmse"],4)}
        adjusted.append(part[["game_id","game_date","split","home_ppg_z","away_def_z","p1_era_normalized","actual_total","predicted_total","adjusted_prediction"]])
    validation=reports["validation"]
    criteria={
      "train_groups_at_least_minimum": bool(len(yes)>=MIN_GROUP and len(no)>=MIN_GROUP),
      "train_bootstrap_interval_excludes_zero": bool(ci[0]>0 or ci[1]<0),
      "validation_groups_at_least_100": bool(validation["qualifying"]>=100 and validation["control"]>=100),
      "validation_effect_same_sign": bool(validation["feature_minus_control_residual"] is not None and np.sign(validation["feature_minus_control_residual"])==np.sign(effect)),
      "validation_mae_improves": bool(validation["mae_change_adjusted_minus_baseline"]<0),
      "validation_rmse_improves": bool(validation["rmse_change_adjusted_minus_baseline"]<0)
    }
    reports["decision"]={"criteria":criteria,"passes_all":all(criteria.values()),"interpretation":"Screening pass only; not deployment approval or proof of profitability."}
    reports["method"]={"feature":"Both home trailing-25 home-game PPG and opponent trailing-25 road defensive rating are at or above their contemporaneous league mean by at least the TRAIN 75th-percentile z-score threshold for that metric.","normalization":"Cross-sectional same-date league mean and population SD across teams with >=5 prior venue-specific games; rates use trailing 25 venue-specific games.","threshold_fit":"TRAIN only; 75th percentile for each z-score, then frozen for validation.","residual":"prediction minus actual; training group-minus-control residual effect is subtracted from predictions for qualifying games.","bootstrap_replicates":BOOTSTRAPS,"seed":SEED,"minimum_train_group":MIN_GROUP,"holdout":"SEALED — not loaded into analysis dataframe, not scored, not used for thresholds or decisions.","same_day":"All games on date are featured before any same-day result enters histories."}
    (OUT/"orbit10_stage3b_pattern1_summary.json").write_text(json.dumps(reports,indent=2))
    pd.concat(adjusted,ignore_index=True).to_csv(OUT/"orbit10_stage3b_pattern1_predictions.csv",index=False)
    print(json.dumps(reports,indent=2))

if __name__=="__main__": main()
