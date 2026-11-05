#!/usr/bin/env python3
"""Objective 6: conservative Over/Under/No Bet selection and fractional staking.

Uses Objective 5 prequential development predictions to establish whether the
residual model beats the no-vig market, then applies fixed rules to the
untouched Objective 5 holdout. It never reads holdout outcomes for decisions.
No bets are placed and no realized ROI is calculated.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

DEFAULT_INPUT="outputs/research/objective5_market_comparison"
DEFAULT_OUTPUT="outputs/research/objective6_bet_selection"
MIN_CALIBRATION_GAMES=100
MIN_DEVELOPMENT_PROBABILITY_GAMES=100
MIN_DEVELOPMENT_DATES=10
MIN_EV=0.02
KELLY_FRACTION=0.25
MAX_STAKE_FRACTION=0.005
MAX_DAILY_EXPOSURE_FRACTION=0.02
WILSON_Z=1.96
BOOTSTRAP_RESAMPLES=5000
BOOTSTRAP_SEED=61

def wilson_lower_bound(probability: float, sample_size: int, z: float=WILSON_Z)->float:
    p=float(probability); n=int(sample_size)
    if not np.isfinite(p) or not 0<=p<=1 or n<1 or z<=0:
        raise ValueError("Wilson inputs must be finite, p in [0,1], n >= 1 and z > 0.")
    z2=z*z; denom=1+z2/n
    center=p+z2/(2*n)
    margin=z*np.sqrt(p*(1-p)/n+z2/(4*n*n))
    return float(max(0.0,(center-margin)/denom))

def _brier_difference_by_date(frame:pd.DataFrame,candidate_col:str,market_col:str,
    n_resamples:int=BOOTSTRAP_RESAMPLES,seed:int=BOOTSTRAP_SEED)->dict[str,Any]:
    d=frame.loc[frame[candidate_col].notna() & frame[market_col].notna()
        & frame.actual_total.notna() & frame.entry_total.notna()].copy()
    d=d.loc[d.actual_total!=d.entry_total]
    if d.empty: return {"games":0,"game_dates":0,"difference":None,"ci_95":None}
    p=d[candidate_col].to_numpy(float)
    push_col=candidate_col.replace("_over_probability","_push_probability")
    if push_col in d: p=p/np.maximum(1-d[push_col].to_numpy(float),1e-12)
    y=(d.actual_total.to_numpy(float)>d.entry_total.to_numpy(float)).astype(float)
    d["loss_difference"]=(p-y)**2-(d[market_col].to_numpy(float)-y)**2
    clusters=[g.loss_difference.to_numpy(float) for _,g in d.groupby("game_date",sort=True)]
    observed=float(d.loss_difference.mean()); rng=np.random.default_rng(seed)
    boot=np.empty(n_resamples)
    for i in range(n_resamples):
        picks=rng.integers(0,len(clusters),size=len(clusters))
        boot[i]=np.concatenate([clusters[j] for j in picks]).mean()
    lo,hi=np.quantile(boot,[0.025,0.975])
    return {"games":int(len(d)),"game_dates":len(clusters),"difference":observed,
        "ci_95":[float(lo),float(hi)],
        "interpretation":"Negative favours residual model; CI upper bound below zero is required."}

def development_gate(frame:pd.DataFrame,data_quality:dict[str,Any])->dict[str,Any]:
    dev=frame.loc[frame.evaluation_split.eq("development")].copy()
    dev=dev.loc[dev.residual_model_over_probability.notna()
        & dev.residual_model_under_probability.notna()
        & dev.residual_model_push_probability.notna()]
    n_dates=int(dev.game_date.nunique()) if not dev.empty else 0
    readiness=data_quality.get("readiness",{})
    calibration_n=int(readiness.get("development_walk_forward_predictions",0) or 0)
    comparison=_brier_difference_by_date(dev,"residual_model_over_probability","over_probability_no_vig") if not dev.empty else {
        "games":0,"game_dates":0,"difference":None,"ci_95":None}
    reasons=[]; ci=comparison.get("ci_95")
    if data_quality.get("status")!="completed": reasons.append("Objective 5 data-quality status is not completed.")
    if calibration_n<MIN_CALIBRATION_GAMES: reasons.append("Fewer than 100 prior OOS residual-model calibration errors.")
    if len(dev)<MIN_DEVELOPMENT_PROBABILITY_GAMES: reasons.append("Fewer than 100 calibrated development games.")
    if n_dates<MIN_DEVELOPMENT_DATES: reasons.append("Fewer than 10 calibrated development dates.")
    if not ci or ci[1] is None or not float(ci[1])<0:
        reasons.append("Development Brier improvement over no-vig market is not established at the 95% date-cluster interval.")
    return {"passed":not reasons,"reasons":reasons,"calibration_errors_available":calibration_n,
        "development_probability_games":int(len(dev)),"development_probability_dates":n_dates,
        "development_brier_vs_market":comparison}

def _candidate_side(row:pd.Series,calibration_n:int,min_ev:float)->dict[str,Any]:
    po,pu,pp=[float(row[k]) for k in ("residual_model_over_probability","residual_model_under_probability","residual_model_push_probability")]
    oo,uo=float(row.entry_over_odds_decimal),float(row.entry_under_odds_decimal)
    vals=[po,pu,pp,oo,uo]
    if not np.isfinite(vals).all() or min(po,pu,pp)<0: raise ValueError("Invalid probability/price input.")
    if oo<=1 or uo<=1: raise ValueError("Decimal odds must exceed 1.")
    if not np.isclose(po+pu+pp,1,atol=1e-6): raise ValueError("Over/Under/Push probabilities must sum to one.")
    nonpush=po+pu
    if nonpush<=0: return {"side":"No Bet","reason":"No non-push probability mass."}
    choices=[]
    for side,pwin,odds in (("Over",po/nonpush,oo),("Under",pu/nonpush,uo)):
        effective_n=max(1,int(np.floor(calibration_n*nonpush)))
        lower=wilson_lower_bound(pwin,effective_n)
        conservative_ev=nonpush*(lower*odds-1)
        b=odds-1
        kelly=max(0.0,(b*lower-(1-lower))/b)
        choices.append({"side":side,"conditional_win_probability":pwin,
            "wilson_lower_probability":lower,"conservative_ev_per_unit":conservative_ev,
            "decimal_odds":odds,"stake_fraction":min(MAX_STAKE_FRACTION,KELLY_FRACTION*kelly)})
    best=max(choices,key=lambda x:x["conservative_ev_per_unit"])
    if best["conservative_ev_per_unit"]<=min_ev:
        return {"side":"No Bet","reason":"No side clears conservative minimum EV.","best_candidate":best}
    if best["stake_fraction"]<=0: return {"side":"No Bet","reason":"Conservative Kelly stake is zero.","best_candidate":best}
    return {"reason":"Passed development gate and conservative EV/stake rules.",**best}

def select_holdout_bets(frame:pd.DataFrame,gate:dict[str,Any],calibration_n:int,
    bankroll_units:float=100.0,min_ev:float=MIN_EV)->pd.DataFrame:
    if not np.isfinite(bankroll_units) or bankroll_units<=0: raise ValueError("Bankroll units must be positive.")
    holdout=frame.loc[frame.evaluation_split.eq("reserved_holdout")].copy(); output=[]
    for _,row in holdout.iterrows():
        base={"game_date":str(pd.Timestamp(row.game_date).date()),
            "commence_time_utc":str(row.commence_time_utc),"home_team":row.get("home_team",""),
            "away_team":row.get("away_team",""),"entry_total":float(row.entry_total),
            "entry_over_odds_decimal":float(row.entry_over_odds_decimal),
            "entry_under_odds_decimal":float(row.entry_under_odds_decimal),"evaluation_split":"reserved_holdout"}
        if not gate["passed"]: decision={"side":"No Bet","reason":"Development evidence gate failed."}
        elif row.get("residual_model_status")!="frozen_holdout_scored":
            decision={"side":"No Bet","reason":"No frozen holdout model prediction."}
        elif any(pd.isna(row.get(k)) for k in ("residual_model_over_probability","residual_model_under_probability","residual_model_push_probability")):
            decision={"side":"No Bet","reason":"Calibrated holdout probabilities unavailable."}
        else: decision=_candidate_side(row,calibration_n,min_ev)
        frac=float(decision.get("stake_fraction",0.0))
        base.update({"decision":decision.get("side","No Bet"),"decision_reason":decision["reason"],
            "conditional_win_probability":decision.get("conditional_win_probability"),
            "wilson_lower_probability":decision.get("wilson_lower_probability"),
            "conservative_ev_per_unit":decision.get("conservative_ev_per_unit"),
            "stake_fraction_of_bankroll":frac,"stake_units_before_daily_cap":frac*bankroll_units})
        output.append(base)
    result=pd.DataFrame(output)
    if result.empty:
        # Keep a header-only CSV so Objective 7 can report an empty holdout
        # as insufficient evidence instead of failing to parse an empty file.
        return pd.DataFrame(columns=[
            "game_date","commence_time_utc","home_team","away_team","entry_total",
            "entry_over_odds_decimal","entry_under_odds_decimal","evaluation_split",
            "decision","decision_reason","conditional_win_probability",
            "wilson_lower_probability","conservative_ev_per_unit",
            "stake_fraction_of_bankroll","stake_units_before_daily_cap","stake_units",
        ])
    result["stake_units"]=0.0
    for _,indexes in result.groupby("game_date",sort=True).groups.items():
        ranked=result.loc[list(indexes)].sort_values("conservative_ev_per_unit",ascending=False,na_position="last")
        remaining=MAX_DAILY_EXPOSURE_FRACTION*bankroll_units
        for idx in ranked.index:
            allowed=min(float(result.at[idx,"stake_units_before_daily_cap"]),remaining)
            result.at[idx,"stake_units"]=allowed
            result.at[idx,"stake_fraction_of_bankroll"]=allowed/bankroll_units
            remaining-=allowed
        idxs=list(indexes); no_stake=result.loc[idxs,"stake_units"]<=0
        result.loc[idxs,"decision"]=np.where(no_stake,"No Bet",result.loc[idxs,"decision"])
        result.loc[idxs,"decision_reason"]=np.where(no_stake,"No positive stake after daily exposure cap.",result.loc[idxs,"decision_reason"])
    return result

def run(args:argparse.Namespace)->dict[str,Any]:
    inp=Path(args.input_dir)
    gp=inp/"objective5_market_comparison_game_results.csv"
    sp=inp/"objective5_market_comparison_summary.json"
    qp=inp/"objective5_market_comparison_data_quality.json"
    missing=[str(p) for p in (gp,sp,qp) if not p.exists()]
    if missing: raise FileNotFoundError("Run Objective 5 first; missing: "+", ".join(missing))
    summary=json.loads(sp.read_text(encoding="utf-8"))
    quality=json.loads(qp.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or gp.stat().st_size == 0:
        # Fail closed before parsing an empty/insufficient Objective 5 output.
        # A header-only decisions file allows Objective 7 to report zero bets cleanly.
        gate={
            "passed":False,
            "reasons":[
                f"Objective 5 status is {summary.get('status')!r}; no candidate bets may be generated."
            ],
            "calibration_errors_available":int(
                quality.get("readiness",{}).get("development_walk_forward_predictions",0) or 0
            ),
            "development_probability_games":0,
            "development_probability_dates":0,
            "development_brier_vs_market":{"games":0,"game_dates":0,"difference":None,"ci_95":None},
        }
        decisions=pd.DataFrame(columns=[
            "game_date","commence_time_utc","home_team","away_team","entry_total",
            "entry_over_odds_decimal","entry_under_odds_decimal","evaluation_split",
            "decision","decision_reason","conditional_win_probability",
            "wilson_lower_probability","conservative_ev_per_unit",
            "stake_fraction_of_bankroll","stake_units_before_daily_cap","stake_units",
        ])
    else:
        frame=pd.read_csv(gp)
        gate=development_gate(frame,quality); n=gate["calibration_errors_available"]
        decisions=select_holdout_bets(frame,gate,n,args.bankroll_units,args.minimum_ev)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dp=out/"objective6_candidate_decisions.csv"; rp=out/"objective6_bet_selection_summary.json"
    decisions.to_csv(dp,index=False)
    result={"status":"candidate_decisions_generated" if gate["passed"] else "no_bet_evidence_gate_failed",
        "objective5_status":summary.get("status"),"development_gate":gate,
        "decision_rules":{"minimum_conservative_ev_per_unit":args.minimum_ev,
        "uncertainty":"95% Wilson lower bound from prior development OOS errors; conditional on non-push outcomes.",
        "kelly_fraction":KELLY_FRACTION,"maximum_stake_per_game_pct_bankroll":100*MAX_STAKE_FRACTION,
        "maximum_daily_exposure_pct_bankroll":100*MAX_DAILY_EXPOSURE_FRACTION,
        "bankroll_units":args.bankroll_units,"selection_period":"reserved_holdout; outcomes not read",
        "evidence_period":"prequential development only"},
        "holdout_games":len(decisions),"over_decisions":int(decisions.decision.eq("Over").sum()) if len(decisions) else 0,
        "under_decisions":int(decisions.decision.eq("Under").sum()) if len(decisions) else 0,
        "no_bet_decisions":int(decisions.decision.eq("No Bet").sum()) if len(decisions) else 0,
        "total_stake_units":float(decisions.stake_units.sum()) if len(decisions) else 0.0,
        "realized_roi_calculated":False,"bets_placed":False,"outputs":[str(dp),str(rp)],
        "warning":"Research-only candidate decisions. Objective 7 must independently validate frozen decisions."}
    rp.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str)); return result

def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",default=DEFAULT_INPUT); p.add_argument("--output-dir",default=DEFAULT_OUTPUT)
    p.add_argument("--bankroll-units",type=float,default=100.0); p.add_argument("--minimum-ev",type=float,default=MIN_EV)
    return p.parse_args()

if __name__=="__main__": run(parse_args())
