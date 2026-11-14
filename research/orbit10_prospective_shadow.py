#!/usr/bin/env python3
"""ORBIT-10 frozen-model prospective shadow capture and scoring."""
from __future__ import annotations
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import importlib.util
import numpy as np
import pandas as pd
import duckdb

ROOT = Path("outputs/research/orbit10_prospective_shadow")
FORECASTS = ROOT / "forecasts.csv"
RESULTS = ROOT / "results.csv"
LOG = ROOT / "run_log.jsonl"
SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
DATA_BASE = "https://basketball-data.billmill.org"
ALPHA, BETA = -0.805967, -5.836135
HOME_Z, AWAY_Z = 0.657915, 0.732879
ROLLING, HALF_LIFE, PRIOR, MIN_PRIOR = 25, 10.0, 10.0, 5
USER_AGENT = "ORBIT-10-prospective-shadow/1.0"

def get_json(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=45) as response:
        raw = response.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def log_event(data):
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at_utc": datetime.now(timezone.utc).isoformat(), **data}, sort_keys=True) + "\n")

def load_spine():
    spec = importlib.util.spec_from_file_location("orbit10_stage2", "research/orbit10_stage2_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    legacy, _, _ = mod.load_legacy()
    modern, _, _ = mod.load_modern()
    spine = pd.concat([legacy, modern], ignore_index=True)
    # Current-season (2026-27, season end-year 2027) completed games are
    # incorporated only as historical information for future shadow dates.
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        for name in ("team_game_logs", "games"):
            url = f"{DATA_BASE}/nba/stats/{name}/season=2027/data.parquet"
            con.execute(f"CREATE VIEW current_{name} AS SELECT * FROM read_parquet('{url}', hive_partitioning=true, union_by_name=true)")
        current = con.execute("""
          SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
                 t.pts, t.fga, t.fta, t.oreb, t.tov, t.home,
                 g.game_type, g.neutral_site
          FROM current_team_game_logs t JOIN current_games g USING (season, game_id)
          WHERE g.game_type='regular_season' AND COALESCE(g.neutral_site,FALSE)=FALSE
        """).fetchdf()
    finally:
        con.close()
    if not current.empty:
        current["game_date"] = pd.to_datetime(current["game_date"], errors="coerce").dt.normalize()
        for c in ("pts","fga","fta","oreb","tov"):
            current[c] = pd.to_numeric(current[c], errors="coerce")
        current = current.dropna(subset=["game_id","game_date","team_id","opp_team_id","pts","fga","fta","oreb","tov","home"])
        current["poss"] = current.fga + .44*current.fta - current.oreb + current.tov
        if (current.poss <= 0).any():
            raise RuntimeError("Current-season source contains non-positive possessions")
        rows=[]
        for gid,g in current.groupby("game_id"):
            if len(g)!=2 or g.home.astype(bool).sum()!=1:
                raise RuntimeError(f"Current-season game {gid} does not have exactly one home and one away row")
            h=g[g.home.astype(bool)].iloc[0]; a=g[~g.home.astype(bool)].iloc[0]
            rows.append({
              "game_id":f"modern_2027_{gid}","source_game_id":str(gid),"season_end_year":2027,
              "game_date":h.game_date,"home_team_id":str(int(h.team_id)) if float(h.team_id).is_integer() else str(h.team_id),
              "away_team_id":str(int(a.team_id)) if float(a.team_id).is_integer() else str(a.team_id),
              "pts_home":float(h.pts),"pts_away":float(a.pts),"poss_home":float(h.poss),"poss_away":float(a.poss),
              "game_pace_poss":float((h.poss+a.poss)/2),"source":"public_current_season_team_game_logs"})
        spine=pd.concat([spine,pd.DataFrame(rows)],ignore_index=True)
    spine["game_date"]=pd.to_datetime(spine.game_date).dt.normalize()
    spine=spine.sort_values(["game_date","game_id"]).reset_index(drop=True)
    if spine.game_id.duplicated().any():
        raise RuntimeError("Historical/current-season spine contains duplicate game IDs")
    return spine

def wavg(vals):
    x=np.asarray(vals[-ROLLING:],dtype=float)
    age=np.arange(len(x)-1,-1,-1,dtype=float)
    return float(np.average(x,weights=np.exp(-np.log(2)*age/HALF_LIFE)))

def shrink(value,n,league):
    return float((min(n,ROLLING)*value+PRIOR*league)/(min(n,ROLLING)+PRIOR))

def build_history(spine, target_date):
    hist={}
    league_off_pts=league_off_poss=league_def_pts=league_def_poss=0.0
    league_paces=[]
    past=spine[spine.game_date < pd.Timestamp(target_date)]
    for day,games in past.groupby("game_date",sort=True):
        for r in games.to_dict("records"):
            hp,ap=float(r["poss_home"]),float(r["poss_away"])
            hpts,apts=float(r["pts_home"]),float(r["pts_away"])
            ht,at=str(r["home_team_id"]),str(r["away_team_id"])
            h=hist.setdefault(ht,{"off":[],"def":[],"pace":[],"home":[],"road":[],"last_date":day})
            a=hist.setdefault(at,{"off":[],"def":[],"pace":[],"home":[],"road":[],"last_date":day})
            h["off"].append(100*hpts/hp); h["def"].append(100*apts/ap); h["pace"].append((hp+ap)/2)
            h["home"].append({"pts":hpts}); h["last_date"]=day
            a["off"].append(100*apts/ap); a["def"].append(100*hpts/hp); a["pace"].append((hp+ap)/2)
            a["road"].append({"opp_pts":hpts,"opp_poss":hp}); a["last_date"]=day
            league_off_pts += hpts+apts; league_off_poss += hp+ap
            league_def_pts += apts+hpts; league_def_poss += ap+hp
            league_paces.append((hp+ap)/2)
    if not hist:
        raise RuntimeError("No strictly pregame historical team data available")
    league_off=100*league_off_pts/league_off_poss if league_off_poss else 110.0
    league_def=100*league_def_pts/league_def_poss if league_def_poss else league_off
    league_pace=float(np.mean(league_paces)) if league_paces else 98.0
    return hist,league_off,league_def,league_pace

def forecast_game(game,hist,league_off,league_def,league_pace,capture):
    ht,at=str(game["home_id"]),str(game["away_id"])
    if ht not in hist or at not in hist:
        raise RuntimeError(f"Team history missing for {ht} or {at}")
    h,a=hist[ht],hist[at]
    hn,an=len(h["off"]),len(a["off"])
    if min(hn,an)<MIN_PRIOR:
        raise RuntimeError(f"Insufficient team history: home={hn}, away={an}")
    ho,hd,hp=shrink(wavg(h["off"]),hn,league_off),shrink(wavg(h["def"]),hn,league_def),shrink(wavg(h["pace"]),hn,league_pace)
    ao,ad,ap=shrink(wavg(a["off"]),an,league_off),shrink(wavg(a["def"]),an,league_def),shrink(wavg(a["pace"]),an,league_pace)
    expected_pace=(hp+ap)/2
    expected_home_eff=(ho+ad)/2
    expected_away_eff=(ao+hd)/2
    model_a=expected_pace*(expected_home_eff+expected_away_eff)/100
    venue={}
    for team,record in hist.items():
        home=record["home"][-ROLLING:]
        road=record["road"][-ROLLING:]
        if len(home)>=5: venue[(team,"home")]=float(np.mean([x["pts"] for x in home]))
        if len(road)>=5 and sum(x["opp_poss"] for x in road)>0:
            venue[(team,"road")]=100*sum(x["opp_pts"] for x in road)/sum(x["opp_poss"] for x in road)
    hvals=np.array([v for (t,k),v in venue.items() if k=="home"],float)
    rvals=np.array([v for (t,k),v in venue.items() if k=="road"],float)
    if ht not in [t for t,k in venue if k=="home"] or at not in [t for t,k in venue if k=="road"] or len(hvals)<2 or len(rvals)<2:
        raise RuntimeError("Pattern 1 venue history or same-date league normalization unavailable")
    hm,hs=float(hvals.mean()),float(hvals.std(ddof=0))
    rm,rs=float(rvals.mean()),float(rvals.std(ddof=0))
    if hs<=0 or rs<=0: raise RuntimeError("Pattern 1 league standard deviation is zero")
    home_ppg=venue[(ht,"home")]; away_road_def=venue[(at,"road")]
    hz=(home_ppg-hm)/hs; az=(away_road_def-rm)/rs
    pattern=int(hz>=HOME_Z and az>=AWAY_Z)
    global_contribution=-ALPHA
    pattern_contribution=-BETA*pattern
    model_b=model_a+global_contribution+pattern_contribution
    return {
      "game_id":game["game_id"],"game_date":game["game_date"],"scheduled_tipoff_utc":game["tipoff"],
      "capture_timestamp_utc":capture,"home_team_id":ht,"home_team":game["home_name"],
      "away_team_id":at,"away_team":game["away_name"],"home_prior_games":hn,"away_prior_games":an,
      "home_off_eff":ho,"away_off_eff":ao,"home_def_eff":hd,"away_def_eff":ad,
      "home_pace":hp,"away_pace":ap,"expected_possessions":expected_pace,
      "expected_home_efficiency":expected_home_eff,"expected_away_efficiency":expected_away_eff,
      "home_trailing25_home_ppg":home_ppg,"away_trailing25_road_def_rating":away_road_def,
      "league_home_ppg_mean":hm,"league_home_ppg_sd":hs,"league_road_def_mean":rm,"league_road_def_sd":rs,
      "home_ppg_z":hz,"away_road_def_z":az,"pattern1_qualifies":pattern,
      "model_a_prediction":model_a,"global_calibration_contribution":global_contribution,
      "pattern1_contribution":pattern_contribution,"model_b_prediction":model_b,
      "max_history_date":max(pd.Timestamp(h["last_date"]),pd.Timestamp(a["last_date"])).date().isoformat(),
      "score_status":"PENDING","actual_home_points":"","actual_away_points":"","actual_total":""
    }

def capture():
    now=datetime.now(timezone.utc); today=now.date().isoformat()
    payload,sha=get_json(SCOREBOARD)
    games=payload.get("scoreboard",{}).get("games",[])
    eligible=[]
    for g in games:
        if int(g.get("gameStatus",0))!=1: continue
        tip=g.get("gameTimeUTC")
        if not tip: continue
        tip_dt=datetime.fromisoformat(tip.replace("Z","+00:00"))
        if (tip_dt-now).total_seconds()<900: continue
        home,away=g["homeTeam"],g["awayTeam"]
        eligible.append({"game_id":str(g["gameId"]),"game_date":g.get("gameEt","")[:10] or today,
          "tipoff":tip,"home_id":str(home["teamId"]),"away_id":str(away["teamId"]),
          "home_name":home.get("teamName",home.get("teamTricode","")),
          "away_name":away.get("teamName",away.get("teamTricode",""))})
    ROOT.mkdir(parents=True,exist_ok=True)
    existing=pd.read_csv(FORECASTS,dtype={"game_id":str}) if FORECASTS.exists() else pd.DataFrame()
    known=set(existing.game_id.astype(str)) if not existing.empty else set()
    spine=load_spine() if eligible else None
    captured=[]
    for g in eligible:
        if g["game_id"] in known: continue
        target=pd.Timestamp(g["game_date"])
        hist,lo,ld,lp=build_history(spine,target)
        row=forecast_game(g,hist,lo,ld,lp,now.isoformat())
        if not (datetime.fromisoformat(row["capture_timestamp_utc"].replace("Z","+00:00")) <
                datetime.fromisoformat(row["scheduled_tipoff_utc"].replace("Z","+00:00"))):
            raise RuntimeError("Capture timestamp is not before scheduled tip-off")
        captured.append(row)
    if captured:
        pd.DataFrame(captured).to_csv(FORECASTS,index=False,mode="a",header=not FORECASTS.exists())
    log_event({"mode":"capture","scoreboard_sha256":sha,"scoreboard_games":len(games),
      "eligible_pregame_games":len(eligible),"new_forecasts":len(captured),"duplicate_games_skipped":len(eligible)-len(captured),
      "forecast_ledger":str(FORECASTS),"status":"ok" if captured or not eligible else "no_new_forecasts"})
    print(json.dumps({"mode":"capture","eligible":len(eligible),"captured":len(captured),"scoreboard_sha256":sha}))

def score():
    if not FORECASTS.exists():
        log_event({"mode":"score","status":"no_forecast_ledger","scored":0}); print("No forecast ledger yet."); return
    forecasts=pd.read_csv(FORECASTS,dtype={"game_id":str})
    if RESULTS.exists():
        results=pd.read_csv(RESULTS,dtype={"game_id":str})
        known=set(results.game_id.astype(str))
    else: results=pd.DataFrame(); known=set()
    rows=[]
    for r in forecasts.to_dict("records"):
        gid=str(r["game_id"])
        if gid in known: continue
        try:
            data,sha=get_json(BOX_URL.format(game_id=gid))
            box=data.get("game",data.get("boxScore",{}))
            status=int(box.get("gameStatus",0))
            if status!=3: continue
            h=box.get("homeTeam",{}); a=box.get("awayTeam",{})
            hp=float(h["score"]); ap=float(a["score"])
            if hp<0 or ap<0: raise ValueError("Invalid final score")
            rows.append({"game_id":gid,"scored_at_utc":datetime.now(timezone.utc).isoformat(),
              "actual_home_points":hp,"actual_away_points":ap,"actual_total":hp+ap,
              "model_a_prediction":float(r["model_a_prediction"]),"model_b_prediction":float(r["model_b_prediction"]),
              "model_a_error":float(r["model_a_prediction"])-(hp+ap),
              "model_b_error":float(r["model_b_prediction"])-(hp+ap),"boxscore_sha256":sha})
        except Exception as exc:
            log_event({"mode":"score","game_id":gid,"status":"source_error","error":str(exc)})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS,index=False,mode="a",header=not RESULTS.exists())
    log_event({"mode":"score","status":"ok","newly_scored":len(rows),"pending":len(forecasts)-len(known)-len(rows)})
    write_metrics()
    print(json.dumps({"mode":"score","newly_scored":len(rows),"pending":len(forecasts)-len(known)-len(rows)}))

def write_metrics():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not FORECASTS.exists(): return
    f=pd.read_csv(FORECASTS,dtype={"game_id":str})
    if RESULTS.exists():
        r=pd.read_csv(RESULTS,dtype={"game_id":str})
        d=f.merge(r[["game_id","actual_total","actual_home_points","actual_away_points"]],on="game_id",how="inner",validate="one_to_one")
    else: d=f.iloc[0:0].copy()
    def stats(part):
        if part.empty: return {"games":0,"mae":None,"rmse":None,"bias_prediction_minus_actual":None,"within_5_pct":None,"within_10_pct":None}
        actual=part.actual_total.to_numpy(float)
        a=part.model_a_prediction.to_numpy(float)-actual
        b=part.model_b_prediction.to_numpy(float)-actual
        def one(e): return {"mae":round(float(np.mean(np.abs(e))),4),"rmse":round(float(np.sqrt(np.mean(e*e))),4),"bias_prediction_minus_actual":round(float(np.mean(e)),4),"within_5_pct":round(float(np.mean(np.abs(e)<=5)*100),2),"within_10_pct":round(float(np.mean(np.abs(e)<=10)*100),2)}
        return {"games":int(len(part)),"model_a":one(a),"model_b":one(b),"model_b_minus_a":{"mae":round(float(np.mean(np.abs(b))-np.mean(np.abs(a))),4),"rmse":round(float(np.sqrt(np.mean(b*b))-np.sqrt(np.mean(a*a))),4)}}
    started=pd.to_datetime(f.capture_timestamp_utc,utc=True).min() if len(f) else pd.NaT
    elapsed=(datetime.now(timezone.utc).date()-started.date()).days+1 if pd.notna(started) else 0
    by_week={}
    by_season={}
    if not d.empty:
        d["game_date"]=pd.to_datetime(d.game_date,errors="coerce")
        d["week_start"]=d.game_date.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
        by_week={str(k):stats(v) for k,v in d.groupby("week_start",sort=True)}
        if "season_end_year" in d.columns:
            by_season={str(int(k)):stats(v) for k,v in d.groupby("season_end_year",sort=True)}
    if elapsed>=60 and len(d)<500: status="INSUFFICIENT SAMPLE"
    elif elapsed>=60 and len(d)>=500: status="READY FOR FINAL REVIEW"
    else: status="COLLECTING"
    summary={"project":"ORBIT-10 prospective shadow evaluation","status":status,"first_capture_utc":started.isoformat() if pd.notna(started) else None,"elapsed_calendar_days":int(elapsed),"planned_duration_days":60,"minimum_eligible_settled_games":500,"captured_forecasts":int(len(f)),"settled_games":int(len(d)),"pending_games":int(len(f)-len(d)),"overall":stats(d),"by_week_start":by_week,"by_season_end_year":by_season,"model_b_frozen_constants":{"alpha":ALPHA,"beta":BETA,"home_ppg_z_threshold":HOME_Z,"away_road_def_z_threshold":AWAY_Z},"interpretation":"Prospective shadow prediction only; no betting, profitability, causal, or deployment claim."}
    (ROOT/"shadow_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in {"capture","score"}:
        raise SystemExit("Usage: orbit10_prospective_shadow.py capture|score")
    if sys.argv[1]=="capture": capture()
    else: score()

if __name__=="__main__": main()
