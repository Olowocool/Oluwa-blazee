"""Research-only inventory and integrity audit for historical NBA totals data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/research/historical_data_foundation")
OUT.mkdir(parents=True, exist_ok=True)
FILES = [
 "data/historical_nba_scores.csv", "data/game.csv", "data/nba_games.csv",
 "outputs/training_dataset.parquet",
 "research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv",
 "research/data/score_history_patch_2026-10-21_to_2026-10-29.csv",
 "research/data/score_history_patch_2026-10-30.csv",
]
DATE_NAMES = ["game_date", "date", "game_date_est", "commence_time_utc", "start_time"]
ID_NAMES = ["game_id", "id", "event_id"]
HOME = ["home_team", "team_name_home", "home_team_name"]
AWAY = ["away_team", "team_name_away", "away_team_name"]
HS = ["actual_home_points", "home_score", "pts_home", "home_points"]
AS = ["actual_away_points", "away_score", "pts_away", "away_points"]

def col(cols, names):
    m = {x.casefold(): x for x in cols}
    return next((m[n.casefold()] for n in names if n.casefold() in m), None)

def profile(path, df):
    cols = list(df.columns)
    dc, ic = col(cols, DATE_NAMES), col(cols, ID_NAMES)
    hc, ac, hsc, asc = col(cols, HOME), col(cols, AWAY), col(cols, HS), col(cols, AS)
    dates = pd.to_datetime(df[dc], errors="coerce", utc=True) if dc else pd.Series(dtype="datetime64[ns, UTC]")
    p = {
      "file": path, "rows": int(len(df)), "columns_count": len(cols), "columns": cols,
      "file_bytes": (Path(path).stat().st_size), "null_cells": int(df.isna().sum().sum()),
      "null_pct": round(float(df.isna().sum().sum()/max(1,df.size)*100),4),
      "date_column": dc, "first_date": dates.min().isoformat() if len(dates) and dates.notna().any() else None,
      "last_date": dates.max().isoformat() if len(dates) and dates.notna().any() else None,
      "invalid_date_rows": int(dates.isna().sum()) if dc else None,
      "id_column": ic, "unique_ids": int(df[ic].nunique(dropna=True)) if ic else None,
      "duplicate_id_rows": int(df[ic].duplicated(keep=False).sum()) if ic else None,
      "home_team_column": hc, "away_team_column": ac, "home_score_column": hsc, "away_score_column": asc,
    }
    if hc and ac:
      p["same_team_home_away_rows"] = int((df[hc].astype("string").str.strip().str.casefold()==df[ac].astype("string").str.strip().str.casefold()).sum())
    if hsc and asc:
      h,a = pd.to_numeric(df[hsc],errors="coerce"),pd.to_numeric(df[asc],errors="coerce")
      total=h+a
      p["score_rows_complete"]=int((h.notna()&a.notna()).sum())
      p["score_total_min"]=float(total.min()) if total.notna().any() else None
      p["score_total_max"]=float(total.max()) if total.notna().any() else None
      tc=col(cols,["actual_total","total_points","total_score"])
      if tc:
        t=pd.to_numeric(df[tc],errors="coerce")
        p["total_score_mismatch_rows"]=int((t.notna()&total.notna()&~np.isclose(t,total,atol=.01)).sum())
    poss={s:[col(cols,[f"{x}_{s}"]) for x in ["fga","fta","oreb","tov"]] for s in ["home","away"]}
    if all(poss["home"]) and all(poss["away"]):
      est={}
      for side,fields in poss.items():
        fga,fta,oreb,tov=[pd.to_numeric(df[x],errors="coerce") for x in fields]
        est[side]=fga+.44*fta-oreb+tov
      ok=est["home"].notna()&est["away"].notna()
      p["true_possession_formula_rows_both_teams"]=int(ok.sum())
      p["mean_home_possessions_estimate"]=float(est["home"][ok].mean()) if ok.any() else None
      p["mean_away_possessions_estimate"]=float(est["away"][ok].mean()) if ok.any() else None
      p["mean_abs_home_away_possession_gap"]=float((est["home"][ok]-est["away"][ok]).abs().mean()) if ok.any() else None
    return p

def main():
  profiles=[]; frames={}
  for path in FILES:
    f=Path(path)
    if not f.exists():
      profiles.append({"file":path,"exists":False}); continue
    df=pd.read_parquet(f) if path.endswith(".parquet") else pd.read_csv(f,low_memory=False)
    profiles.append(profile(path,df)); frames[path]=df
  cross={}
  x,y="data/historical_nba_scores.csv","data/game.csv"
  if x in frames and y in frames:
    a,b=frames[x],frames[y]
    ad,bd=col(a.columns,DATE_NAMES),col(b.columns,DATE_NAMES)
    ah,bh=col(a.columns,HOME),col(b.columns,HOME)
    aa,ba=col(a.columns,AWAY),col(b.columns,AWAY)
    if all([ad,bd,ah,bh,aa,ba]):
      def key(df,d,h,a):
        dt=pd.to_datetime(df[d],errors="coerce").dt.strftime("%Y-%m-%d")
        hn=df[h].astype("string").str.lower().str.replace(r"[^a-z0-9]","",regex=True)
        an=df[a].astype("string").str.lower().str.replace(r"[^a-z0-9]","",regex=True)
        return dt+"|"+hn+"|"+an
      ka,kb=key(a,ad,ah,aa),key(b,bd,bh,ba)
      cross={"method":"normalized date + ordered home/away team names","score_rows":len(ka),"game_csv_rows":len(kb),
       "score_unique_keys":int(ka.nunique()),"game_csv_unique_keys":int(kb.nunique()),
       "matched_unique_keys":len(set(ka.dropna())&set(kb.dropna())),
       "score_only_unique_keys":len(set(ka.dropna())-set(kb.dropna())),
       "game_csv_only_unique_keys":len(set(kb.dropna())-set(ka.dropna()))}
    else: cross={"status":"not_computable","reason":"date/home/away columns not identified"}
  else: cross={"status":"not_computable","reason":"principal source missing"}
  summary={"audit":"historical_data_foundation","scope":"repository datasets only; no production data","files":profiles,"cross_source_match":cross}
  (OUT/"historical_data_foundation.json").write_text(json.dumps(summary,indent=2,default=str))
  pd.DataFrame([{k:v for k,v in p.items() if k!="columns"}|{"columns":", ".join(p.get("columns",[]))} for p in profiles]).to_csv(OUT/"dataset_inventory.csv",index=False)
  lines=["# Historical data foundation audit","","Research-only inventory and integrity checks. No production/live/paper-trading data was read or changed.","",
   "| Dataset | Rows | Date range | Columns | Null cells | Duplicate IDs |","|---|---:|---|---:|---:|---:|"]
  for p in profiles:
    if not p.get("exists",True): lines.append(f"| {p['file']} | missing | — | — | — | — |")
    else: lines.append(f"| {p['file']} | {p.get('rows',0):,} | {p.get('first_date') or 'unknown'} to {p.get('last_date') or 'unknown'} | {p.get('columns_count',0)} | {p.get('null_cells',0):,} | {p.get('duplicate_id_rows') if p.get('duplicate_id_rows') is not None else 'n/a'} |")
  lines += ["","## Cross-source matchability","",json.dumps(cross,indent=2),"","## Full per-file diagnostics","",json.dumps(profiles,indent=2,default=str),"",
   "## Interpretation rules","",
   "- Row count is not necessarily unique game count; team-game tables may have two rows per game.",
   "- Missing IDs, ambiguous orientation, invalid dates, duplicate keys, and score mismatches must be resolved or excluded.",
   "- Possessions are calculated only where FGA, FTA, OREB, and TOV exist for both teams.",
   "- Historical statistics are predictors only when their observation time precedes tipoff; final box scores do not prove point-in-time availability.",
   "- Opening/closing market endpoints are not timestamped entry prices unless an actual contemporaneous price timestamp exists.",""]
  (OUT/"historical_data_foundation.md").write_text("\\n".join(lines))
  print(json.dumps({"report_dir":str(OUT),"inventory":[{k:p.get(k) for k in ["file","rows","first_date","last_date","columns_count","null_cells","duplicate_id_rows"]} for p in profiles],"cross_source_match":cross},indent=2,default=str))

if __name__=="__main__": main()
