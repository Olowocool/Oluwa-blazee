#!/usr/bin/env python3
"""Fetch point-in-time team/player inputs for the standalone NBA Excel model.

Uses public completed-game parquet partitions and ESPN's public roster endpoint.
All historical queries are strictly before the target game date. Injury and
minute estimates are labelled as source status / recent-minute estimates,
not official pregame projections.
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import duckdb, pandas as pd, requests

BASE="https://basketball-data.billmill.org/nba"
ESPN="https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def parquet_url(source,table,season):
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"

def fetch_json(url):
    r=requests.get(url,timeout=30,headers={"User-Agent":"NBA-Total-Points-Research-Workbook/1.0"})
    r.raise_for_status()
    return r.json()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date",required=True)
    ap.add_argument("--output",default="excel_totals_model/source_inputs.json")
    a=ap.parse_args()
    target=pd.Timestamp(a.date).normalize()
    season=2027 if target.month>=7 else target.year
    seasons=[season-1,season]
    con=duckdb.connect(config={"memory_limit":"1GB","threads":"2"})
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        team_paths=",".join("'"+parquet_url("stats","team_game_logs",s)+"'" for s in seasons)
        game_paths=",".join("'"+parquet_url("stats","games",s)+"'" for s in seasons)
        player_paths=",".join("'"+parquet_url("stats","player_game_logs",s)+"'" for s in seasons)
        team=con.execute(f"""
          SELECT t.game_date,t.team_id,t.pts,t.poss,t.pace,t.off_rating,t.def_rating
          FROM read_parquet([{team_paths}],hive_partitioning=true,union_by_name=true) t
          JOIN read_parquet([{game_paths}],hive_partitioning=true,union_by_name=true) g
            ON CAST(t.game_id AS VARCHAR)=CAST(g.game_id AS VARCHAR)
          WHERE CAST(t.game_date AS DATE) < DATE '{target.date()}'
            AND lower(CAST(g.game_type AS VARCHAR))='regular_season'
        """).fetchdf()
        player=con.execute(f"""
          SELECT game_date,team_id,player_id,min,pts
          FROM read_parquet([{player_paths}],hive_partitioning=true,union_by_name=true)
          WHERE CAST(game_date AS DATE) < DATE '{target.date()}'
        """).fetchdf()
        names=con.execute(f"""
          SELECT player_id,name FROM read_parquet([
            '{parquet_url("stats","player_seasons",season)}'
          ],hive_partitioning=true,union_by_name=true)
        """).fetchdf()
    finally:
        con.close()
    if team.empty:
        raise RuntimeError("No completed pre-target team-game history was returned")
    for col in ["pts","poss","pace","off_rating","def_rating"]:
        team[col]=pd.to_numeric(team[col],errors="coerce")
    team["game_date"]=pd.to_datetime(team.game_date,errors="coerce")
    team=team.dropna(subset=["game_date","team_id","pts","poss","pace","off_rating","def_rating"])
    team=team[team.game_date < target].sort_values("game_date")
    team_rows=[]
    for tid,g in team.groupby("team_id"):
        recent=g.tail(10)
        if len(recent)<3: continue
        weights=[0.90**i for i in range(len(recent)-1,-1,-1)]
        w=pd.Series(weights,index=recent.index)
        team_rows.append({"team_id":str(tid),"games_used":len(recent),
          "ortg":float((recent.off_rating*w).sum()/w.sum()),
          "drtg":float((recent.def_rating*w).sum()/w.sum()),
          "pace":float((recent.pace*w).sum()/w.sum()),
          "as_of":str(recent.game_date.max().date())})
    player["game_date"]=pd.to_datetime(player.game_date,errors="coerce")
    player["min"]=pd.to_numeric(player["min"],errors="coerce").fillna(0)
    player["pts"]=pd.to_numeric(player["pts"],errors="coerce").fillna(0)
    player=player[(player.game_date<target)&(player["min"]>0)]
    name_map={str(r.player_id):str(r.name) for r in names.itertuples()}
    player_rows=[]
    team_map={str(r["team_id"]):r for r in team_rows}
    for tid,g in player.groupby(player.team_id.astype(str)):
        recent_games=sorted(g.game_date.unique())[-10:]
        recent=g[g.game_date.isin(recent_games)]
        avg=recent.groupby("player_id").agg(minutes=("min","mean"),points=("pts","mean"),apps=("game_date","nunique")).reset_index()
        if avg.empty: continue
        avg["ppm"]=avg.apply(lambda r: float(r.points/r.minutes) if r.minutes>0 else 0,axis=1)
        replacement=float(avg[avg.minutes>0].ppm.median()) if (avg.minutes>0).any() else 0.0
        active=avg[avg.minutes>0].copy()
        active["minutes"]=active.minutes.clip(upper=48)
        total=float(active.minutes.sum())
        if total>0: active["expected_minutes"]=active.minutes*240.0/total
        else: active["expected_minutes"]=0.0
        active["expected_minutes"]=active.expected_minutes.clip(upper=48)
        # Reconcile to 240 while respecting the 48-minute per-player cap.
        for _ in range(8):
            rem=240.0-float(active.expected_minutes.sum())
            room=active.expected_minutes<48
            if abs(rem)<0.01 or not room.any(): break
            active.loc[room,"expected_minutes"] += rem*active.loc[room,"minutes"]/active.loc[room,"minutes"].sum()
            active["expected_minutes"]=active.expected_minutes.clip(upper=48)
        for r in active.itertuples():
            player_rows.append({"team_id":tid,"player_id":str(r.player_id),
              "player":name_map.get(str(r.player_id),str(r.player_id)),
              "recent_games":int(r.apps),"recent_avg_minutes":round(float(r.minutes),2),
              "expected_minutes_estimate":round(float(r.expected_minutes),2),
              "points_per_minute":round(float(r.ppm),4),
              "impact_pts_per_100_poss":round((float(r.ppm)-replacement)*48.0,3),
              "availability":"UNKNOWN — verify official report","status_source":"Recent player-game history only",
              "history_as_of":str(max(recent_games).date())})
    payload={"schema":"nba-excel-inputs-v1","target_date":str(target.date()),
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "team_inputs":team_rows,"player_inputs":player_rows,
      "source_notes":["Team ratings and pace: last 10 regular-season team games, recency weighted, strict pre-target cutoff.",
      "Player minutes: average over up to 10 most recent appearances, normalized to a 240-minute team rotation with a 48-minute cap; estimate, not a validated rotation forecast.",
      "Player impact: recent points-per-minute relative to that team's median player rate; proxy only, not RAPM or causal impact.",
      "Availability is UNKNOWN until an official game-date injury report is attached. Do not treat roster presence as availability."]}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"target_date":str(target.date()),"teams":len(team_rows),"players":len(player_rows),
      "latest_team_history":max(x["as_of"] for x in team_rows) if team_rows else None,
      "latest_player_history":max(x["history_as_of"] for x in player_rows) if player_rows else None,
      "availability":"not verified; forecasts must remain provisional"},indent=2))
if __name__=="__main__": main()
