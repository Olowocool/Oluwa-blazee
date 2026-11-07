#!/usr/bin/env python3
"""Fetch strictly pre-target team/player history and timestamped NBA injury-report snapshots."""
from __future__ import annotations
import argparse, json, io, re, unicodedata, difflib
from datetime import datetime, timezone
from pathlib import Path
import duckdb, pandas as pd, requests, pdfplumber

BASE="https://basketball-data.billmill.org/nba"
ESPN="https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
INJURY_RELEASE="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_injury_report/injury_report_{asset_year}.parquet"

def parquet_url(source,table,season):
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date",required=True)
    ap.add_argument("--output",default="excel_totals_model/source_inputs.json")
    a=ap.parse_args()
    target=pd.Timestamp(a.date).normalize()
    season=target.year+1 if target.month>=7 else target.year
    seasons=[season-1,season]
    con=duckdb.connect(config={"memory_limit":"1GB","threads":"2"})
    injury_rows=[]; injury_source_status="UNAVAILABLE — official report archive could not be read"
    injury_source_url=INJURY_RELEASE.format(asset_year=season)
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
          SELECT p.game_date,p.team_id,p.player_id,p.min,p.pts
          FROM read_parquet([{player_paths}],hive_partitioning=true,union_by_name=true) p
          JOIN read_parquet([{game_paths}],hive_partitioning=true,union_by_name=true) g
            ON CAST(p.game_id AS VARCHAR)=CAST(g.game_id AS VARCHAR)
          WHERE CAST(p.game_date AS DATE) < DATE '{target.date()}'
            AND lower(CAST(g.game_type AS VARCHAR))='regular_season'
        """).fetchdf()
        names=con.execute(f"""
          SELECT player_id,name FROM read_parquet([
            '{parquet_url("stats","player_seasons",season)}',
            '{parquet_url("stats","player_seasons",season-1)}'
          ],hive_partitioning=true,union_by_name=true)
        """).fetchdf()
        team_names=con.execute(f"""
          SELECT team_id,full_name,nba_abbrev AS abbreviation FROM read_parquet([
            '{parquet_url("stats","team_seasons",season)}'
          ],hive_partitioning=true,union_by_name=true)
        """).fetchdf()
        # First use the NBA's own current-season page and its linked PDF snapshots.
        # The PDF filename carries the report issue time; preserve that source time.
        try:
            season_page=f"https://official.nba.com/nba-injury-report-{season-1}-{str(season)[-2:]}-season/"
            page=requests.get(season_page,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
            page.raise_for_status()
            hrefs=re.findall(r"""href=["']([^"']+Injury-Report_[^"']+\.pdf)["']""",page.text,re.I)
            candidates=[]
            for href in hrefs:
                url=href.replace("&amp;","&")
                match=re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})_(\d{2})(AM|PM)",url,re.I)
                if not match or match.group(1)!=target.strftime("%Y-%m-%d"): continue
                hour=int(match.group(2))%12
                if match.group(4).upper()=="PM": hour+=12
                minute=int(match.group(3))
                candidates.append(((hour,minute),url))
            if not candidates:
                # The official page sometimes omits clickable anchors from its
                # server-rendered HTML. Probe the documented 15-minute PDF
                # filename grid for the most recent six hours of this game date.
                for minutes_back in range(345,-1,-15):
                    total_minutes=minutes_back
                    hh=(total_minutes//60)%24
                    mm=total_minutes%60
                    suffix="AM" if hh<12 else "PM"
                    clock_hour=hh%12 or 12
                    probe=f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{target:%Y-%m-%d}_{clock_hour:02d}_{mm:02d}{suffix}.pdf"
                    try:
                        test=requests.get(probe,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
                        if test.status_code==200 and test.content[:4]==b"%PDF":
                            candidates.append(((hh,mm),probe))
                            break
                    except requests.RequestException:
                        continue
            if candidates:
                _,official_pdf=max(candidates,key=lambda x:x[0])
                response=requests.get(official_pdf,timeout=45,headers={"User-Agent":"Mozilla/5.0"})
                response.raise_for_status()
                with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                    for page_no,pdf_page in enumerate(pdf.pages,1):
                        tables=pdf_page.extract_tables() or []
                        if not tables:
                            tables=pdf_page.extract_tables(table_settings={"vertical_strategy":"text","horizontal_strategy":"text","snap_tolerance":3,"join_tolerance":3,"min_words_vertical":1,"min_words_horizontal":1}) or []
                        for table in tables:
                            current_date=""; current_matchup=""; current_team=""
                            for cells in table:
                                cells=[str(x or "").strip() for x in cells]
                                if len(cells)<7: continue
                                if "Game Date" in " ".join(cells): continue
                                gd,gt,mu,tm,pn,st,reason=(cells+[""]*7)[:7]
                                if gd: current_date=gd
                                if mu: current_matchup=mu
                                if tm: current_team=tm
                                if not current_team: continue
                                if not pn and not st and "NOT YET SUBMITTED" not in reason.upper(): continue
                                parsed_date=pd.to_datetime(current_date,errors="coerce")
                                if pd.isna(parsed_date) or parsed_date.date()!=target.date(): continue
                                injury_rows.append({"report_datetime":re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})_(\d{2})(AM|PM)",official_pdf,re.I).group(0).replace("Injury-Report_","").replace("_"," ") if "Injury-Report_" in official_pdf else "",
                                  "game_date":str(parsed_date.date()),"matchup":current_matchup,"team":current_team,
                                  "player_report_name":pn,"status":st,"reason":reason,"source_page":page_no})
                if injury_rows:
                    injury_source_url=official_pdf
                    injury_source_status=f"NBA OFFICIAL PDF — {len(injury_rows)} rows; source {official_pdf.rsplit('/',1)[-1]}"
            else:
                injury_source_status="NO NBA OFFICIAL PDF LINK FOR TARGET DATE — availability remains unknown"
        except Exception as exc:
            injury_source_status=f"NBA OFFICIAL PDF FETCH/PARSE FAILED — {type(exc).__name__}: {exc}"
        # Fallback archive provides a timestamped cross-check, but is not allowed
        # to overwrite direct official PDF rows or imply a player is available.
        if not injury_rows:
            try:
                injury=con.execute(f"""
                  SELECT report_datetime,game_date,report_matchup,team,
                         report_player_name,current_status,reason
                  FROM read_parquet('{injury_source_url}')
                  WHERE CAST(game_date AS DATE)=DATE '{target.date()}'
                """).fetchdf()
                if not injury.empty:
                    injury["report_datetime"]=pd.to_datetime(injury["report_datetime"],errors="coerce")
                    injury=injury.dropna(subset=["report_datetime"])
                    injury=injury.sort_values("report_datetime").drop_duplicates(
                        subset=["game_date","report_matchup","team","report_player_name"],keep="last")
                    for r in injury.to_dict("records"):
                        injury_rows.append({"report_datetime":str(r["report_datetime"]),
                          "game_date":str(pd.Timestamp(r["game_date"]).date()),"matchup":str(r["report_matchup"] or ""),
                          "team":str(r["team"] or ""),"player_report_name":str(r["report_player_name"] or ""),
                          "status":str(r["current_status"] or ""),"reason":str(r["reason"] or "")})
                    injury_source_status=f"ARCHIVED PARSED NBA REPORT — {len(injury_rows)} rows; latest archive snapshot {max(x['report_datetime'] for x in injury_rows)}"
            except Exception as exc:
                injury_source_status += f"; archive fallback unavailable: {type(exc).__name__}: {exc}"
    finally:
        con.close()
    if team.empty:
        raise RuntimeError("No completed pre-target team-game history was returned")
    for col in ["pts","poss","pace","off_rating","def_rating"]:
        team[col]=pd.to_numeric(team[col],errors="coerce")
    team["game_date"]=pd.to_datetime(team.game_date,errors="coerce")
    team=team.dropna(subset=["game_date","team_id","pts","poss","pace","off_rating","def_rating"])
    team=team[team.game_date < target].sort_values("game_date")
    team_name_map={str(r.team_id):str(r.full_name) for r in team_names.itertuples()}
    team_abbr_map={str(r.team_id):str(r.abbreviation) for r in team_names.itertuples() if getattr(r,"abbreviation",None)}
    team_rows=[]
    for tid,g in team.groupby("team_id"):
        recent=g.tail(10)
        if len(recent)<3: continue
        weights=[0.90**i for i in range(len(recent)-1,-1,-1)]
        w=pd.Series(weights,index=recent.index)
        team_rows.append({"team_id":str(tid),"team_name":team_name_map.get(str(tid),str(tid)),"games_used":len(recent),
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
        active["expected_minutes"]=active.minutes*240.0/total if total>0 else 0.0
        active["expected_minutes"]=active["expected_minutes"].clip(upper=48)
        for _ in range(8):
            rem=240.0-float(active.expected_minutes.sum())
            room=active.expected_minutes<48
            if abs(rem)<0.01 or not room.any(): break
            active.loc[room,"expected_minutes"] += rem*active.loc[room,"minutes"]/active.loc[room,"minutes"].sum()
            active["expected_minutes"]=active["expected_minutes"].clip(upper=48)
        for r in active.itertuples():
            player_rows.append({"team_id":str(tid),"player_id":str(r.player_id),
              "player":name_map.get(str(r.player_id),str(r.player_id)),
              "recent_games":int(r.apps),"recent_avg_minutes":round(float(r.minutes),2),
              "expected_minutes_estimate":round(float(r.expected_minutes),2),
              "points_per_minute":round(float(r.ppm),4),
              "impact_pts_per_100_poss":round((float(r.ppm)-replacement)*48.0,3),
              "availability":"UNKNOWN — verify official report","status_source":"No matched official report row",
              "official_status":"","official_reason":"","report_datetime":"",
              "history_as_of":str(recent.game_date.max().date())})
    # Exact normalized name matching within the team is deliberately conservative.
    def name_key(value):
        value=unicodedata.normalize("NFKD",str(value or ""))
        value="".join(ch for ch in value if not unicodedata.combining(ch)).lower()
        value=value.replace(".","").replace("'","").replace("-"," ").replace(","," ")
        return "".join(x for x in value.split() if x)
    player_lookup={(str(p["team_id"]),name_key(p["player"])):p for p in player_rows}
    team_id_by_name={name_key(v):k for k,v in team_name_map.items()}
    team_id_by_name.update({name_key(v):k for k,v in team_abbr_map.items()})
    matched=0
    for row in injury_rows:
        if not row.get("player_report_name"): continue
        tid=team_id_by_name.get(name_key(row["team"]))
        reported=row["player_report_name"]
        parts=[x.strip() for x in reported.split(",")]
        candidates=[name_key(reported)]
        if len(parts)==2: candidates.append(name_key(parts[1]+" "+parts[0]))
        suffixes={"jr","sr","ii","iii","iv"}
        candidates += ["".join(tok for tok in k.split() if tok not in suffixes) for k in list(candidates)]
        target_player=next((player_lookup[(tid,k)] for k in candidates if tid and (tid,k) in player_lookup),None)
        if target_player is None and tid:
            team_candidates=[(k,p) for (team_id,k),p in player_lookup.items() if team_id==tid]
            scored=sorted(((difflib.SequenceMatcher(None,c,k).ratio(),p) for c in candidates for k,p in team_candidates),key=lambda x:x[0],reverse=True)
            if scored and scored[0][0]>=0.94 and (len(scored)==1 or scored[0][0]-scored[1][0]>=0.03):
                target_player=scored[0][1]
        if target_player:
            target_player["official_status"]=row["status"]
            target_player["official_reason"]=row["reason"]
            target_player["report_datetime"]=row["report_datetime"]
            target_player["availability"]=row["status"] or "UNKNOWN — blank status in report"
            target_player["status_source"]="NBA official injury report PDF" if "NBA OFFICIAL PDF" in injury_source_status else "NBA official injury report archive"
            matched+=1
    payload={"schema":"nba-excel-inputs-v2","target_date":str(target.date()),
      "generated_at_utc":datetime.now(timezone.utc).isoformat(),
      "team_inputs":team_rows,"player_inputs":player_rows,"injury_report_rows":injury_rows,
      "injury_report_source_url":injury_source_url,"injury_report_source_status":injury_source_status,
      "injury_report_matched_players":matched,"team_name_map":team_name_map,
      "source_notes":["Team ratings and pace: last 10 regular-season team games, recency weighted, strict pre-target cutoff.",
      "Player minutes: average over up to 10 most recent appearances, normalized to a 240-minute team rotation with a 48-minute cap; estimate, not a validated rotation forecast.",
      "Player impact: recent points-per-minute relative to that team's median player rate; proxy only, not RAPM or causal impact.",
      "Official injury reports are read from NBA-hosted PDFs when available, with the timestamped SportsDataverse archive as fallback. Team abbreviation/full-name aliases and conservative normalized player-name matching are used; unmatched players remain UNKNOWN.",
      "A missing report, missing player row, or NOT YET SUBMITTED team is not evidence of availability."]}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"target_date":str(target.date()),"teams":len(team_rows),"players":len(player_rows),
      "latest_team_history":max(x["as_of"] for x in team_rows) if team_rows else None,
      "latest_player_history":max(x["history_as_of"] for x in player_rows) if player_rows else None,
      "injury_report_status":injury_source_status,"official_report_rows":len(injury_rows),
      "matched_player_rows":matched,"availability":"unmatched or absent statuses remain unknown"},indent=2))
if __name__=="__main__": main()
