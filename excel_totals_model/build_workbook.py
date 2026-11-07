from __future__ import annotations
import os, json, datetime as dt, re, unicodedata
from pathlib import Path
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

OUT=Path("NBA_Total_Points_Model_Automatic.xlsx")
BASE="https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def get_json(url):
    r=requests.get(url,timeout=25,headers={"User-Agent":"NBA-Total-Points-Research-Workbook/1.0"})
    r.raise_for_status()
    return r.json()

def norm(value):
    value=unicodedata.normalize("NFKD",str(value or ""))
    value="".join(ch for ch in value if not unicodedata.combining(ch)).lower().replace("&","and")
    return re.sub(r"[^a-z0-9]","",value)

def scheduled_games(day):
    data=get_json(f"{BASE}/scoreboard?dates={day:%Y%m%d}&limit=500")
    rows=[]
    for event in data.get("events",[]):
        comp=(event.get("competitions") or [{}])[0]
        competitors=comp.get("competitors",[])
        home=next((x for x in competitors if x.get("homeAway")=="home"),{})
        away=next((x for x in competitors if x.get("homeAway")=="away"),{})
        ht=home.get("team",{}); at=away.get("team",{})
        if ht.get("abbreviation") and at.get("abbreviation"):
            rows.append({"id":str(event.get("id")),"date":event.get("date"),
              "away_code":at.get("abbreviation"),"away_name":at.get("displayName"),
              "home_code":ht.get("abbreviation"),"home_name":ht.get("displayName"),
              "status":event.get("status",{}).get("type",{}).get("description","Scheduled")})
    return rows

def roster_status(abbr):
    data=get_json(f"{BASE}/teams/{abbr.lower()}/roster")
    result={}
    def walk(items):
        for x in items or []:
            if not isinstance(x,dict): continue
            if x.get("id") and (x.get("fullName") or x.get("displayName")):
                injuries=x.get("injuries") or []
                injury_text="; ".join(str(z.get("type",{}).get("description") or z.get("shortComment") or z.get("longComment") or "listed") for z in injuries if isinstance(z,dict))
                result[str(x["id"])]={
                  "name":x.get("fullName") or x.get("displayName"),
                  "status":(x.get("status") or {}).get("name","Unknown"),
                  "injury":injury_text or "No injury detail returned by roster endpoint"}
            if isinstance(x.get("items"),list): walk(x["items"])
    walk(data.get("athletes",[]))
    return result

def style(ws):
    ws.freeze_panes="A2"; ws.sheet_view.showGridLines=False
    for cell in ws[1]:
        cell.font=Font(bold=True,color="FFFFFF")
        cell.fill=PatternFill("solid",fgColor="17365D")
        cell.alignment=Alignment(wrap_text=True,vertical="center")
    ws.row_dimensions[1].height=34
    for col in ws.columns:
        letter=get_column_letter(col[0].column)
        width=min(42,max(12,max((len(str(x.value or "")) for x in col[:100]),default=10)+2))
        ws.column_dimensions[letter].width=width

def main():
    day=dt.date.fromisoformat(os.getenv("GAME_DATE") or dt.date.today().isoformat())
    games=scheduled_games(day)
    source_path=Path("excel_totals_model/source_inputs.json")
    if not source_path.exists(): raise RuntimeError("Point-in-time source_inputs.json missing; run refresh_sources.py first")
    src=json.loads(source_path.read_text(encoding="utf-8"))
    if src.get("target_date")!=day.isoformat(): raise RuntimeError("Source inputs target date does not match selected game date")
    injury_matches=0
    team_by_name={norm(x["team_name"]):x for x in src.get("team_inputs",[])}
    team_code_map={}
    for g in games:
        for code,name in ((g["away_code"],g["away_name"]),(g["home_code"],g["home_name"])):
            match=team_by_name.get(norm(name))
            if not match:
                aliases={"LA Clippers":"Los Angeles Clippers","LA Lakers":"Los Angeles Lakers"}
                match=team_by_name.get(norm(aliases.get(name,name)))
            if match: team_code_map[code]=match
    wb=Workbook(); dash=wb.active; dash.title="Dashboard"
    for row in [
      ["NBA TOTAL POINTS — INDEPENDENT MODEL"],["Game date",day.isoformat()],["Selected Game ID",""],
      ["Away team",'=IFERROR(INDEX(Schedule!D:D,MATCH(B3,Schedule!A:A,0)),"")'],
      ["Home team",'=IFERROR(INDEX(Schedule!F:F,MATCH(B3,Schedule!A:A,0)),"")'],
      ["Projected possessions",'=IFERROR(INDEX(\'Game Engine\'!K:K,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Away projected points",'=IFERROR(INDEX(\'Game Engine\'!L:L,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Home projected points",'=IFERROR(INDEX(\'Game Engine\'!M:M,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Predicted combined points",'=IFERROR(INDEX(\'Game Engine\'!N:N,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Input status",'=IF(B3="","Select a Game ID",IFERROR(INDEX(\'Game Engine\'!O:O,MATCH(B3,\'Game Engine\'!A:A,0)),"Game not found"))'],
      ["Model classification","PROVISIONAL — recent-form and player-rate proxy; not validated or calibrated."],
      ["Availability warning","Only matched timestamped official NBA injury-report rows receive a status. Missing rows and NOT YET SUBMITTED remain unknown."],
      ["Official report feed",src.get("injury_report_source_status","Not loaded")],
      ["Official report matched players",injury_matches]
    ]: dash.append(row)
    sch=wb.create_sheet("Schedule"); sch.append(["Game ID","Start UTC","Away code","Away team","Home code","Home team","Status"])
    for g in games: sch.append([g["id"],g["date"],g["away_code"],g["away_name"],g["home_code"],g["home_name"],g["status"]])
    ti=wb.create_sheet("Team Inputs"); ti.append(["Team code","Team name","Offensive rating","Defensive rating allowed","Pace","Games used","History as of","Source"])
    for code,m in sorted(team_code_map.items()):
        ti.append([code,m["team_name"],m["ortg"],m["drtg"],m["pace"],m["games_used"],m["as_of"],"Public team logs; last 10 regular-season games, recency weighted"])
    pi=wb.create_sheet("Player Inputs")
    pi.append(["Team code","Player ID","Player","Recent games","Recent avg minutes","Expected minutes estimate","Impact pts/100 poss","Availability","Official report reason","Official report timestamp (source local)","Status source","Roster note","History as of"])
    code_by_teamid={str(v["team_id"]):k for k,v in team_code_map.items()}
    roster_cache={}
    for code in team_code_map:
        try: roster_cache[code]=roster_status(code)
        except Exception: roster_cache[code]={}
    # Build a complete roster view, not just players who appeared in the last ten games.
    # This allows an official report to match players who have missed recent games.
    all_players={(str(p["team_id"]),str(p["player_id"])):dict(p) for p in src.get("player_inputs",[])}
    code_alias={}
    for code,g in ((g["away_code"],g) for g in games):
        code_alias[norm(code)]=code; code_alias[norm(g["away_name"])]=code
    for g in games:
        code_alias[norm(g["home_code"])]=g["home_code"]; code_alias[norm(g["home_name"])]=g["home_code"]
    for code,team in team_code_map.items():
        tid=str(team["team_id"])
        for pid,athlete in roster_cache.get(code,{}).items():
            key=(tid,str(pid))
            roster_name=athlete.get("name") or str(pid)
            existing=all_players.get(key)
            if existing is None:
                existing=next((row for row in all_players.values()
                  if str(row["team_id"])==tid and norm(row.get("player"))==norm(roster_name)),None)
            if existing is not None:
                existing["_roster_name"]=roster_name
                existing["_roster_injury"]=athlete.get("injury","")
            else:
                all_players[key]={"team_id":tid,"player_id":str(pid),"player":roster_name,
                  "recent_games":0,"recent_avg_minutes":0.0,"expected_minutes_estimate":0.0,
                  "impact_pts_per_100_poss":0.0,"availability":"UNKNOWN — no recent history",
                  "status_source":"Roster identity only; no recent player history","official_status":"",
                  "official_reason":"","report_datetime":"","history_as_of":"",
                  "_roster_name":roster_name,"_roster_injury":athlete.get("injury","")}
    matched_roster_players=set()
    injury_matches=0
    for injury in src.get("injury_report_rows",[]):
        code=code_alias.get(norm(injury.get("team")))
        if not code:
            continue
        team=str(team_code_map.get(code,{}).get("team_id",""))
        if not team:
            continue
        reported=str(injury.get("player_report_name") or "")
        if not reported:
            continue
        report_names={norm(reported)}
        if "," in reported:
            last,first=[x.strip() for x in reported.split(",",1)]
            report_names.add(norm(first+" "+last))
        roster=roster_cache.get(code,{})
        match=next(((pid,a) for pid,a in roster.items() if norm(a.get("name")) in report_names),None)
        if match:
            pid,athlete=match
            row=all_players.get((team,str(pid)))
            if row is None:
                row={"team_id":team,"player_id":str(pid),"player":athlete.get("name") or reported,
                  "recent_games":0,"recent_avg_minutes":0.0,"expected_minutes_estimate":0.0,
                  "impact_pts_per_100_poss":0.0,"availability":"UNKNOWN — no recent history",
                  "status_source":"Roster identity only; no recent player history","official_status":"",
                  "official_reason":"","report_datetime":"","history_as_of":""}
                all_players[(team,str(pid))]=row
            row["official_status"]=injury.get("status") or ""
            row["official_reason"]=injury.get("reason") or ""
            row["report_datetime"]=injury.get("report_datetime") or ""
            row["availability"]=injury.get("status") or "UNKNOWN — blank status in report"
            row["status_source"]="NBA official injury report PDF/archive"
            injury_matches+=1
    for p in all_players.values():
        code=code_by_teamid.get(str(p["team_id"]))
        if not code: continue
        r=roster_cache.get(code,{}).get(str(p["player_id"]),{})
        pi.append([code,p["player_id"],r.get("name") or p["player"],p["recent_games"],p["recent_avg_minutes"],
          p["expected_minutes_estimate"],p["impact_pts_per_100_poss"],p.get("official_status") or "UNKNOWN — no matched official row",
          p.get("official_reason") or "",p.get("report_datetime") or "",
          p.get("status_source") or "No matched official report row",
          r.get("injury","No matching ESPN roster identity; roster status is not official availability"),p.get("history_as_of") or "No recent player history"])
    eng=wb.create_sheet("Game Engine")
    eng.append(["Game ID","Away code","Home code","Away ORtg","Away DRtg","Home ORtg","Home DRtg","Away pace","Home pace","Expected pace","Projected possessions","Away points","Home points","Combined total","Input status","Away player delta","Home player delta"])
    for i,g in enumerate(games,2):
        eng.append([g["id"],g["away_code"],g["home_code"],
          f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,3,FALSE),0)',f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,4,FALSE),0)',
          f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,3,FALSE),0)',f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,4,FALSE),0)',
          f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,5,FALSE),0)',f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,5,FALSE),0)',
          f'=IF(AND(H{i}>0,I{i}>0),AVERAGE(H{i},I{i}),0)',f'=J{i}',
          f'=IF(AND(D{i}>0,G{i}>0,K{i}>0),K{i}*(D{i}+G{i}-Settings!$B$2)/100+P{i},0)',
          f'=IF(AND(F{i}>0,E{i}>0,K{i}>0),K{i}*(F{i}+E{i}-Settings!$B$2)/100+Q{i},0)',
          f'=IF(AND(L{i}>0,M{i}>0),L{i}+M{i},0)',
          f'=IF(OR(D{i}<=0,E{i}<=0,F{i}<=0,G{i}<=0,H{i}<=0,I{i}<=0),"MISSING TEAM HISTORY / CROSSWALK","PROVISIONAL — PLAYER MINUTES AND AVAILABILITY NOT VERIFIED")',
          f'=IFERROR(SUMPRODUCT((\'Player Inputs\'!$A$2:$A$2500=B{i})*(\'Player Inputs\'!$F$2:$F$2500)/48*(\'Player Inputs\'!$G$2:$G$2500))*K{i}*Settings!$B$3/100,0)',
          f'=IFERROR(SUMPRODUCT((\'Player Inputs\'!$A$2:$A$2500=C{i})*(\'Player Inputs\'!$F$2:$F$2500)/48*(\'Player Inputs\'!$G$2:$G$2500))*K{i}*Settings!$B$3/100,0)'])
    settings=wb.create_sheet("Settings")
    for row in [["Parameter","Value","Meaning"],["League offensive rating anchor",115.0,"Provisional matchup efficiency anchor"],
      ["Player impact scale",1.0,"Multiplier on recent points-per-minute proxy relative to team median"],
      ["Recent team games",10,"Last ten completed regular-season team games"],["Recency decay",0.90,"Newest team game receives highest weight"],
      ["Expected minutes method","Recent 10 appearances, normalized to 240","Estimate, not confirmed rotation"],
      ["Injury status","Timestamped official NBA injury report when matched; otherwise UNKNOWN","Never infer availability from missing data"],
      ["Prediction confidence","Not calibrated","No confidence interval is provided"]]: settings.append(row)
    log=wb.create_sheet("Source Log"); log.append(["Retrieved UTC","Source","Coverage","Limitations"])
    stamp=src.get("generated_at_utc","")
    log.append([stamp,"basketball-data.billmill.org public parquet",f"{len(src.get('team_inputs',[]))} teams; {len(src.get('player_inputs',[]))} player rows","Completed-game history; strict pre-target cutoff; source vintages may be revised"])
    log.append([stamp,"ESPN public scoreboard",f"{len(games)} fixtures for {day.isoformat()}","Unofficial public endpoint"])
    log.append([stamp,"ESPN public team rosters",f"{sum(len(x) for x in roster_cache.values())} roster identities","Roster status is not a substitute for game-date availability"])
    log.append([stamp,"NBA official injury report snapshots via SportsDataverse",src.get("injury_report_source_status","Not loaded"),"Timestamp and labels retained; exact team/name matching only; missing rows remain unknown"])
    log.append([stamp,"Player adjustment","Recent scoring rate above team median","Descriptive proxy only; not RAPM or causal impact"])
    log.append([stamp,"Availability/minutes",f"{injury_matches} matched roster identities","Expected minutes remain historical estimates, not confirmed rotations"])
    for ws in wb.worksheets: style(ws)
    if games:
        dash["B3"]=games[0]["id"]
        dv=DataValidation(type="list",formula1='=INDIRECT("Schedule!$A$2:$A$'+str(min(len(games)+1,1000))+'")',allow_blank=True)
        dash.add_data_validation(dv); dv.add(dash["B3"])
    dash.column_dimensions["B"].width=74
    wb.calculation.fullCalcOnLoad=True; wb.calculation.forceFullCalc=True
    wb.save(OUT)
    print(json.dumps({"workbook":str(OUT),"date":day.isoformat(),"fixtures":len(games),
      "team_crosswalks":len(team_code_map),"player_rows":pi.max_row-1,"official_report_matched_players":injury_matches,"sheets":wb.sheetnames,
      "official_report_status":src.get("injury_report_source_status"),"official_report_matched_players":injury_matches,
      "classification":"provisional"},indent=2))
if __name__=="__main__": main()
