from __future__ import annotations
import os, json, datetime as dt, re
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
    return re.sub(r"[^a-z0-9]","",str(value or "").lower().replace("&","and"))

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
              "away_code":at.get("abbreviation"),
              "away_name":at.get("displayName"),"home_code":ht.get("abbreviation"),
              "home_name":ht.get("displayName"),
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
                  "position":(x.get("position") or {}).get("abbreviation",""),
                  "status":(x.get("status") or {}).get("name","Unknown"),
                  "injury":injury_text or "No injury detail returned by roster endpoint"
                }
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
    if not source_path.exists():
        raise RuntimeError("Point-in-time source_inputs.json missing; run refresh_sources.py first")
    src=json.loads(source_path.read_text(encoding="utf-8"))
    if src.get("target_date")!=day.isoformat():
        raise RuntimeError("Source inputs target date does not match selected game date")
    team_by_name={norm(x["team_name"]):x for x in src.get("team_inputs",[])}
    team_code_map={}
    for g in games:
        for code,name in ((g["away_code"],g["away_name"]),(g["home_code"],g["home_name"])):
            match=team_by_name.get(norm(name))
            if not match:
                # Limited known naming variants only; no fuzzy team-ID guessing.
                aliases={"LA Clippers":"Los Angeles Clippers","LA Lakers":"Los Angeles Lakers"}
                match=team_by_name.get(norm(aliases.get(name,name)))
            if match: team_code_map[code]=match
    wb=Workbook()
    dash=wb.active; dash.title="Dashboard"
    for row in [
      ["NBA TOTAL POINTS — INDEPENDENT MODEL"],
      ["Game date",day.isoformat()],["Selected Game ID",""],
      ["Away team",'=IFERROR(INDEX(Schedule!D:D,MATCH(B3,Schedule!A:A,0)),"")'],
      ["Home team",'=IFERROR(INDEX(Schedule!F:F,MATCH(B3,Schedule!A:A,0)),"")'],
      ["Projected possessions",'=IFERROR(INDEX(\'Game Engine\'!K:K,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Away projected points",'=IFERROR(INDEX(\'Game Engine\'!L:L,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Home projected points",'=IFERROR(INDEX(\'Game Engine\'!M:M,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Predicted combined points",'=IFERROR(INDEX(\'Game Engine\'!N:N,MATCH(B3,\'Game Engine\'!A:A,0)),"")'],
      ["Input status",'=IF(B3="","Select a Game ID",IFERROR(INDEX(\'Game Engine\'!O:O,MATCH(B3,\'Game Engine\'!A:A,0)),"Game not found"))'],
      ["Model classification","PROVISIONAL — recent-form and player-rate proxy; not validated or calibrated."],
      ["Availability warning","Roster status is not an official game-date injury report. Verify availability before relying on any projection."]
    ]: dash.append(row)
    sch=wb.create_sheet("Schedule")
    sch.append(["Game ID","Start UTC","Away code","Away team","Home code","Home team","Status"])
    for g in games: sch.append([g["id"],g["date"],g["away_code"],g["away_name"],g["home_code"],g["home_name"],g["status"]])
    ti=wb.create_sheet("Team Inputs")
    ti.append(["Team code","Team name","Offensive rating","Defensive rating allowed","Pace","Games used","History as of","Source"])
    for code,match in sorted(team_code_map.items()):
        ti.append([code,match["team_name"],match["ortg"],match["drtg"],match["pace"],match["games_used"],match["as_of"],"Public team logs; last 10 regular-season games, recency weighted"])
    pi=wb.create_sheet("Player Inputs")
    pi.append(["Team code","Player ID","Player","Recent games","Recent avg minutes","Expected minutes estimate","Impact pts/100 poss","Availability","Injury detail / source note","History as of"])
    code_by_teamid={str(v["team_id"]):k for k,v in team_code_map.items()}
    roster_cache={}
    for code in team_code_map:
        try: roster_cache[code]=roster_status(code)
        except Exception as exc: roster_cache[code]={}
    for p in src.get("player_inputs",[]):
        code=code_by_teamid.get(str(p["team_id"]))
        if not code: continue
        r=roster_cache.get(code,{}).get(str(p["player_id"]),{})
        pi.append([code,p["player_id"],r.get("name") or p["player"],p["recent_games"],p["recent_avg_minutes"],p["expected_minutes_estimate"],p["impact_pts_per_100_poss"],"UNKNOWN — official report required",r.get("injury","No matching ESPN roster identity; availability unknown"),p["history_as_of"]])
    eng=wb.create_sheet("Game Engine")
    eng.append(["Game ID","Away code","Home code","Away ORtg","Away DRtg","Home ORtg","Home DRtg","Away pace","Home pace","Expected pace","Projected possessions","Away points","Home points","Combined total","Input status","Away player delta","Home player delta"])
    for i,g in enumerate(games,2):
        eng.append([g["id"],g["away_code"],g["home_code"],
          f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,3,FALSE),0)',
          f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,4,FALSE),0)',
          f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,3,FALSE),0)',
          f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,4,FALSE),0)',
          f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,5,FALSE),0)',
          f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,5,FALSE),0)',
          f'=IF(AND(H{i}>0,I{i}>0),AVERAGE(H{i},I{i}),0)',
          f'=J{i}',
          f'=IF(AND(D{i}>0,G{i}>0,K{i}>0),K{i}*(D{i}+G{i}-Settings!$B$2)/100+P{i},0)',
          f'=IF(AND(F{i}>0,E{i}>0,K{i}>0),K{i}*(F{i}+E{i}-Settings!$B$2)/100+Q{i},0)',
          f'=IF(AND(L{i}>0,M{i}>0),L{i}+M{i},0)',
          f'=IF(OR(D{i}<=0,E{i}<=0,F{i}<=0,G{i}<=0,H{i}<=0,I{i}<=0),"MISSING TEAM HISTORY / CROSSWALK","PROVISIONAL — PLAYER MINUTES AND AVAILABILITY NOT VERIFIED")',
          f'=IFERROR(SUMPRODUCT((\'Player Inputs\'!$A$2:$A$2500=B{i})*(\'Player Inputs\'!$F$2:$F$2500)*(\'Player Inputs\'!$G$2:$G$2500))*K{i}*Settings!$B$3/100,0)',
          f'=IFERROR(SUMPRODUCT((\'Player Inputs\'!$A$2:$A$2500=C{i})*(\'Player Inputs\'!$F$2:$F$2500)*(\'Player Inputs\'!$G$2:$G$2500))*K{i}/100,0)'])
    settings=wb.create_sheet("Settings")
    for row in [
      ["Parameter","Value","Meaning"],
      ["League offensive rating anchor",115.0,"Provisional anchor; matchup adjustment = own ORtg + opponent DRtg − anchor"],
      ["Player impact scale",1.0,"Multiplier on recent points-per-minute proxy relative to team median"],
      ["Recent team games",10,"Last ten completed regular-season team games"],
      ["Recency decay",0.90,"Applied to team rating observations; newest game has highest weight"],
      ["Expected minutes method","Recent 10 appearances, normalized to 240","Estimated rotation, not official projection"],
      ["Injury status","Unknown pending official report","No automatic injury confirmation is claimed"],
      ["Prediction confidence","Not calibrated","No confidence interval is provided"]
    ]: settings.append(row)
    log=wb.create_sheet("Source Log")
    log.append(["Retrieved UTC","Source","Coverage","Limitations"])
    stamp=src.get("generated_at_utc","")
    log.append([stamp,"basketball-data.billmill.org public parquet",f"{len(src.get('team_inputs',[]))} teams; {len(src.get('player_inputs',[]))} player rows","Recent completed-game stats; strictly before target date; historical source vintages may be revised"])
    log.append([stamp,"ESPN public scoreboard",f"{len(games)} scheduled fixtures for {day.isoformat()}","Unofficial public endpoint; may change without notice"])
    log.append([stamp,"ESPN public team rosters",f"{sum(len(x) for x in roster_cache.values())} roster identities","Roster/injury snippet is not a complete official NBA injury report"])
    log.append([stamp,"Player adjustment", "Recent scoring rate above team median player rate","Descriptive proxy only; not RAPM, causal impact, or a validated forecast adjustment"])
    log.append([stamp,"Availability/minutes","Unknown / estimated","Expected minutes estimated from recent appearances; official injury report not connected"])
    for ws in wb.worksheets: style(ws)
    if games:
        dv=DataValidation(type="list",formula1=f"=Schedule!$A$2:$A${min(len(games)+1,1000)}",allow_blank=True)
        dash.add_data_validation(dv); dv.add(dash["B3"])
    dash.column_dimensions["B"].width=74
    wb.calculation.fullCalcOnLoad=True; wb.calculation.forceFullCalc=True
    wb.save(OUT)
    print(json.dumps({"workbook":str(OUT),"date":day.isoformat(),"fixtures":len(games),
      "team_crosswalks":len(team_code_map),"player_rows":pi.max_row-1,
      "sheets":wb.sheetnames,"classification":"provisional"},indent=2))
if __name__=="__main__": main()
