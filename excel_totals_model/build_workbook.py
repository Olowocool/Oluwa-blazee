from __future__ import annotations
import os, json, datetime as dt
from pathlib import Path
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

OUT = Path("NBA_Total_Points_Model_Automatic.xlsx")
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def get_json(url):
    r=requests.get(url, timeout=25, headers={"User-Agent":"NBA-Totals-Research-Workbook/1.0"})
    r.raise_for_status()
    return r.json()

def scheduled_games(day):
    data=get_json(f"{BASE}/scoreboard?dates={day:%Y%m%d}&limit=500")
    out=[]
    for e in data.get("events",[]):
        comp=(e.get("competitions") or [{}])[0]
        competitors=comp.get("competitors",[])
        home=next((x for x in competitors if x.get("homeAway")=="home"),{})
        away=next((x for x in competitors if x.get("homeAway")=="away"),{})
        ht=home.get("team",{}); at=away.get("team",{})
        if ht.get("abbreviation") and at.get("abbreviation"):
            out.append([e.get("id"),e.get("date"),at.get("abbreviation"),at.get("displayName"),ht.get("abbreviation"),ht.get("displayName"),e.get("status",{}).get("type",{}).get("description","Scheduled")])
    return out

def roster_rows(abbr):
    data=get_json(f"{BASE}/teams/{abbr.lower()}/roster")
    athletes=data.get("athletes",[])
    rows=[]
    def walk(items):
        for x in items:
            if isinstance(x,dict) and x.get("id") and x.get("fullName"):
                rows.append([abbr,x.get("id"),x.get("fullName"),x.get("position",{}).get("abbreviation",""),"","","","ESPN roster; availability/minutes not inferred"])
            elif isinstance(x,dict) and isinstance(x.get("items"),list):
                walk(x["items"])
    walk(athletes)
    return rows

def style(ws, freeze="A2"):
    ws.freeze_panes=freeze
    ws.sheet_view.showGridLines=False
    for c in ws[1]:
        c.font=Font(bold=True,color="FFFFFF")
        c.fill=PatternFill("solid",fgColor="17365D")
        c.alignment=Alignment(wrap_text=True,vertical="center")
    ws.row_dimensions[1].height=32
    for col in ws.columns:
        letter=get_column_letter(col[0].column)
        width=min(38,max(13,max((len(str(c.value or "")) for c in col[:80]),default=10)+2))
        ws.column_dimensions[letter].width=width

def main():
    today=dt.date.today()
    date_text=os.getenv("GAME_DATE",today.isoformat())
    day=dt.date.fromisoformat(date_text)
    games=scheduled_games(day)
    wb=Workbook()
    dash=wb.active; dash.title="Dashboard"
    dash.append(["NBA TOTAL POINTS — INDEPENDENT MODEL"])
    dash.append(["Game date",day.isoformat()])
    dash.append(["Selected game ID",""])
    dash.append(["Away team",""]); dash.append(["Home team",""])
    dash.append(["Projected possessions","=IFERROR(INDEX('Game Engine'!G:G,MATCH(B3,'Game Engine'!A:A,0)),\"\")"])
    dash.append(["Away projected points","=IFERROR(INDEX('Game Engine'!H:H,MATCH(B3,'Game Engine'!A:A,0)),\"\")"])
    dash.append(["Home projected points","=IFERROR(INDEX('Game Engine'!I:I,MATCH(B3,'Game Engine'!A:A,0)),\"\")"])
    dash.append(["Predicted combined points","=IFERROR(INDEX('Game Engine'!J:J,MATCH(B3,'Game Engine'!A:A,0)),\"\")"])
    dash.append(["Input status","=IF(B3=\"\",\"Select a Game ID\",IFERROR(INDEX('Game Engine'!K:K,MATCH(B3,'Game Engine'!A:A,0)),\"Game not found\"))"])
    dash.append(["Model note","Provisional until team metrics and expected player minutes are populated from verified pregame sources."])
    sch=wb.create_sheet("Schedule")
    sch.append(["Game ID","Start UTC","Away code","Away team","Home code","Home team","Status"])
    for row in games: sch.append(row)
    ti=wb.create_sheet("Team Inputs")
    ti.append(["Team code","Team name","Offensive rating (pts/100 poss)","Defensive rating allowed (pts/100 poss)","Pace (poss/48)","Data as-of UTC","Source"])
    teams={}
    for g in games:
        teams[g[2]]=g[3]; teams[g[4]]=g[5]
    for code,name in sorted(teams.items()): ti.append([code,name,None,None,None,None,"Populate from verified pregame team logs"])
    pi=wb.create_sheet("Player Inputs")
    pi.append(["Team code","Player ID","Player","Position","Expected minutes (0-48)","Points per 100 poss above replacement","Availability","Source / timestamp"])
    for code in sorted(teams):
        try:
            for row in roster_rows(code): pi.append(row)
        except Exception as e:
            pi.append([code,"","ROSTER FETCH FAILED","","","","Unknown",str(e)[:250]])
    eng=wb.create_sheet("Game Engine")
    eng.append(["Game ID","Away code","Home code","Away ORtg","Home ORtg","Expected pace","Projected possessions","Away points","Home points","Combined total","Input status","Away player delta","Home player delta"])
    for i,g in enumerate(games,2):
        eng.append([g[0],g[2],g[4],
            f'=IFERROR(VLOOKUP(B{i},\'Team Inputs\'!A:E,3,FALSE),0)',
            f'=IFERROR(VLOOKUP(C{i},\'Team Inputs\'!A:E,3,FALSE),0)',
            f'=IFERROR(AVERAGE(VLOOKUP(B{i},\'Team Inputs\'!A:E,5,FALSE),VLOOKUP(C{i},\'Team Inputs\'!A:E,5,FALSE)),0)',
            f'=IF(F{i}>0,F{i},0)',
            f'=IF(AND(D{i}>0,E{i}>0,G{i}>0),G{i}*D{i}/100+L{i},0)',
            f'=IF(AND(D{i}>0,E{i}>0,G{i}>0),G{i}*E{i}/100+M{i},0)',
            f'=IF(AND(H{i}>0,I{i}>0),H{i}+I{i},0)',
            f'=IF(OR(D{i}<=0,E{i}<=0,F{i}<=0), "MISSING TEAM METRICS", "TEAM INPUTS PRESENT; PLAYER ADJUSTMENTS REQUIRE VERIFIED MINUTES")',
            0,0])
    settings=wb.create_sheet("Settings")
    for row in [["Parameter","Value","Meaning"],["League pace anchor",99.0,"Editable provisional anchor; replace with sourced season value"],["League offensive rating anchor",115.0,"Editable provisional anchor; points per 100 possessions"],["Home court scoring adjustment",0.0,"Default zero until independently estimated"],["Player adjustment method","Incremental above-replacement","Do not sum player PPG; requires expected minutes and validated impact rates"],["Confidence interval","Not calibrated","No statistical confidence range is claimed"]]: settings.append(row)
    log=wb.create_sheet("Source Log")
    log.append(["Retrieved UTC","Source","Coverage","Limitations"])
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    log.append([stamp,"ESPN public scoreboard API",f"{len(games)} fixtures for {day.isoformat()}","Schedule feed only; fixture coverage can vary"])
    log.append([stamp,"ESPN public team roster API",f"{pi.max_row-1} roster rows","Roster is not injury status; no expected minutes inferred"])
    log.append([stamp,"Model workbook formulas","Created","Team ratings and expected player minutes are not yet automatically sourced; dashboard flags missing inputs"])
    for ws in wb.worksheets: style(ws)
    dash.column_dimensions["B"].width=60
    wb.calculation.fullCalcOnLoad=True
    wb.calculation.forceFullCalc=True
    wb.save(OUT)
    print(json.dumps({"workbook":str(OUT),"game_date":day.isoformat(),"fixtures":len(games),"roster_rows":pi.max_row-1,"sheets":wb.sheetnames},indent=2))

if __name__=="__main__": main()
