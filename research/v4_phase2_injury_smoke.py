#!/usr/bin/env python3
"""Smoke-test the official NBA injury-report PDF archive and parser."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import requests
from PyPDF2 import PdfReader
from nbainjuries import injury
from nbainjuries._util import _gen_filepath

OUT=Path("outputs/research/v4_phase2_injury_smoke")
CACHE=Path("data/research/v4_phase2_injury_smoke")
TIMES=[
    datetime(2025,4,25,17,0),
    datetime(2025,4,25,10,0),
    datetime(2025,4,24,17,0),
]

def main():
    CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for ts in TIMES:
        url=injury.gen_url(ts); path=Path(_gen_filepath(ts,CACHE))
        try:
            if not path.exists():
                r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=45)
                r.raise_for_status(); path.write_bytes(r.content)
            pdf=PdfReader(str(path))
            text="\n".join(p.extract_text() or "" for p in pdf.pages)
            df=injury.get_reportdata(ts,local=True,localdir=CACHE,return_df=True)
            results.append({
                "timestamp":ts.isoformat(),"url":url,"pages":len(pdf.pages),
                "bytes":path.stat().st_size,
                "unsubmitted_phrase_present":"NOT YET SUBMITTED" in text.upper(),
                "rows":int(len(df)),"columns":list(df.columns),
                "sample":df.head(5).to_dict("records"),
            })
        except Exception as e:
            results.append({"timestamp":ts.isoformat(),"url":url,"error":f"{type(e).__name__}: {e}"})
    (OUT/"smoke.json").write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps(results,indent=2,default=str))
    if not any("rows" in x and x["rows"]>0 for x in results):
        raise RuntimeError("No official injury report PDF could be downloaded and parsed")

if __name__=="__main__": main()
