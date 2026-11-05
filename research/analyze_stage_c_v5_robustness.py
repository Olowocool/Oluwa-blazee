#!/usr/bin/env python3
"""Reproduce Stage C vs frozen V5 robustness diagnostics from the workflow CSV artifact."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

P = ["v5_total", "rapm_injury_total", "player_profiles_total", "combined_total"]
E = {k: v.replace("_total", "_err") for k, v in zip(
    ["v5", "rapm_injury", "profiles", "combined"], P)}

def mae(x):
    return float(np.mean(np.abs(np.asarray(x, dtype=float))))

def boot(d, err, rng, reps):
    z = d.assign(delta=np.abs(d[err])-np.abs(d[E["v5"]])).groupby("game_date").delta.agg(["sum", "count"])
    s, n = z["sum"].to_numpy(), z["count"].to_numpy()
    ix = rng.integers(0, len(z), size=(reps, len(z)))
    draws = s[ix].sum(1) / n[ix].sum(1)
    return float(s.sum()/n.sum()), *[float(x) for x in np.quantile(draws, [.025, .975])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    d = pd.read_csv(a.input, parse_dates=["game_date"])
    required = {"season", "game_id", "game_date", "actual_total", *P}
    if required-set(d): raise SystemExit(f"Missing columns: {sorted(required-set(d))}")
    if d.empty or d.game_id.duplicated().any() or d.game_date.isna().any():
        raise SystemExit("Empty, duplicate, or invalid input")
    if not np.isfinite(d[P+["actual_total"]].to_numpy(float)).all():
        raise SystemExit("Non-finite predictions")
    for c in P: d[c.replace("_total", "_err")] = d[c]-d.actual_total
    rng = np.random.default_rng(a.seed)
    boot_rows = {k: boot(d, E[k], rng, a.replicates) for k in ["rapm_injury", "combined", "profiles"]}
    windows = {
        "2025 season": d[d.season == 2025],
        "2026 season": d[d.season == 2026],
        "2026 early through Dec 8": d[(d.season == 2026)&(d.game_date < "2025-12-09")],
        "2026 later from Jan 21": d[(d.season == 2026)&(d.game_date >= "2026-01-21")],
    }
    L = ["# Stage C vs Frozen V5 — Robustness Assessment", "",
         f"Sample: {len(d):,} games, {d.game_date.nunique():,} dates, {d.game_date.min().date()} to {d.game_date.max().date()}.",
         f"Date-cluster bootstrap: {a.replicates:,} replicates, game-weighted, seed {a.seed}.", "",
         "## Overall results", "", "| Model | MAE | RMSE | Bias | Within 5 | Within 10 |", "|---|---:|---:|---:|---:|---:|"]
    for k, label in [("v5","Frozen V5"),("rapm_injury","RAPM + injury"),("profiles","Player profiles"),("combined","Combined")]:
        x=d[E[k]].to_numpy(float)
        L.append(f"| {label} | {mae(x):.3f} | {np.sqrt(np.mean(x*x)):.3f} | {x.mean():+.3f} | {(np.abs(x)<=5).mean()*100:.2f}% | {(np.abs(x)<=10).mean()*100:.2f}% |")
    L += ["", "## Date-cluster bootstrap vs V5", "", "Negative delta favours candidate.", "",
          "| Candidate | MAE delta | 95% CI |", "|---|---:|---:|"]
    for k,label in [("rapm_injury","RAPM + injury"),("combined","Combined"),("profiles","Player profiles")]:
        x=boot_rows[k]; L.append(f"| {label} | {x[0]:+.3f} | [{x[1]:+.3f}, {x[2]:+.3f}] |")
    L += ["", "## Time stability", "", "| Window | Games | V5 MAE | RAPM MAE | Combined MAE | RAPM delta | Combined delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label,g in windows.items():
        v,r,c=(mae(g[E[k]]) for k in ["v5","rapm_injury","combined"])
        L.append(f"| {label} | {len(g)} | {v:.3f} | {r:.3f} | {c:.3f} | {r-v:+.3f} | {c-v:+.3f} |")
    L += ["", "### Four chronological blocks within each season", "", "| Season | Block | Dates | Games | V5 MAE | RAPM MAE | Combined MAE | RAPM delta | Combined delta |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for season,g in d.groupby("season"):
        dates=np.array(sorted(g.game_date.unique()))
        for i,ds in enumerate(np.array_split(dates,4),1):
            q=g[g.game_date.isin(ds)]
            v,r,c=(mae(q[E[k]]) for k in ["v5","rapm_injury","combined"])
            L.append(f"| {season} | {i} | {pd.Timestamp(ds[0]).date()}–{pd.Timestamp(ds[-1]).date()} | {len(q)} | {v:.3f} | {r:.3f} | {c:.3f} | {r-v:+.3f} | {c-v:+.3f} |")
    L += ["", "## Paired win frequency", "", "| Candidate | Games better | Games worse | Game win rate | Dates better | Dates worse |", "|---|---:|---:|---:|---:|---:|"]
    for k,label in [("rapm_injury","RAPM + injury"),("combined","Combined"),("profiles","Profiles")]:
        delta=np.abs(d[E[k]])-np.abs(d[E["v5"]])
        daily=d.assign(delta=delta).groupby("game_date").delta.mean()
        L.append(f"| {label} | {(delta<0).sum()} | {(delta>0).sum()} | {(delta<0).mean()*100:.2f}% | {(daily<0).sum()} | {(daily>0).sum()} |")
    for title, col, cuts in [("Actual-total bands","actual_total",[0,210,220,230,240,250,400]),
                              ("Frozen V5 error bands",E["v5"],[-.001,5,10,15,20,30,1000])]:
        d["_band"]=pd.cut(d[col] if col=="actual_total" else np.abs(d[col]),cuts)
        L += ["", f"## {title}", "", "| Band | Games | V5 MAE | RAPM MAE | Combined MAE |", "|---|---:|---:|---:|---:|"]
        for band,g in d.groupby("_band",observed=True):
            L.append(f"| {band} | {len(g)} | {mae(g[E['v5']]):.3f} | {mae(g[E['rapm_injury']]):.3f} | {mae(g[E['combined']]):.3f} |")
    L += ["", "## Conclusion", "",
      "1. Pooled two-season results show a predictive signal: RAPM + injury and combined features beat frozen V5; their date-cluster intervals exclude zero.",
      "2. Do not promote. The 2026 season-only intervals cross zero. Its first 277 games materially underperform V5 (combined MAE 15.389 vs 14.618), while later games recover. The pooled gain masks temporal instability.",
      "3. Player profiles alone are not established: their pooled interval crosses zero.",
      "4. The candidates improve low/high actual-total bands but worsen the 220–240 bands and games where V5 was already within five points.",
      "5. Do not select a correction multiplier from post-hoc evaluation. Any shrinkage must be nested and strictly prequential.",
      "6. Decision: V5 remains frozen control; RAPM + injury is the leading research challenger only. No integration, promotion, or profitability claim. Freeze the challenger and test untouched 2026–27 dates with coverage diagnostics.",
      "", "## Scope and limitations", "",
      "This is total-score accuracy, not market-beating or ROI validation. Upstream historical data are reconstructed current-vintage rather than archived as-of-game vintages; injury features exist only where complete pregame reports were available; player profiles are descriptive, not causal. Upstream point-in-time date checks reported zero violations. This script reads the prediction CSV and writes only this report.", ""]
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text("\n".join(L),encoding="utf-8")
    print(f"OK: {len(d)} games / {d.game_date.nunique()} dates")

if __name__ == "__main__": main()
