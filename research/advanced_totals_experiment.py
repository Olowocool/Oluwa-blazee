#!/usr/bin/env python3
"""Research-only, point-in-time NBA totals experiment; production code is untouched."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://basketball-data.billmill.org"
ARCHIVE = Path("research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv")
OUT = Path("outputs/research/advanced_totals_experiment")
HOLDOUT_START = pd.Timestamp("2025-11-26")
VAL_START = pd.Timestamp("2024-12-01")
VAL_END = pd.Timestamp("2025-07-01")
MIN_GAMES = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0]

FEATURES = [
    "formula_total", "expected_pace", "league_ortg", "league_drtg", "league_pace",
    "a_games", "b_games", "mean_pace_l10", "pace_gap_l10", "mean_pace_season",
    "mean_adj_ortg_l10", "adj_ortg_gap_l10", "mean_adj_ortg_season",
    "mean_adj_drtg_l10", "adj_drtg_gap_l10", "mean_adj_drtg_season",
    "mean_raw_ortg_l10", "mean_raw_drtg_l10", "neutral_site",
]


def team_key(value):
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^los angeles ", "la ", s)
    return re.sub(r"\s+", " ", s)


def connect():
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as r:
        meta = json.loads(r.read().decode())["datasets"]
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    for name in ("team_game_logs", "games", "team_seasons"):
        m = meta[name]
        first, last, count = int(m["first_season"]), int(m["last_season"]), int(m["seasons"])
        seasons = list(range(first, last + 1))
        if len(seasons) != count:
            raise RuntimeError(f"{name} metadata indicates missing seasons")
        urls = ", ".join("'" + f"{BASE}/nba/stats/{name}/season={s}/data.parquet" + "'" for s in seasons)
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet([{urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
    return con, meta


def load_data(con):
    games = con.execute("""
        SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
               t.pts, t.poss, t.pace, t.off_rating, t.def_rating,
               g.game_type, g.neutral_site
        FROM team_game_logs t JOIN games g USING (season, game_id)
        WHERE g.game_type IN ('regular_season', 'cup_final')
        ORDER BY t.game_date, t.season, t.game_id, t.team_id
    """).fetchdf()
    teams = con.execute("""
        SELECT season, team_id, full_name, is_nba
        FROM team_seasons WHERE season = 2026
    """).fetchdf()
    return games, teams


def mean(rows, key, default):
    x = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.mean(x)) if x else float(default)


def pregame(team_history, season, league):
    h = [x for x in team_history if x["season"] == season]
    if len(h) < MIN_GAMES:
        return None
    r = h[-10:]
    return {
        "games": float(len(h)),
        "pace10": mean(r, "pace", league["pace"]),
        "pace_season": mean(h, "pace", league["pace"]),
        "poss10": mean(r, "poss", league["pace"]),
        "raw_o10": mean(r, "off_rating", league["ortg"]),
        "raw_o_season": mean(h, "off_rating", league["ortg"]),
        "raw_d10": mean(r, "def_rating", league["drtg"]),
        "raw_d_season": mean(h, "def_rating", league["drtg"]),
        "adj_o10": mean(r, "adj_o", league["ortg"]),
        "adj_o_season": mean(h, "adj_o", league["ortg"]),
        "adj_d10": mean(r, "adj_d", league["drtg"]),
        "adj_d_season": mean(h, "adj_d", league["drtg"]),
        "last_date": max(x["date"] for x in h),
    }


def build_features(raw):
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    for c in ("pts", "poss", "pace", "off_rating", "def_rating"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["season"] = pd.to_numeric(d.season, errors="coerce")
    d["team_id"] = d.team_id.astype(str)
    d["opp_team_id"] = d.opp_team_id.astype(str)
    d = d.dropna(subset=["game_date", "season", "pts", "poss", "pace", "off_rating", "def_rating"])
    d = d[d.game_type.isin(["regular_season", "cup_final"])].copy()
    if d.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate season/game/team key")
    grouped = defaultdict(list)
    for (season, gid), g in d.groupby(["season", "game_id"], sort=False):
        rows = g.to_dict("records")
        if len(rows) != 2:
            raise RuntimeError(f"Expected two team rows for {gid}; found {len(rows)}")
        rows.sort(key=lambda r: str(r["team_id"]))
        grouped[pd.Timestamp(rows[0]["game_date"]).normalize()].append((int(season), str(gid), rows))

    history = defaultdict(list)
    league_history = defaultdict(list)
    output = []

    for date in sorted(grouped):
        today = grouped[date]
        league_by_season = {}
        for season in {x[0] for x in today}:
            lh = league_history[season]
            league_by_season[season] = (
                {"ortg": mean(lh, "off_rating", 110), "drtg": mean(lh, "def_rating", 110),
                 "pace": mean(lh, "pace", 100)}
                if lh else {"ortg": 110.0, "drtg": 110.0, "pace": 100.0}
            )

        updates = []
        for season, gid, rows in today:
            a, b = rows
            aid, bid = str(a["team_id"]), str(b["team_id"])
            if aid == bid or str(a["opp_team_id"]) != bid or str(b["opp_team_id"]) != aid:
                raise RuntimeError(f"Broken opponent pair on {gid}")
            lg = league_by_season[season]
            pa, pb = pregame(history[aid], season, lg), pregame(history[bid], season, lg)
            if pa is not None and pb is not None:
                pace = np.mean([pa["pace10"], pb["pace10"], pa["pace_season"], pb["pace_season"]])
                ao = np.mean([pa["adj_o10"], pa["adj_o_season"]])
                bo = np.mean([pb["adj_o10"], pb["adj_o_season"]])
                ad = np.mean([pa["adj_d10"], pa["adj_d_season"]])
                bd = np.mean([pb["adj_d10"], pb["adj_d_season"]])
                ae = lg["ortg"] + .5 * (ao - lg["ortg"]) + .5 * (bd - lg["drtg"])
                be = lg["ortg"] + .5 * (bo - lg["ortg"]) + .5 * (ad - lg["drtg"])
                lastdate = max(pa["last_date"], pb["last_date"])
                if not lastdate < date:
                    raise AssertionError(f"Temporal leakage: {lastdate} >= {date}")
                neutral = a.get("neutral_site", False)
                neutral = int(bool(neutral)) if pd.notna(neutral) else 0
                row = {
                    "season": season, "game_id": gid, "game_date": date,
                    "team_a_id": aid, "team_b_id": bid,
                    "team_a_points": float(a["pts"]), "team_b_points": float(b["pts"]),
                    "actual_total_source": float(a["pts"]) + float(b["pts"]),
                    "game_type": str(a["game_type"]), "neutral_site": neutral,
                    "league_ortg": lg["ortg"], "league_drtg": lg["drtg"], "league_pace": lg["pace"],
                    "a_games": pa["games"], "b_games": pb["games"],
                    "a_pace10": pa["pace10"], "b_pace10": pb["pace10"],
                    "a_pace_season": pa["pace_season"], "b_pace_season": pb["pace_season"],
                    "a_poss10": pa["poss10"], "b_poss10": pb["poss10"],
                    "a_raw_o10": pa["raw_o10"], "b_raw_o10": pb["raw_o10"],
                    "a_raw_o_season": pa["raw_o_season"], "b_raw_o_season": pb["raw_o_season"],
                    "a_raw_d10": pa["raw_d10"], "b_raw_d10": pb["raw_d10"],
                    "a_raw_d_season": pa["raw_d_season"], "b_raw_d_season": pb["raw_d_season"],
                    "a_adj_o10": pa["adj_o10"], "b_adj_o10": pb["adj_o10"],
                    "a_adj_o_season": pa["adj_o_season"], "b_adj_o_season": pb["adj_o_season"],
                    "a_adj_d10": pa["adj_d10"], "b_adj_d10": pb["adj_d10"],
                    "a_adj_d_season": pa["adj_d_season"], "b_adj_d_season": pb["adj_d_season"],
                    "expected_pace": pace,
                    "mean_pace_l10": np.mean([pa["pace10"], pb["pace10"]]),
                    "pace_gap_l10": abs(pa["pace10"] - pb["pace10"]),
                    "mean_pace_season": np.mean([pa["pace_season"], pb["pace_season"]]),
                    "mean_adj_ortg_l10": np.mean([pa["adj_o10"], pb["adj_o10"]]),
                    "adj_ortg_gap_l10": abs(pa["adj_o10"] - pb["adj_o10"]),
                    "mean_adj_ortg_season": np.mean([pa["adj_o_season"], pb["adj_o_season"]]),
                    "mean_adj_drtg_l10": np.mean([pa["adj_d10"], pb["adj_d10"]]),
                    "adj_drtg_gap_l10": abs(pa["adj_d10"] - pb["adj_d10"]),
                    "mean_adj_drtg_season": np.mean([pa["adj_d_season"], pb["adj_d_season"]]),
                    "mean_raw_ortg_l10": np.mean([pa["raw_o10"], pb["raw_o10"]]),
                    "mean_raw_drtg_l10": np.mean([pa["raw_d10"], pb["raw_d10"]]),
                    "expected_a_ortg": ae, "expected_b_ortg": be,
                    "formula_total": pace * (ae + be) / 100.0,
                    "max_history_date": lastdate,
                }
                output.append(row)

            # Compute each game's adjusted observations from opponent form and
            # league baselines as known before the whole date began.
            for r, oppid in ((a, bid), (b, aid)):
                opp_rows = [x for x in history[oppid] if x["season"] == season][-10:]
                opp_o = mean(opp_rows, "off_rating", lg["ortg"])
                opp_d = mean(opp_rows, "def_rating", lg["drtg"])
                updates.append((str(r["team_id"]), {
                    "season": season, "date": date, "game_id": gid,
                    "poss": float(r["poss"]), "pace": float(r["pace"]),
                    "off_rating": float(r["off_rating"]), "def_rating": float(r["def_rating"]),
                    "adj_o": float(r["off_rating"]) - (opp_d - lg["drtg"]),
                    "adj_d": float(r["def_rating"]) - (opp_o - lg["ortg"]),
                }))

        # Date-batch update: no result from this date can influence another
        # same-date game's features.
        for teamid, update in updates:
            history[teamid].append(update)
        for season, gid, rows in today:
            for r in rows:
                league_history[season].append({
                    "off_rating": float(r["off_rating"]),
                    "def_rating": float(r["def_rating"]),
                    "pace": float(r["pace"]),
                })

    return pd.DataFrame(output)


def make_team_lookup(teams):
    t = teams[teams.is_nba.fillna(False).astype(bool)]
    lookup = {}
    for r in t.to_dict("records"):
        k = team_key(r["full_name"])
        if k in lookup and lookup[k] != str(r["team_id"]):
            raise RuntimeError(f"Ambiguous team lookup: {r['full_name']}")
        lookup[k] = str(r["team_id"])
    for alias, canonical in (("los angeles clippers", "la clippers"),
                             ("los angeles lakers", "la lakers")):
        if team_key(canonical) in lookup:
            lookup[team_key(alias)] = lookup[team_key(canonical)]
    return lookup


def mae(x):
    return float(np.mean(np.abs(np.asarray(x, dtype=float))))


def run_model(features, teams):
    m = pd.read_csv(ARCHIVE)
    m["game_date"] = pd.to_datetime(m.game_date).dt.normalize()
    lookup = make_team_lookup(teams)
    m["home_id"] = m.home_team.map(lambda x: lookup.get(team_key(x)))
    m["away_id"] = m.away_team.map(lambda x: lookup.get(team_key(x)))
    if m[["home_id", "away_id"]].isna().any().any():
        bad = m[m.home_id.isna() | m.away_id.isna()]
        raise RuntimeError("Unmapped market teams: " + str(bad[["home_team", "away_team"]].drop_duplicates().values.tolist()))

    f = features.copy()
    f["pair_key"] = f.apply(lambda r: (r.game_date, *sorted([str(r.team_a_id), str(r.team_b_id)])), axis=1)
    m["pair_key"] = m.apply(lambda r: (r.game_date, *sorted([str(r.home_id), str(r.away_id)])), axis=1)
    if f.pair_key.duplicated().any():
        raise RuntimeError("Duplicate date/team-pair feature row")
    joined = m.merge(f, on="pair_key", how="left", validate="one_to_one", suffixes=("_market", ""))
    if joined.formula_total.isna().any():
        raise RuntimeError(f"Only {joined.formula_total.notna().sum()}/{len(joined)} market rows matched features")

    # Score provenance audit against NBA Stats; use the archive's saved total for
    # comparable historical evaluation and list disagreements explicitly.
    stats_home = np.where(joined.home_id == joined.team_a_id, joined.team_a_points, joined.team_b_points)
    stats_away = np.where(joined.away_id == joined.team_a_id, joined.team_a_points, joined.team_b_points)
    joined["nba_stats_home_points"] = stats_home
    joined["nba_stats_away_points"] = stats_away
    joined["nba_stats_total"] = stats_home + stats_away
    joined["score_mismatch"] = ((stats_home != joined.actual_home_points) | (stats_away != joined.actual_away_points))

    train = features[(features.game_date < VAL_START) & (features.game_type == "regular_season")]
    val = features[(features.game_date >= VAL_START) & (features.game_date < VAL_END) & (features.game_type == "regular_season")]
    final_train = features[(features.game_date < HOLDOUT_START) & (features.game_type == "regular_season")]
    if min(len(train), len(final_train)) < 1000 or len(val) < 100:
        raise RuntimeError(f"Insufficient chronological samples train={len(train)}, val={len(val)}, final={len(final_train)}")
    for frame in (train, val, final_train):
        if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
            raise RuntimeError("Non-finite feature values")

    alphas = []
    for alpha in ALPHAS:
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(train[FEATURES], train.actual_total_source)
        p = model.predict(val[FEATURES])
        alphas.append({"alpha": alpha, "validation_mae": mae(p - val.actual_total_source.to_numpy())})
    best = min(alphas, key=lambda x: (x["validation_mae"], x["alpha"]))["alpha"]
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=best))])
    model.fit(final_train[FEATURES], final_train.actual_total_source)
    joined["ridge_total"] = model.predict(joined[FEATURES])
    joined["formula_error"] = joined.formula_total - joined.actual_total
    joined["ridge_error"] = joined.ridge_total - joined.actual_total
    joined["market_error"] = joined.total_close - joined.actual_total

    metrics = {}
    for name, err in (
        ("advanced_formula", joined.formula_error),
        ("opponent_adjusted_ridge", joined.ridge_error),
        ("draftkings_close", joined.market_error),
    ):
        metrics[name] = {
            "games": int(len(err)),
            "mae": mae(err),
            "rmse": float(np.sqrt(np.mean(np.asarray(err, float) ** 2))),
            "bias": float(np.mean(err)),
        }
    metrics["production_points_engine_prior_report"] = {
        "games": 181, "mae": 16.688,
        "source": "2026-10-30 profitability diagnostic; historical reported figure, not recomputed here"
    }
    segments = {}
    for label, part in (
        ("through_2025-12-18", joined[joined.game_date < "2025-12-19"]),
        ("2025-12-19_to_2025-12-23", joined[joined.game_date >= "2025-12-19"]),
    ):
        segments[label] = {
            "games": int(len(part)), "formula_mae": mae(part.formula_error),
            "ridge_mae": mae(part.ridge_error), "market_close_mae": mae(part.market_error),
        }

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": "llimllib/nba_data v2 team_game_logs",
        "market_archive_games": int(len(joined)),
        "archive_start": str(joined.game_date.min().date()),
        "archive_end": str(joined.game_date.max().date()),
        "features_total": int(len(features)),
        "features_regular_season": int((features.game_type == "regular_season").sum()),
        "validation_train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "final_train_rows": int(len(final_train)),
        "final_train_latest_date": str(final_train.game_date.max().date()),
        "ridge_alpha": best,
        "alpha_scores": alphas,
        "score_mismatch_count_vs_nba_stats": int(joined.score_mismatch.sum()),
        "score_mismatch_game_ids": joined.loc[joined.score_mismatch, "game_id"].astype(str).tolist(),
        "metrics": metrics,
        "segments": segments,
        "same_day_results_excluded": True,
        "all_feature_history_dates_strictly_prior": bool((features.max_history_date < features.game_date).all()),
        "training_types": ["regular_season"],
        "market_types": sorted(joined.game_type.unique().tolist()),
        "warning": "Retrospective exploratory accuracy comparison only. Closing prices are not verified entry prices. No ROI/profitability conclusion.",
    }
    return joined, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    con, meta = connect()
    raw, teams = load_data(con)
    meta_frame = pd.DataFrame([
        {"dataset": k, **{x: meta[k].get(x) for x in ("first_season", "last_season", "seasons", "updated")}}
        for k in ("team_game_logs", "games", "team_seasons")
    ])
    meta_frame.to_csv(OUT / "source_metadata.csv", index=False)

    features = build_features(raw)
    if features.empty or not (features.max_history_date < features.game_date).all():
        raise RuntimeError("Feature dataset empty or leakage check failed")
    joined, summary = run_model(features, teams)
    features.to_parquet(OUT / "advanced_totals_pregame_features.parquet", index=False)
    features.to_csv(OUT / "advanced_totals_pregame_features.csv", index=False)
    joined.to_csv(OUT / "market_archive_predictions.csv", index=False)
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Advanced NBA totals experiment",
        "",
        f"Run UTC: {summary['run_utc']}",
        f"Point-in-time feature rows: {summary['features_total']:,}",
        f"Regular-season feature rows: {summary['features_regular_season']:,}",
        f"Chronological validation training rows: {summary['validation_train_rows']:,}",
        f"Chronological validation rows: {summary['validation_rows']:,}",
        f"Final training rows before archive: {summary['final_train_rows']:,}",
        f"Final training latest date: {summary['final_train_latest_date']}",
        f"181-game archive rows: {summary['market_archive_games']}",
        f"Same-day results excluded: {summary['same_day_results_excluded']}",
        f"All feature history dates strictly prior: {summary['all_feature_history_dates_strictly_prior']}",
        f"NBA Stats vs market archive score mismatches: {summary['score_mismatch_count_vs_nba_stats']} ({', '.join(summary['score_mismatch_game_ids']) or 'none'})",
        "",
        "## Accuracy metrics",
        "",
        "| Predictor | Games | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in summary["metrics"].items():
        lines.append(f"| {k} | {v['games']} | {v['mae'] if 'mae' in v else '—'} | {v.get('rmse', '—')} | {v.get('bias', '—')} |")
    lines += ["", "## Chronological segments", "", "| Segment | Games | Formula MAE | Ridge MAE | Closing-line MAE |", "|---|---:|---:|---:|---:|"]
    for k, v in summary["segments"].items():
        lines.append(f"| {k} | {v['games']} | {v['formula_mae']:.3f} | {v['ridge_mae']:.3f} | {v['market_close_mae']:.3f} |")
    lines += ["", "## Ridge alpha validation", "", "| Alpha | Validation MAE |", "|---:|---:|"]
    for v in summary["alpha_scores"]:
        lines.append(f"| {v['alpha']} | {v['validation_mae']:.3f} |")
    lines += [
        "", "## Interpretation safeguards", "",
        "- Training includes regular-season games only; All-Star, preseason, playoffs and play-in are excluded.",
        "- Market archive rows retain the NBA Cup Final; the Cup Final is not used as a model-training target.",
        "- Every team's features use only team games whose game_date is strictly earlier than the target date. All games on a date are featured before any same-day results are added to history.",
        "- Opponent-adjusted efficiencies correct each past team's efficiency by opponent ratings known before that past game.",
        "- Ridge regularization is selected using an earlier chronological validation window; final model training ends before 2025-11-26.",
        "- This archive has already been examined in previous work; it is not an untouched confirmatory holdout.",
        "- Closing-line MAE measures forecast accuracy against the market, not executable entry-price ROI or betting profitability.",
        "- Production code and the deployed Points Engine are not imported, modified, retrained or deployed.",
        "", summary["warning"], "",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(lines))
    print((OUT / "experiment_report.md").read_text())


if __name__ == "__main__":
    main()
