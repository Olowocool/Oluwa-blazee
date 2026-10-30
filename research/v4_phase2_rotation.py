#!/usr/bin/env python3
"""V4 Phase 2: point-in-time player RAPM, injury status and rotation minutes.

Research-only. No production model, frontend, deployment or paper ledger is touched.
Evaluation: 2024-25 and 2025-26, trained chronologically on earlier seasons.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd
import requests
from PyPDF2 import PdfReader
from rapidfuzz import process, fuzz
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from nbainjuries import injury
from nbainjuries._util import _gen_filepath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_phase1_core as phase1

BASE = "https://basketball-data.billmill.org"
SEASONS = [2022, 2023, 2024, 2025, 2026]
ROTATION_SEASONS = [2023, 2024, 2025, 2026]
TARGET_SEASONS = [2024, 2025, 2026]
OUT = Path("outputs/research/v4_phase2")
CACHE = Path("data/research/v4_phase2_injury_reports")
RAPM_ALPHA = 1000.0
RATING_HALF_LIFE_DAYS = 60.0
RATING_REFRESH_DAYS = 14
RAPM_EXPOSURE_SHRINK = 500.0
MIN_ROTATION_MINUTES = 1.0
MAX_ROTATION_PLAYERS = 15
PLAYER_MINUTES_WINDOW = 10
PLAYER_MINUTES_HALF_LIFE = 5.0
STATUS_PRIOR_STRENGTH = 10.0
STATUS_PRIORS = {
    "OUT": 0.02,
    "DOUBTFUL": 0.10,
    "QUESTIONABLE": 0.50,
    "PROBABLE": 0.85,
    "AVAILABLE": 0.98,
}

def parquet_urls(source: str, table: str, seasons: list[int]) -> str:
    return "[" + ",".join(
        "'" + f"{BASE}/nba/{source}/{table}/season={s}/data.parquet" + "'"
        for s in seasons
    ) + "]"


def read_table(con, source, table, seasons, columns="*"):
    urls = parquet_urls(source, table, seasons)
    return con.execute(
        f"SELECT {columns} FROM read_parquet({urls}, hive_partitioning=true, "
        "hive_types={'season': INTEGER}, union_by_name=true)"
    ).fetchdf()


def load_sources():
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    games = read_table(
        con, "stats", "games", SEASONS,
        "season, game_id, game_date, game_type, home_team_id, away_team_id, neutral_site"
    )
    games["game_date"] = pd.to_datetime(games.game_date).dt.normalize()
    games = games[games.game_type.astype(str).str.lower().eq("regular_season")].copy()
    games["game_id"] = games.game_id.astype(str)
    games["home_team_id"] = games.home_team_id.astype(str)
    games["away_team_id"] = games.away_team_id.astype(str)

    team_logs = read_table(
        con, "stats", "team_game_logs", SEASONS,
        "season, game_id, game_date, team_id, opp_team_id, pts, poss, home"
    )
    team_logs["game_id"] = team_logs.game_id.astype(str)
    team_logs["team_id"] = team_logs.team_id.astype(str)
    team_logs["game_date"] = pd.to_datetime(team_logs.game_date).dt.normalize()
    team_logs = team_logs.merge(
        games[["season", "game_id"]], on=["season", "game_id"], how="inner", validate="many_to_one"
    )

    player_logs = read_table(
        con, "stats", "player_game_logs", SEASONS,
        "season, game_id, game_date, player_id, team_id, min"
    )
    player_logs["game_id"] = player_logs.game_id.astype(str)
    player_logs["player_id"] = player_logs.player_id.astype(str)
    player_logs["team_id"] = player_logs.team_id.astype(str)
    player_logs["game_date"] = pd.to_datetime(player_logs.game_date).dt.normalize()
    player_logs["min"] = pd.to_numeric(player_logs["min"], errors="coerce").fillna(0.0)
    player_logs = player_logs.merge(
        games[["season", "game_id"]], on=["season", "game_id"], how="inner", validate="many_to_one"
    )

    player_names = read_table(con, "stats", "player_seasons", SEASONS, "season, player_id, name")
    player_names["player_id"] = player_names.player_id.astype(str)
    team_names = read_table(con, "stats", "team_seasons", SEASONS, "season, team_id, full_name")
    team_names["team_id"] = team_names.team_id.astype(str)

    rotations = read_table(
        con, "stats", "game_rotations", ROTATION_SEASONS,
        "season, game_id, team_id, player_id, period, in_time, out_time, plus_minus, matches_box_score"
    )
    rotations["game_id"] = rotations.game_id.astype(str)
    rotations["team_id"] = rotations.team_id.astype(str)
    rotations["player_id"] = rotations.player_id.astype(str)
    rotations = rotations.merge(
        games[["season", "game_id", "game_date", "home_team_id", "away_team_id", "neutral_site"]],
        on=["season", "game_id"], how="inner", validate="many_to_one"
    )

    events = read_table(
        con, "stats", "game_events", ROTATION_SEASONS,
        "season, game_id, period, time, team_id, pts"
    )
    events["game_id"] = events.game_id.astype(str)
    events["team_id"] = events.team_id.astype(str)
    events["pts"] = pd.to_numeric(events.pts, errors="coerce").fillna(0).astype(int)
    events["time"] = pd.to_numeric(events.time, errors="coerce")
    events = events[(events.pts > 0) & events.game_id.isin(set(games.game_id))].copy()

    return games, team_logs, player_logs, player_names, team_names, rotations, events


def build_lineup_segments(games, team_logs, rotations, events):
    """Reconstruct common five-on-five lineup intervals and verify scores/plus-minus."""
    game_info = games.set_index("game_id").to_dict("index")
    poss_map, score_map = {}, {}
    for gid, group in team_logs.groupby("game_id"):
        poss_map[str(gid)] = {
            str(r.team_id): float(r.poss) for r in group.itertuples()
            if pd.notna(r.poss) and float(r.poss) > 0
        }
        score_map[str(gid)] = {str(r.team_id): int(r.pts) for r in group.itertuples()}
    event_groups = {}
    for (gid, period), group in events.groupby(["game_id", "period"], sort=False):
        group = group.sort_values("time")
        event_groups[(str(gid), int(period))] = group

    segments = []
    audit = {
        "regular_season_games_with_rotation_rows": int(rotations.game_id.nunique()),
        "games_reconstructed": 0,
        "games_exact_score_match": 0,
        "games_player_stint_plusminus_match": 0,
        "games_skipped_bad_rotation_flag": 0,
        "games_skipped_lineup_or_event_gap": 0,
        "games_skipped_score_mismatch": 0,
        "games_skipped_player_plusminus_mismatch": 0,
        "segments": 0,
    }
    for gid, rgame in rotations.groupby("game_id", sort=False):
        gid = str(gid)
        info = game_info.get(gid)
        if info is None:
            continue
        if not rgame.matches_box_score.fillna(False).all():
            audit["games_skipped_bad_rotation_flag"] += 1
            continue
        home_id, away_id = str(info["home_team_id"]), str(info["away_team_id"])
        game_poss = poss_map.get(gid, {})
        if home_id not in game_poss or away_id not in game_poss:
            audit["games_skipped_lineup_or_event_gap"] += 1
            continue
        avg_poss = (game_poss[home_id] + game_poss[away_id]) / 2.0
        total_seconds = float(rgame.out_time.max())
        if not np.isfinite(total_seconds) or total_seconds <= 0:
            audit["games_skipped_lineup_or_event_gap"] += 1
            continue

        game_segments = []
        pm_accum = defaultdict(int)
        score_sums = {home_id: 0, away_id: 0}
        for period, period_rows in rgame.groupby("period", sort=True):
            period_rows = period_rows.copy()
            boundaries = sorted(set(period_rows.in_time.astype(float)) | set(period_rows.out_time.astype(float)))
            ev = event_groups.get((gid, int(period)))
            if ev is None:
                ev_times = np.array([], dtype=float)
                ev_home = np.array([], dtype=int)
                ev_away = np.array([], dtype=int)
            else:
                ev_times = ev.time.to_numpy(dtype=float)
                ev_team = ev.team_id.astype(str).to_numpy()
                ev_pts = ev.pts.to_numpy(dtype=int)
                ev_home = np.where(ev_team == home_id, ev_pts, 0)
                ev_away = np.where(ev_team == away_id, ev_pts, 0)
                ev_home = np.cumsum(ev_home)
                ev_away = np.cumsum(ev_away)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                duration = float(end - start)
                if duration <= 0.5:
                    continue
                active = period_rows[
                    (period_rows.in_time.astype(float) <= float(start) + 1e-6)
                    & (period_rows.out_time.astype(float) >= float(end) - 1e-6)
                ]
                home_rows = active[active.team_id.astype(str) == home_id]
                away_rows = active[active.team_id.astype(str) == away_id]
                if home_rows.player_id.nunique() != 5 or away_rows.player_id.nunique() != 5:
                    continue
                if len(home_rows) != 5 or len(away_rows) != 5:
                    continue
                left = int(np.searchsorted(ev_times, float(start), side="right"))
                right = int(np.searchsorted(ev_times, float(end), side="right"))
                hp = int((ev_home[right - 1] if right else 0) - (ev_home[left - 1] if left else 0))
                ap = int((ev_away[right - 1] if right else 0) - (ev_away[left - 1] if left else 0))
                poss = avg_poss * duration / total_seconds
                if poss < 0.10:
                    continue
                hline = tuple(sorted(home_rows.player_id.astype(str)))
                aline = tuple(sorted(away_rows.player_id.astype(str)))
                margin = hp - ap
                for row in active.itertuples():
                    key = (int(row.period), str(row.team_id), str(row.player_id), float(row.in_time), float(row.out_time))
                    pm_accum[key] += margin if str(row.team_id) == home_id else -margin
                game_segments.append({
                    "season": int(info["season"]), "game_id": gid, "game_date": info["game_date"],
                    "home_team_id": home_id, "away_team_id": away_id,
                    "home_lineup": hline, "away_lineup": aline,
                    "home_pts": hp, "away_pts": ap, "poss": float(poss),
                    "home_flag": 0 if bool(info.get("neutral_site", False)) else 1,
                })
                score_sums[home_id] += hp
                score_sums[away_id] += ap

        expected_scores = score_map.get(gid, {})
        if not game_segments:
            audit["games_skipped_lineup_or_event_gap"] += 1
            continue
        if score_sums[home_id] != expected_scores.get(home_id) or score_sums[away_id] != expected_scores.get(away_id):
            audit["games_skipped_score_mismatch"] += 1
            continue

        # Every reconstructed player's on-court margin must add to the source stint plus-minus.
        mismatch = False
        for row in rgame.itertuples():
            key = (int(row.period), str(row.team_id), str(row.player_id), float(row.in_time), float(row.out_time))
            if pm_accum.get(key, 0) != int(row.plus_minus):
                mismatch = True
                break
        if mismatch:
            audit["games_skipped_player_plusminus_mismatch"] += 1
            continue

        segments.extend(game_segments)
        audit["games_reconstructed"] += 1
        audit["games_exact_score_match"] += 1
        audit["games_player_stint_plusminus_match"] += 1
    audit["segments"] = len(segments)
    result = pd.DataFrame(segments)
    if result.empty:
        raise RuntimeError(f"No valid lineup segments reconstructed; audit={audit}")
    return result, audit


def fit_player_rapm(history: pd.DataFrame, snapshot_date: pd.Timestamp):
    """Fit separate player offense/defense coefficients from five-on-five scoring stints."""
    if history.empty or len(history) < 1000:
        return {}
    h = history.copy()
    player_ids = sorted({p for line in h.home_lineup for p in line} | {p for line in h.away_lineup for p in line})
    team_ids = sorted(set(h.home_team_id.astype(str)) | set(h.away_team_id.astype(str)))
    pidx = {p: i for i, p in enumerate(player_ids)}
    tidx = {t: i for i, t in enumerate(team_ids)}
    npj, nt = len(player_ids), len(team_ids)
    home_col = 2 * npj + 2 * nt
    rows, cols, vals = [], [], []
    y = np.empty(2 * len(h), dtype=float)
    weights = np.empty(2 * len(h), dtype=float)
    exposure = defaultdict(float)

    for i, seg in enumerate(h.itertuples()):
        poss = float(seg.poss)
        age_days = max(0.0, float((snapshot_date - seg.game_date).days))
        if int(seg.season) == int(snapshot_date.year if snapshot_date.month >= 7 else snapshot_date.year + 1):
            recency = math.exp(-math.log(2.0) * age_days / RATING_HALF_LIFE_DAYS)
        else:
            recency = 0.25
        wt = poss * recency
        rh, ra = 2 * i, 2 * i + 1
        for pid in seg.home_lineup:
            rows.extend([rh, ra])
            cols.extend([pidx[pid], npj + pidx[pid]])
            vals.extend([1.0, 1.0])
            exposure[pid] += wt
        for pid in seg.away_lineup:
            rows.extend([rh, ra])
            cols.extend([npj + pidx[pid], pidx[pid]])
            vals.extend([1.0, 1.0])
            exposure[pid] += wt
        rows.extend([rh, rh, ra, ra])
        cols.extend([
            2 * npj + tidx[str(seg.home_team_id)],
            2 * npj + nt + tidx[str(seg.away_team_id)],
            2 * npj + tidx[str(seg.away_team_id)],
            2 * npj + nt + tidx[str(seg.home_team_id)],
        ])
        vals.extend([1.0, 1.0, 1.0, 1.0])
        if int(seg.home_flag):
            rows.append(rh); cols.append(home_col); vals.append(1.0)
        y[rh] = 100.0 * float(seg.home_pts) / poss
        y[ra] = 100.0 * float(seg.away_pts) / poss
        weights[rh] = wt
        weights[ra] = wt

    x = coo_matrix((vals, (rows, cols)), shape=(2 * len(h), home_col + 1)).tocsr()
    model = Ridge(alpha=RAPM_ALPHA, fit_intercept=True, solver="lsqr", max_iter=1500, tol=1e-4)
    model.fit(x, y, sample_weight=weights)
    coef = model.coef_
    ratings = {}
    for pid, idx in pidx.items():
        exp = float(exposure[pid])
        shrink = exp / (exp + RAPM_EXPOSURE_SHRINK)
        ratings[pid] = {
            "o_rapm": float(coef[idx] * shrink),
            "d_rapm": float(coef[npj + idx] * shrink),
            "net_rapm": float((coef[idx] + coef[npj + idx]) * shrink),
            "weighted_possessions": exp,
        }
    return ratings


def build_rapm_snapshots(segments):
    snapshots = {}
    manifest = []
    for season in TARGET_SEASONS:
        current = segments[segments.season == season]
        prior = segments[segments.season == season - 1]
        if current.empty or prior.empty:
            raise RuntimeError(f"Missing current/prior season lineup stints for {season}")
        first = pd.Timestamp(current.game_date.min()).normalize()
        last = pd.Timestamp(current.game_date.max()).normalize()
        dates = [first]
        d = first + pd.Timedelta(days=RATING_REFRESH_DAYS)
        while d <= last:
            dates.append(d)
            d += pd.Timedelta(days=RATING_REFRESH_DAYS)
        for snap in dates:
            hist = pd.concat([prior, current[current.game_date < snap]], ignore_index=True)
            if hist.empty or pd.to_datetime(hist.game_date).max() >= snap:
                raise RuntimeError(f"RAPM snapshot {snap.date()} contains same-day/future stints")
            ratings = fit_player_rapm(hist, snap)
            if not ratings:
                raise RuntimeError(f"RAPM fit failed for {season} snapshot {snap.date()}")
            snapshots[(season, snap)] = ratings
            manifest.append({
                "season": season, "snapshot_date": str(snap.date()),
                "history_rows": int(len(hist)), "max_history_date": str(pd.to_datetime(hist.game_date).max().date()),
                "rated_players": int(len(ratings)),
            })
    return snapshots, pd.DataFrame(manifest)


def normalize_name(value: str) -> str:
    s = str(value or "").strip()
    if "," in s:
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(tokens)


def season_for_date(d: pd.Timestamp) -> int:
    return int(d.year + 1 if d.month >= 7 else d.year)


def fetch_injury_reports(game_dates):
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for game_date in sorted(pd.Timestamp(x).normalize() for x in game_dates):
        # The official archive changed from hourly names to 15-minute names in Dec 2025.
        if game_date.date() in {date(2025, 12, 20), date(2025, 12, 21)}:
            manifest.append({"game_date": str(game_date.date()), "status": "archive_timestamp_gap"})
            continue
        candidates = [time(10, 0), time(9, 0), time(8, 0)]
        if game_date >= pd.Timestamp("2025-12-22"):
            candidates = [time(10, 0), time(9, 45), time(9, 30), time(9, 15), time(9, 0), time(8, 45), time(8, 30), time(8, 15), time(8, 0)]
        found = False
        errors = []
        for clock in candidates:
            ts = datetime.combine(game_date.date(), clock)
            try:
                url = injury.gen_url(ts)
                path = Path(_gen_filepath(ts, CACHE))
                if not path.exists():
                    response = session.get(url, timeout=45)
                    if response.status_code in (403, 404):
                        errors.append(f"{clock}:HTTP{response.status_code}")
                        continue
                    response.raise_for_status()
                    path.write_bytes(response.content)
                pdf = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if not text.strip():
                    errors.append(f"{clock}:no_pdf_text")
                    continue
                if "NOT YET SUBMITTED" in text.upper():
                    errors.append(f"{clock}:unsubmitted")
                    continue
                frame = injury.get_reportdata(ts, local=True, localdir=CACHE, return_df=True)
                if frame.empty:
                    errors.append(f"{clock}:empty_table")
                    continue
                frame["game_date"] = pd.to_datetime(frame["Game Date"], errors="coerce").dt.normalize()
                frame = frame[frame.game_date == game_date].copy()
                if frame.empty:
                    errors.append(f"{clock}:no_target_game_date")
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                frame["snapshot_time"] = ts.isoformat()
                frame["snapshot_url"] = url
                frame["snapshot_sha256"] = digest
                rows.extend(frame.to_dict("records"))
                manifest.append({
                    "game_date": str(game_date.date()), "status": "complete",
                    "snapshot_time": ts.isoformat(), "url": url, "sha256": digest,
                    "rows": int(len(frame)), "pdf_pages": len(pdf.pages),
                })
                found = True
                break
            except Exception as exc:
                errors.append(f"{clock}:{type(exc).__name__}:{str(exc)[:160]}")
        if not found:
            manifest.append({"game_date": str(game_date.date()), "status": "missing_or_incomplete", "attempts": errors})
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def map_injury_rows(raw_reports, games, player_names, team_names):
    if raw_reports.empty:
        raise RuntimeError("No complete official injury reports were retrieved")
    game_lookup = {}
    for row in games.itertuples():
        game_lookup[(int(row.season), pd.Timestamp(row.game_date), str(row.home_team_id))] = str(row.game_id)
        game_lookup[(int(row.season), pd.Timestamp(row.game_date), str(row.away_team_id))] = str(row.game_id)

    mapped = []
    unmatched = []
    names_by_season = {}
    for season, group in player_names.groupby("season"):
        m = defaultdict(list)
        for row in group.itertuples():
            m[normalize_name(row.name)].append(str(row.player_id))
        names_by_season[int(season)] = m
    teams_by_season = {}
    for season, group in team_names.groupby("season"):
        m = {}
        for row in group.itertuples():
            m[normalize_name(row.full_name)] = str(row.team_id)
        teams_by_season[int(season)] = m

    for row in raw_reports.itertuples(index=False, name=None):
        gd = pd.Timestamp(row[7]).normalize()
        season = season_for_date(gd)
        team_name = str(row[3])
        raw_name = str(row[4])
        status_value = str(row[5])
        snapshot_value = str(row[8])
        tmap = teams_by_season.get(season, {})
        tnorm = normalize_name(team_name)
        team_id = tmap.get(tnorm)
        if team_id is None and tmap:
            hit = process.extractOne(tnorm, list(tmap.keys()), scorer=fuzz.ratio)
            if hit and hit[1] >= 95:
                team_id = tmap[hit[0]]
        if team_id is None:
            unmatched.append({"game_date": str(gd.date()), "team": team_name, "player": raw_name, "reason": "team_unmatched"})
            continue
        gid = game_lookup.get((season, gd, team_id))
        if gid is None:
            unmatched.append({"game_date": str(gd.date()), "team": str(row.Team), "player": str(row._asdict().get("Player Name","")), "reason": "team_game_unmatched"})
            continue
        pnorm = normalize_name(raw_name)
        pmap = names_by_season.get(season, {})
        candidates = pmap.get(pnorm, [])
        player_id = candidates[0] if len(candidates) == 1 else None
        if player_id is None and pmap:
            all_names = list(pmap.keys())
            hits = process.extract(pnorm, all_names, scorer=fuzz.ratio, limit=2)
            if hits and hits[0][1] >= 97 and (len(hits) == 1 or hits[0][1] - hits[1][1] >= 5):
                ids = pmap[hits[0][0]]
                if len(ids) == 1:
                    player_id = ids[0]
        if player_id is None:
            unmatched.append({"game_date": str(gd.date()), "team": str(row.Team), "player": raw_name, "reason": "player_unmatched"})
            continue
        status = status_value.strip().upper()
        if status not in STATUS_PRIORS:
            status = "UNKNOWN"
        mapped.append({
            "season": season, "game_date": gd, "game_id": gid, "team_id": team_id,
            "player_id": player_id, "status": status,
            "snapshot_time": snapshot_value,
        })
    mapped_df = pd.DataFrame(mapped)
    unmatched_df = pd.DataFrame(unmatched)
    if mapped_df.empty:
        raise RuntimeError(f"No injury rows mapped to player/game ids; unmatched={len(unmatched_df)}")
    return mapped_df, unmatched_df


def build_player_history(player_logs):
    team_hist, player_hist, first_seen = defaultdict(list), defaultdict(list), {}
    for row in player_logs.sort_values("game_date").itertuples():
        item = (pd.Timestamp(row.game_date), str(row.game_id), str(row.team_id), float(row.min))
        team_hist[(str(row.team_id), str(row.player_id))].append(item)
        player_hist[str(row.player_id)].append(item)
        key = (str(row.team_id), str(row.player_id))
        first_seen[key] = min(first_seen.get(key, pd.Timestamp.max), pd.Timestamp(row.game_date))
    return team_hist, player_hist, first_seen


def recent_minutes(hist, game_date):
    dates = [x[0] for x in hist]
    ix = bisect_left(dates, game_date)
    prior = hist[max(0, ix - PLAYER_MINUTES_WINDOW):ix]
    if not prior:
        return 0.0
    prior = list(reversed(prior))
    weights = np.array([math.exp(-math.log(2.0) * i / PLAYER_MINUTES_HALF_LIFE) for i in range(len(prior))])
    mins = np.array([x[3] for x in prior], dtype=float)
    wsum = float(weights.sum())
    mean = float(np.average(mins, weights=weights))
    n = len(prior)
    # Stabilise sparse player histories toward a low-rotation 12-minute role.
    return float((mean * wsum + 5.0 * 12.0) / (wsum + 5.0)) if n < 5 else mean


def status_probability(status, history, game_date):
    prior = STATUS_PRIORS.get(status, 0.50)
    past = history[history.game_date < game_date] if not history.empty else history
    if past.empty:
        return prior
    rows = past[past.status == status]
    n = len(rows)
    if n == 0:
        return prior
    played = float(rows.played.sum())
    return float((played + STATUS_PRIOR_STRENGTH * prior) / (n + STATUS_PRIOR_STRENGTH))


def rotation_for_team(game_id, team_id, game_date, report_rows, team_hist, player_hist, first_seen, actual_lookup, rapm):
    candidate_ids = {
        pid for (tid, pid), first in first_seen.items()
        if tid == team_id and first < game_date
    }
    team_reports = report_rows[report_rows.team_id == team_id]
    candidate_ids.update(team_reports.player_id.astype(str).tolist())
    estimates = {}
    for pid in list(candidate_ids):
        all_hist = player_hist.get(pid, [])
        all_dates = [x[0] for x in all_hist]
        last_idx = bisect_left(all_dates, game_date) - 1
        if last_idx >= 0 and all_hist[last_idx][2] != team_id and pid not in set(team_reports.player_id.astype(str)):
            candidate_ids.discard(pid)
            continue
        hist = team_hist.get((team_id, pid), [])
        if not hist:
            hist = all_hist
        m = recent_minutes(hist, game_date)
        if m > MIN_ROTATION_MINUTES:
            estimates[pid] = m
    if not estimates:
        return None, None
    ranked = sorted(estimates, key=estimates.get, reverse=True)[:MAX_ROTATION_PLAYERS]
    base = np.array([estimates[p] for p in ranked], dtype=float)
    base = base * (240.0 / base.sum())
    status_map = {
        str(r.player_id): str(r.status)
        for r in team_reports.itertuples()
    }
    exp_raw, probs, statuses = [], [], []
    for pid, m in zip(ranked, base):
        status = status_map.get(pid, "NOT_REPORTED")
        prob = 1.0 if status == "NOT_REPORTED" else status_probability(status, STATUS_HISTORY, game_date)
        exp_raw.append(m * prob)
        probs.append(prob)
        statuses.append(status)
    exp_raw = np.asarray(exp_raw, dtype=float)
    if exp_raw.sum() <= 1e-8:
        exp = base.copy()
    else:
        exp = exp_raw * (240.0 / exp_raw.sum())
    d = pd.DataFrame({
        "player_id": ranked, "baseline_minutes": base, "expected_minutes": exp,
        "availability_probability": probs, "status": statuses,
        "actual_minutes": [actual_lookup.get((game_id, p), 0.0) for p in ranked],
    })
    for col in ("o_rapm", "d_rapm"):
        d[col] = [rapm.get(p, {}).get(col, 0.0) for p in ranked]
    d["o_delta_contribution"] = (d.expected_minutes - d.baseline_minutes) / 48.0 * d.o_rapm
    d["d_delta_contribution"] = (d.expected_minutes - d.baseline_minutes) / 48.0 * d.d_rapm
    features = {
        "off_delta": float(d.o_delta_contribution.sum()),
        "def_delta": float(d.d_delta_contribution.sum()),
        "minutes_changed": float(np.abs(d.expected_minutes - d.baseline_minutes).sum() / 2.0),
        "reported_out_minutes": float(d.loc[d.status == "OUT", "baseline_minutes"].sum()),
        "questionable_minutes": float(d.loc[d.status.isin(["QUESTIONABLE", "DOUBTFUL"]), "baseline_minutes"].sum()),
    }
    actual = d.actual_minutes.to_numpy(dtype=float)
    if actual.sum() > 0:
        d["actual_minutes_240"] = actual * (240.0 / actual.sum())
    else:
        d["actual_minutes_240"] = 0.0
    return features, d


# Populated before any target games are evaluated; status_probability reads only earlier dates.
STATUS_HISTORY = pd.DataFrame(columns=["game_date", "status", "played"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games, team_logs, player_logs, player_names, team_names, rotations, events = load_sources()
    segments, segment_audit = build_lineup_segments(games, team_logs, rotations, events)
    segments.to_pickle(OUT / "v4_phase2_lineup_segments.pkl")
    snapshots, snapshot_manifest = build_rapm_snapshots(segments)
    snapshot_manifest.to_csv(OUT / "v4_phase2_rapm_snapshots.csv", index=False)

    # Retrieve only official reports from 10:00 a.m. ET or an earlier available snapshot.
    report_dates = games[games.season.isin(SEASONS)].game_date.drop_duplicates().tolist()
    raw_reports, report_manifest = fetch_injury_reports(report_dates)
    report_manifest.to_csv(OUT / "v4_phase2_injury_report_manifest.csv", index=False)
    mapped_reports, unmatched_reports = map_injury_rows(raw_reports, games, player_names, team_names)
    unmatched_reports.to_csv(OUT / "v4_phase2_unmatched_injury_rows.csv", index=False)

    actual_played = set(zip(player_logs.game_id.astype(str), player_logs.player_id.astype(str)))
    mapped_reports["played"] = [
        1 if (str(g), str(p)) in actual_played else 0
        for g, p in zip(mapped_reports.game_id, mapped_reports.player_id)
    ]
    mapped_reports = mapped_reports.sort_values(["game_date", "game_id", "player_id"])
    mapped_reports.to_csv(OUT / "v4_phase2_mapped_injury_status.csv", index=False)

    # Player minutes history and actual-minute lookup.
    team_hist, player_hist, first_seen = build_player_history(player_logs)
    actual_lookup = {
        (str(r.game_id), str(r.player_id)): float(r.min)
        for r in player_logs.itertuples()
    }
    game_reports = {
        gid: group.copy()
        for gid, group in mapped_reports.groupby("game_id")
    }
    baseline_raw, _ = phase1.connect_and_load()
    baseline = phase1.build_point_in_time(baseline_raw)
    baseline = baseline[baseline.season.isin(TARGET_SEASONS)].copy()
    baseline["game_date"] = pd.to_datetime(baseline.game_date).dt.normalize()
    baseline_lookup = baseline.set_index("game_id").to_dict("index")

    snapshot_dates_by_season = defaultdict(list)
    for (season, snap_date), ratings in snapshots.items():
        snapshot_dates_by_season[season].append(snap_date)
    for season in snapshot_dates_by_season:
        snapshot_dates_by_season[season].sort()

    feature_rows, minute_rows = [], []
    status_rows = mapped_reports[["game_date", "status", "played"]].copy()
    global STATUS_HISTORY
    STATUS_HISTORY = status_rows
    target_games = games[games.season.isin(TARGET_SEASONS)].sort_values(["game_date", "game_id"])
    for g in target_games.itertuples():
        gid, season, gd = str(g.game_id), int(g.season), pd.Timestamp(g.game_date).normalize()
        base = baseline_lookup.get(gid)
        if base is None:
            continue
        report = game_reports.get(gid, pd.DataFrame(columns=mapped_reports.columns))
        snap_dates = snapshot_dates_by_season[season]
        pos = bisect_right(snap_dates, gd) - 1
        if pos < 0:
            continue
        snap_date = snap_dates[pos]
        rapm = snapshots[(season, snap_date)]
        home_report = report[report.team_id == str(g.home_team_id)] if not report.empty else report
        away_report = report[report.team_id == str(g.away_team_id)] if not report.empty else report
        hfeat, hmins = rotation_for_team(
            gid, str(g.home_team_id), gd, home_report, team_hist, player_hist,
            first_seen, actual_lookup, rapm
        )
        afeat, amins = rotation_for_team(
            gid, str(g.away_team_id), gd, away_report, team_hist, player_hist,
            first_seen, actual_lookup, rapm
        )
        if hfeat is None or afeat is None:
            continue
        poss = float(base["expected_possessions"])
        raw_delta = poss / 100.0 * (hfeat["off_delta"] + hfeat["def_delta"] + afeat["off_delta"] + afeat["def_delta"])
        feature_rows.append({
            "season": season, "game_id": gid, "game_date": gd,
            "home_team_id": str(g.home_team_id), "away_team_id": str(g.away_team_id),
            "v4_phase1_total": float(base["projected_total"]),
            "actual_total": float(base["actual_total"]),
            "expected_possessions": poss, "rapm_snapshot_date": snap_date,
            "home_off_delta": hfeat["off_delta"], "home_def_delta": hfeat["def_delta"],
            "away_off_delta": afeat["off_delta"], "away_def_delta": afeat["def_delta"],
            "raw_injury_total_delta": raw_delta,
            "home_minutes_changed": hfeat["minutes_changed"],
            "away_minutes_changed": afeat["minutes_changed"],
            "home_reported_out_minutes": hfeat["reported_out_minutes"],
            "away_reported_out_minutes": afeat["reported_out_minutes"],
            "home_questionable_minutes": hfeat["questionable_minutes"],
            "away_questionable_minutes": afeat["questionable_minutes"],
        })
        for team_side, frame in (("home", hmins), ("away", amins)):
            frame = frame.copy()
            frame["season"] = season
            frame["game_id"] = gid
            frame["game_date"] = gd
            frame["side"] = team_side
            minute_rows.extend(frame.to_dict("records"))

    features = pd.DataFrame(feature_rows)
    minutes = pd.DataFrame(minute_rows)
    if features.empty:
        raise RuntimeError("No injury-adjusted game features were generated")
    features.to_csv(OUT / "v4_phase2_game_features.csv", index=False)
    minutes.to_csv(OUT / "v4_phase2_projected_minutes.csv", index=False)

    # Forward-only residual calibration: 2023-24 trains 2024-25; prior seasons train 2025-26.
    predictions = []
    metric_rows = []
    for season in [2025, 2026]:
        train = features[(features.season < season) & (features.season >= 2024)].dropna()
        test = features[features.season == season].dropna().copy()
        if len(train) < 300 or len(test) < 300:
            raise RuntimeError(f"Insufficient chronological train/test rows for {season}: train={len(train)}, test={len(test)}")
        cols = ["home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta"]
        x_train = train[cols].to_numpy(float)
        y_train = (train.actual_total - train.v4_phase1_total).to_numpy(float)
        x_test = test[cols].to_numpy(float)
        intercept = float(np.mean(y_train))
        intercept_pred = test.v4_phase1_total.to_numpy(float) + intercept
        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(x_train, y_train)
        injury_pred = test.v4_phase1_total.to_numpy(float) + model.predict(x_test)
        v4_pred = test.v4_phase1_total.to_numpy(float)
        actual = test.actual_total.to_numpy(float)
        def metric(name, pred):
            err = pred - actual
            return {
                "season": season, "model": name, "games": int(len(test)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "bias_prediction_minus_actual": float(np.mean(err)),
                "within_5_points_pct": float(np.mean(np.abs(err) <= 5) * 100),
                "within_10_points_pct": float(np.mean(np.abs(err) <= 10) * 100),
            }
        metric_rows.extend([
            metric("V4 Phase 1 baseline", v4_pred),
            metric("V4 Phase 1 + train-season intercept only", intercept_pred),
            metric("V4 Phase 1 + injury/RAPM rotation model", injury_pred),
        ])
        test["intercept_only_total"] = intercept_pred
        test["injury_adjusted_total"] = injury_pred
        test["injury_model_train_games"] = len(train)
        test["injury_model_intercept"] = intercept
        for col, value in zip(cols, model.coef_):
            test[f"coef_{col}"] = float(value)
        predictions.append(test)

    oos = pd.concat(predictions, ignore_index=True)
    oos.to_csv(OUT / "v4_phase2_forward_predictions.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "v4_phase2_metrics.csv", index=False)

    # Independent diagnostics: status calibration and projected minutes.
    status_eval = mapped_reports[mapped_reports.season.isin(TARGET_SEASONS)].copy()
    status_summary = status_eval.groupby("status").agg(
        rows=("played", "size"), appeared=("played", "sum"), empirical_appearance_rate=("played", "mean")
    ).reset_index()
    minute_error = minutes.copy()
    minute_error["absolute_minutes_error"] = np.abs(minute_error.expected_minutes - minute_error.actual_minutes_240)
    minute_summary = minute_error.groupby("season").agg(
        player_game_rows=("player_id", "size"),
        minute_mae=("absolute_minutes_error", "mean"),
        mean_projected_team_minutes=("expected_minutes", "mean"),
        mean_actual_team_minutes_240=("actual_minutes_240", "mean"),
    ).reset_index()
    status_summary.to_csv(OUT / "v4_phase2_status_calibration.csv", index=False)
    minute_summary.to_csv(OUT / "v4_phase2_minutes_metrics.csv", index=False)

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "V4 Phase 2: official pregame injury reports + projected rotations + point-in-time lineup RAPM",
        "source": "llimllib/nba_data v2 (NBA Stats team/player logs, play-by-play-derived game_rotations and game_events) plus official NBA injury report PDFs parsed with nbainjuries 1.1.1",
        "target_seasons_end_year": TARGET_SEASONS,
        "oos_test_seasons_end_year": [2025, 2026],
        "rapm": {
            "method": "Ridge model on reconstructed five-on-five lineup scoring stints; separate player offense and defense coefficients; team offense/defense fixed effects",
            "alpha": RAPM_ALPHA,
            "rating_half_life_days": RATING_HALF_LIFE_DAYS,
            "refresh_days": RATING_REFRESH_DAYS,
            "exposure_shrink_possessions": RAPM_EXPOSURE_SHRINK,
            "all_rating_history_strictly_before_snapshot": True,
        },
        "injury_reports": {
            "snapshot_policy": "latest complete official report at or before 10:00 a.m. ET on game day; earlier 9:00/8:00 a.m. fallback; dates with NOT YET SUBMITTED rows excluded",
            "status_probabilities": "empirical status-to-appearance rates from earlier report dates, shrunk toward conservative status priors",
            "unmatched_rows": int(len(unmatched_reports)),
            "mapped_rows": int(len(mapped_reports)),
            "complete_report_dates": int((report_manifest.status == "complete").sum()),
            "missing_or_incomplete_report_dates": int((report_manifest.status != "complete").sum()),
            "archive_timestamp_gap_dates": int((report_manifest.status == "archive_timestamp_gap").sum()),
        },
        "lineup_reconstruction_audit": segment_audit,
        "forward_metrics": metric_rows,
        "status_calibration": status_summary.to_dict("records"),
        "projected_minutes_metrics": minute_summary.to_dict("records"),
        "feature_games": int(len(features)),
        "oos_games": int(len(oos)),
        "production_modified": False,
        "notes": [
            "OpenRAPM's currently published historical ratings are rebuilt as a single current version across seasons; they were not used as historical pregame features.",
            "Player ratings here are recomputed from lineup stints and scoring events using only prior-season and current-season-to-date games.",
            "Each injury snapshot is captured at or before 10:00 a.m. ET on the game date; incomplete reports are excluded, never treated as all players available.",
            "The official injury report status is converted to an appearance probability using only earlier report-to-game outcomes; unreported players are assumed available.",
            "Projected minutes are recent player-team rotation minutes, shrunk for short histories, then rescaled to 240 regulation minutes.",
            "RAPM values are team-adjusted player on-court effects, not proprietary EPM/DARKO ratings.",
            "Only 2024-25 and 2025-26 are held out for forward injury-model evaluation; 2023-24 is the initial training season.",
            "This phase evaluates total-score MAE/RMSE, not betting ROI or profitability."
        ],
    }
    (OUT / "v4_phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
