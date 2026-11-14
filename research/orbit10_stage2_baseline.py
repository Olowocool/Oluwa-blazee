#!/usr/bin/env python3
"""ORBIT-10 Stage 2: frozen team-strength + possessions combined-total baseline.

Primary game spine:
- 2000-01 through 2008-09: repository data/game.csv.
- 2009-10 through 2025-26: public llimllib/nba_data v2 team_game_logs.
- 2009-10 through 2022-23 overlap: cross-source score and possession audit.

No odds, market totals, injury adjustments, or ORBIT pattern features enter
this baseline. Every game on a calendar date is forecast from history strictly
before that date; same-day games never update one another.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

INPUT = Path("data/game.csv")
OUT = Path("outputs/research/orbit10_stage2")
BASE = "https://basketball-data.billmill.org"
START_DATE = pd.Timestamp("2000-07-01")
LAST_COMPLETE_SEASON_END_YEAR = 2026
ROLLING_GAMES = 25
RECENCY_HALF_LIFE = 10.0
PRIOR_STRENGTH_GAMES = 10.0
MIN_PRIOR_GAMES = 5

LEGACY_COLUMNS = [
    "game_id", "season_id", "game_date", "season_type",
    "team_id_home", "team_id_away",
    "pts_home", "pts_away",
    "fga_home", "fta_home", "oreb_home", "tov_home",
    "fga_away", "fta_away", "oreb_away", "tov_away",
]


def id_text(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and float(number).is_integer():
        return str(int(number))
    return str(value).strip()


def possessions(row: dict, side: str) -> float:
    return float(row[f"fga_{side}"] + 0.44 * row[f"fta_{side}"]
                 - row[f"oreb_{side}"] + row[f"tov_{side}"])


def load_legacy() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Required historical source is missing: {INPUT}")
    header = pd.read_csv(INPUT, nrows=0)
    missing = sorted(set(LEGACY_COLUMNS) - set(header.columns))
    if missing:
        raise RuntimeError(f"data/game.csv schema mismatch; missing columns: {missing}")
    raw = pd.read_csv(INPUT, usecols=LEGACY_COLUMNS, low_memory=False)
    raw_rows = len(raw)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.normalize()
    for col in LEGACY_COLUMNS:
        if col not in ("game_id", "season_type", "game_date"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    # NBA Stats season_id 22010 denotes the 2010-11 season; the
    # season-end year is therefore season_id - 19999, not -20000.
    raw["season_end_year"] = raw["season_id"] - 19999
    raw["season_type_norm"] = raw["season_type"].astype(str).str.strip().str.lower()
    raw = raw[
        raw["season_type_norm"].str.contains("regular", na=False)
        & raw["game_date"].ge(START_DATE)
        & raw["season_end_year"].between(2001, 2023)
    ].copy()

    required = [
        "game_id", "season_end_year", "game_date", "team_id_home", "team_id_away",
        "pts_home", "pts_away", "fga_home", "fta_home", "oreb_home", "tov_home",
        "fga_away", "fta_away", "oreb_away", "tov_away",
    ]
    missing_rows = int(raw[required].isna().any(axis=1).sum())
    raw = raw.dropna(subset=required).copy()
    raw["game_id"] = raw["game_id"].map(id_text)
    raw["team_id_home"] = raw["team_id_home"].map(id_text)
    raw["team_id_away"] = raw["team_id_away"].map(id_text)
    if raw["game_id"].duplicated().any():
        dup = int(raw["game_id"].duplicated(keep=False).sum())
        raise RuntimeError(f"Legacy regular-season duplicate game IDs remain: {dup}")

    rows, overlap = [], []
    for r in raw.to_dict(orient="records"):
        hp, ap = possessions(r, "home"), possessions(r, "away")
        if min(hp, ap) <= 0:
            raise RuntimeError(f"Non-positive possession estimate in legacy game {r['game_id']}")
        common = {
            "game_id": f"legacy_{int(r['season_end_year'])}_{r['game_id']}",
            "source_game_id": r["game_id"],
            "season_end_year": int(r["season_end_year"]),
            "game_date": r["game_date"],
            "home_team_id": r["team_id_home"],
            "away_team_id": r["team_id_away"],
            "pts_home": float(r["pts_home"]),
            "pts_away": float(r["pts_away"]),
            "poss_home": hp,
            "poss_away": ap,
            "game_pace_poss": (hp + ap) / 2.0,
            "source": "repository_game_csv",
        }
        if int(r["season_end_year"]) <= 2009:
            rows.append(common)
        else:
            overlap.extend([
                {"season_end_year": int(r["season_end_year"]), "source_game_id": r["game_id"],
                 "team_id": r["team_id_home"], "game_date": r["game_date"].date().isoformat(),
                 "is_home": True, "pts": float(r["pts_home"]), "formula_poss": hp},
                {"season_end_year": int(r["season_end_year"]), "source_game_id": r["game_id"],
                 "team_id": r["team_id_away"], "game_date": r["game_date"].date().isoformat(),
                 "is_home": False, "pts": float(r["pts_away"]), "formula_poss": ap},
            ])
    audit = {
        "legacy_raw_rows": int(raw_rows),
        "legacy_regular_season_rows_after_required_field_filter": int(len(raw)),
        "legacy_rows_excluded_for_missing_required_values": missing_rows,
        "legacy_baseline_games_used_2000_01_to_2008_09": int(len(rows)),
        "season_id_examples": sorted(raw["season_id"].dropna().astype(int).unique().tolist())[:40],
        "season_end_year_examples_current_mapping": sorted(raw["season_end_year"].dropna().astype(int).unique().tolist())[:40],
        "game_date_year_examples": sorted(raw["game_date"].dt.year.dropna().unique().tolist())[:40],
        "overlap_rows_by_season_end_year": {
            str(int(y)): int(n) for y, n in pd.DataFrame(overlap).groupby("season_end_year").size().items()
        } if overlap else {},
    }
    return pd.DataFrame(rows), pd.DataFrame(overlap), audit


def load_modern() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))["datasets"]
    for name in ("team_game_logs", "games"):
        if name not in metadata:
            raise RuntimeError(f"Public data catalog missing dataset: {name}")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        seasons = list(range(2010, LAST_COMPLETE_SEASON_END_YEAR + 1))
        for name in ("team_game_logs", "games"):
            urls = ", ".join(
                "'" + f"{BASE}/nba/stats/{name}/season={season}/data.parquet" + "'"
                for season in seasons
            )
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet([{urls}], "
                "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
            )
        modern = con.execute("""
            SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
                   t.pts, t.fga, t.fta, t.oreb, t.tov, t.poss AS source_poss,
                   t.home, g.game_type, g.neutral_site
            FROM team_game_logs t
            JOIN games g USING (season, game_id)
            WHERE t.season BETWEEN 2010 AND 2026
              AND g.game_type = 'regular_season'
              AND COALESCE(g.neutral_site, FALSE) = FALSE
            ORDER BY t.game_date, t.season, t.game_id, t.team_id
        """).fetchdf()
    finally:
        con.close()

    if modern.empty:
        raise RuntimeError("No modern regular-season team-game rows loaded.")
    for col in ("pts", "fga", "fta", "oreb", "tov", "source_poss"):
        modern[col] = pd.to_numeric(modern[col], errors="coerce")
    modern["game_date"] = pd.to_datetime(modern["game_date"], errors="coerce").dt.normalize()
    modern["team_id"] = modern["team_id"].map(id_text)
    modern["opp_team_id"] = modern["opp_team_id"].map(id_text)
    modern["game_id"] = modern["game_id"].map(id_text)
    modern = modern.dropna(subset=[
        "season", "game_id", "game_date", "team_id", "opp_team_id",
        "pts", "fga", "fta", "oreb", "tov", "home",
    ]).copy()
    modern["formula_poss"] = (
        modern["fga"] + 0.44 * modern["fta"] - modern["oreb"] + modern["tov"]
    )
    if (modern["formula_poss"] <= 0).any():
        raise RuntimeError("Modern source contains non-positive formula possessions.")
    if modern.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate modern (season, game_id, team_id) keys.")

    modern_rows, cross_rows = [], []
    for (season, gid), group in modern.groupby(["season", "game_id"], sort=False):
        if len(group) != 2:
            raise RuntimeError(f"Modern game {season}/{gid} has {len(group)} team rows, expected 2.")
        home_rows = group[group["home"].astype(bool)]
        away_rows = group[~group["home"].astype(bool)]
        if len(home_rows) != 1 or len(away_rows) != 1:
            raise RuntimeError(f"Modern game {season}/{gid} has invalid home flags.")
        h, a = home_rows.iloc[0], away_rows.iloc[0]
        modern_rows.append({
            "game_id": f"modern_{int(season)}_{gid}",
            "source_game_id": str(gid),
            "season_end_year": int(season),
            "game_date": h["game_date"],
            "home_team_id": h["team_id"],
            "away_team_id": a["team_id"],
            "pts_home": float(h["pts"]),
            "pts_away": float(a["pts"]),
            "poss_home": float(h["formula_poss"]),
            "poss_away": float(a["formula_poss"]),
            "game_pace_poss": float((h["formula_poss"] + a["formula_poss"]) / 2.0),
            "source": "public_team_game_logs",
        })
        for team in (h, a):
            cross_rows.append({
                "season_end_year": int(season),
                "source_game_id": str(gid),
                "game_date": team["game_date"].date().isoformat(),
                "is_home": bool(team["home"]),
                "team_id": team["team_id"],
                "pts": float(team["pts"]),
                "formula_poss": float(team["formula_poss"]),
                "provider_poss": float(team["source_poss"]),
            })

    provider_team = pd.DataFrame(cross_rows)
    modern_games = pd.DataFrame(modern_rows)
    # Cross-source comparison is performed on overlapping 2010-2023 games.
    return modern_games, provider_team, {
        "modern_team_game_rows": int(len(modern)),
        "modern_regular_season_games_2009_10_to_2025_26": int(len(modern_games)),
        "modern_games_by_season_end_year": {
            str(int(y)): int(n) for y, n in modern_games.groupby("season_end_year").size().items()
        },
    }


def weighted_recent(values: list[float]) -> float:
    x = np.asarray(values[-ROLLING_GAMES:], dtype=float)
    if not len(x):
        return float("nan")
    age = np.arange(len(x) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE)
    return float(np.average(x, weights=weights))


def shrunk(value: float, n: int, league_value: float) -> float:
    if not np.isfinite(value):
        return league_value
    effective_n = min(int(n), ROLLING_GAMES)
    return float((effective_n * value + PRIOR_STRENGTH_GAMES * league_value) /
                 (effective_n + PRIOR_STRENGTH_GAMES))


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"games": 0, "mae": None, "rmse": None, "bias_prediction_minus_actual": None}
    err = frame["predicted_total"].to_numpy(float) - frame["actual_total"].to_numpy(float)
    return {
        "games": int(len(frame)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias_prediction_minus_actual": round(float(np.mean(err)), 4),
        "within_5_points_pct": round(float(np.mean(np.abs(err) <= 5) * 100), 2),
        "within_10_points_pct": round(float(np.mean(np.abs(err) <= 10) * 100), 2),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legacy_games, legacy_overlap, legacy_audit = load_legacy()
    modern_games, modern_team, modern_audit = load_modern()

    # Audit raw formula possessions against the modern provider's possession
    # field, and independently compare legacy box-score rows with modern rows
    # over the shared 2010-2023 period.
    provider_delta = modern_team["formula_poss"] - modern_team["provider_poss"]
    provider_poss_audit = {
        "team_game_rows": int(len(modern_team)),
        "mean_abs_formula_vs_provider_poss": round(float(provider_delta.abs().mean()), 5),
        "median_abs_formula_vs_provider_poss": round(float(provider_delta.abs().median()), 5),
        "p95_abs_formula_vs_provider_poss": round(float(provider_delta.abs().quantile(0.95)), 5),
        "correlation": round(float(modern_team[["formula_poss", "provider_poss"]].corr().iloc[0, 1]), 6),
        "within_0_01_possessions_pct": round(float(np.isclose(provider_delta, 0, atol=0.01).mean() * 100), 2),
    }

    modern_overlap = modern_team[modern_team["season_end_year"].between(2010, 2023)].copy()
    # Save forensic counts before enforcing the reconciliation gate.
    game_keys = ["season_end_year", "source_game_id"]
    game_matches = legacy_overlap.merge(modern_overlap, on=game_keys, how="inner", suffixes=("_legacy", "_modern"))
    home_matches = legacy_overlap.merge(modern_overlap, on=game_keys + ["is_home"], how="inner", suffixes=("_legacy", "_modern"))
    date_score_matches = legacy_overlap.merge(modern_overlap, on=["season_end_year", "game_date", "is_home", "pts"], how="inner", suffixes=("_legacy", "_modern"))
    diagnostics = {
        "legacy_overlap_rows": int(len(legacy_overlap)),
        "modern_overlap_rows": int(len(modern_overlap)),
        "legacy_audit": legacy_audit,
        "modern_season_end_years": sorted(modern_overlap["season_end_year"].unique().astype(int).tolist()),
        "legacy_sample": legacy_overlap.head(12).astype(str).to_dict(orient="records"),
        "modern_sample": modern_overlap.head(12).astype(str).to_dict(orient="records"),
        "same_season_game_id_team_rows": int(len(game_matches)),
        "same_season_game_id_and_home_flag_rows": int(len(home_matches)),
        "same_season_date_home_score_rows": int(len(date_score_matches)),
        "same_game_id_team_id_rows": int(legacy_overlap.merge(modern_overlap, on=game_keys + ["team_id"], how="inner").shape[0]),
        "legacy_team_id_examples": sorted(legacy_overlap["team_id"].astype(str).unique())[:30],
        "modern_team_id_examples": sorted(modern_overlap["team_id"].astype(str).unique())[:30],
    }
    (OUT / "orbit10_source_reconciliation_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    overlap = legacy_overlap.merge(modern_overlap, on=game_keys + ["team_id"], how="inner", suffixes=("_legacy", "_modern"), validate="one_to_one")
    if len(overlap) < 1000:
        raise RuntimeError(f"Insufficient cross-source overlap: {len(overlap)} team-games; diagnostic counts: game_id={len(game_matches)}, game_id+home={len(home_matches)}, date+home+score={len(date_score_matches)}.")
    score_diff = overlap["pts_legacy"] - overlap["pts_modern"]
    poss_diff = overlap["formula_poss_legacy"] - overlap["formula_poss_modern"]
    cross_source_audit = {
        "matched_team_games_2009_10_to_2022_23": int(len(overlap)),
        "score_mismatch_rows": int((score_diff.abs() > 0.01).sum()),
        "mean_abs_possession_difference": round(float(poss_diff.abs().mean()), 5),
        "median_abs_possession_difference": round(float(poss_diff.abs().median()), 5),
        "p95_abs_possession_difference": round(float(poss_diff.abs().quantile(0.95)), 5),
        "possession_correlation": round(float(overlap[["formula_poss_legacy", "formula_poss_modern"]].corr().iloc[0, 1]), 6),
    }
    if cross_source_audit["score_mismatch_rows"] > 0:
        raise RuntimeError("Legacy/provider overlap contains score mismatches; investigate source mapping.")

    # Use the audited repository box-score spine for 2000-01 to 2009-10 and
    # the modern public team-game spine from 2010-11 through 2025-26.
    spine = pd.concat([legacy_games, modern_games], ignore_index=True)
    spine = spine[
        spine["season_end_year"].between(2001, LAST_COMPLETE_SEASON_END_YEAR)
        & spine["game_date"].ge(START_DATE)
    ].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if spine.empty:
        raise RuntimeError("Combined ORBIT-10 game spine is empty.")
    if spine["game_id"].duplicated().any():
        raise RuntimeError("Combined game spine contains duplicate keys.")
    if (spine[["poss_home", "poss_away"]] <= 0).any().any():
        raise RuntimeError("Combined game spine contains non-positive possessions.")

    def split_for_year(y: int) -> str:
        if y <= 2016:
            return "train"
        if y <= 2020:
            return "validation"
        return "holdout"

    spine["split"] = spine["season_end_year"].astype(int).map(split_for_year)
    history: dict[str, dict[str, list[float] | pd.Timestamp]] = {}
    league_off_points: list[float] = []
    league_off_poss: list[float] = []
    league_def_points: list[float] = []
    league_def_poss: list[float] = []
    league_pace_values: list[float] = []
    predictions: list[dict] = []

    for game_date, day in spine.groupby("game_date", sort=True):
        league_off = 100.0 * sum(league_off_points) / sum(league_off_poss) if sum(league_off_poss) else 110.0
        league_def = 100.0 * sum(league_def_points) / sum(league_def_poss) if sum(league_def_poss) else league_off
        league_pace = float(np.mean(league_pace_values)) if league_pace_values else 98.0

        for row in day.to_dict(orient="records"):
            home, away = str(row["home_team_id"]), str(row["away_team_id"])
            hh, ah = history.get(home), history.get(away)
            hn, an = (len(hh["off"]), len(ah["off"])) if hh and ah else (0, 0)
            if min(hn, an) < MIN_PRIOR_GAMES:
                continue

            h_off = shrunk(weighted_recent(hh["off"]), hn, league_off)
            h_def = shrunk(weighted_recent(hh["def"]), hn, league_def)
            h_pace = shrunk(weighted_recent(hh["pace"]), hn, league_pace)
            a_off = shrunk(weighted_recent(ah["off"]), an, league_off)
            a_def = shrunk(weighted_recent(ah["def"]), an, league_def)
            a_pace = shrunk(weighted_recent(ah["pace"]), an, league_pace)
            expected_pace = (h_pace + a_pace) / 2.0
            expected_home_eff = (h_off + a_def) / 2.0
            expected_away_eff = (a_off + h_def) / 2.0
            predicted_total = expected_pace * (expected_home_eff + expected_away_eff) / 100.0

            predictions.append({
                "game_id": row["game_id"],
                "game_date": game_date.date().isoformat(),
                "season_end_year": int(row["season_end_year"]),
                "split": row["split"],
                "home_team_id": home,
                "away_team_id": away,
                "home_prior_games": hn,
                "away_prior_games": an,
                "home_off_eff": h_off,
                "away_off_eff": a_off,
                "home_def_eff": h_def,
                "away_def_eff": a_def,
                "home_pace": h_pace,
                "away_pace": a_pace,
                "expected_possessions": expected_pace,
                "expected_home_efficiency": expected_home_eff,
                "expected_away_efficiency": expected_away_eff,
                "predicted_total": predicted_total,
                "actual_total": float(row["pts_home"] + row["pts_away"]),
                "max_history_date": max(hh["last_date"], ah["last_date"]).date().isoformat(),
            })

        for row in day.to_dict(orient="records"):
            home, away = str(row["home_team_id"]), str(row["away_team_id"])
            hp, ap = float(row["poss_home"]), float(row["poss_away"])
            hpts, apts = float(row["pts_home"]), float(row["pts_away"])
            hrec = history.setdefault(home, {"off": [], "def": [], "pace": [], "last_date": game_date})
            arec = history.setdefault(away, {"off": [], "def": [], "pace": [], "last_date": game_date})
            hrec["off"].append(100.0 * hpts / hp)
            hrec["def"].append(100.0 * apts / ap)
            hrec["pace"].append(float((hp + ap) / 2.0))
            hrec["last_date"] = game_date
            arec["off"].append(100.0 * apts / ap)
            arec["def"].append(100.0 * hpts / hp)
            arec["pace"].append(float((hp + ap) / 2.0))
            arec["last_date"] = game_date
            league_off_points.extend([hpts, apts])
            league_off_poss.extend([hp, ap])
            league_def_points.extend([apts, hpts])
            league_def_poss.extend([ap, hp])
            league_pace_values.append(float((hp + ap) / 2.0))

    pred = pd.DataFrame(predictions)
    if pred.empty:
        raise RuntimeError("No predictions generated; inspect source coverage and season filters.")
    if not (pd.to_datetime(pred["max_history_date"]) < pd.to_datetime(pred["game_date"])).all():
        raise RuntimeError("Temporal leakage check failed: history is not strictly prior to target date.")
    if pred["game_id"].duplicated().any():
        raise RuntimeError("Prediction output contains duplicate game IDs.")

    pred.to_csv(OUT / "orbit10_baseline_predictions.csv", index=False)
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ORBIT-10 Stage 2 — frozen baseline",
        "sources": {
            "legacy": "data/game.csv, regular-season 2000-01 to 2009-10",
            "modern": "public llimllib/nba_data v2 team_game_logs + games, regular-season 2009-10 to 2025-26",
        },
        "legacy_audit": legacy_audit,
        "modern_audit": modern_audit,
        "provider_possession_audit": provider_poss_audit,
        "cross_source_audit": cross_source_audit,
        "combined_spine_games": int(len(spine)),
        "combined_spine_by_season_end_year": {
            str(int(y)): int(n) for y, n in spine.groupby("season_end_year").size().items()
        },
        "possession_formula": "FGA + 0.44*FTA - OREB + TOV",
        "pace_definition": "mean of both teams' formula possession estimates in each game",
        "team_offensive_efficiency": "100 * team points / team formula possessions",
        "team_defensive_efficiency": "100 * opponent points / opponent formula possessions",
        "rolling_window_games": ROLLING_GAMES,
        "recency_half_life_team_games": RECENCY_HALF_LIFE,
        "league_prior_strength_equivalent_games": PRIOR_STRENGTH_GAMES,
        "minimum_prior_games_per_team": MIN_PRIOR_GAMES,
        "prediction_equation": "expected_pace * (expected_home_efficiency + expected_away_efficiency) / 100",
        "split_definitions": {
            "train": "NBA season end-years 2001-2016",
            "validation": "NBA season end-years 2017-2020",
            "holdout": "NBA season end-years 2021-2026",
        },
        "prediction_rows": int(len(pred)),
        "by_split": {name: metrics(part) for name, part in pred.groupby("split", sort=False)},
        "by_season_end_year": {
            str(int(year)): metrics(part)
            for year, part in pred.groupby("season_end_year", sort=True)
        },
        "strict_prior_date_check_passed": True,
        "same_day_games_excluded_from_history": True,
        "market_totals_used_as_features": False,
        "injury_or_pattern_adjustments_used": False,
        "production_or_main_branch_modified": False,
        "interpretation": "Baseline benchmark only; no pattern features, betting profitability claim, or deployment.",
    }
    (OUT / "orbit10_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
