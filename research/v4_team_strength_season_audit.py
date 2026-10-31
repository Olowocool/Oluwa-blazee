#!/usr/bin/env python3
"""Research-only NBA team-form and season-strength audit.

Uses NBA Stats team game logs from the public llimllib/nba_data v2 mirror.
Season-end team profiles are descriptive. Forecast features and factor tests
are built point-in-time: all same-date games share only strictly earlier games.
No production files, models, or betting ledgers are changed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import sys

import duckdb
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.v4_phase1_core import (
    BASE,
    FIRST_SEASON,
    LAST_SEASON,
    TEST_SEASONS,
    build_point_in_time,
    fit_rating_model,
    metrics,
)

OUT = Path("outputs/research/v4_team_strength_season_audit")
PROFILE_REQUIRED = {
    "season", "game_id", "game_date", "team_id", "team_name",
    "opp_team_id", "pts", "fga", "fta", "oreb", "tov", "fgm", "fg3m", "fg3a",
}


def load_raw_regular_season_logs() -> tuple[pd.DataFrame, dict]:
    """Load completed regular seasons 2009-10 through 2025-26."""
    catalog_url = f"{BASE}/nba/metadata.json"
    with urlopen(catalog_url, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))["datasets"]

    first_season = max(2010, int(metadata["team_game_logs"]["first_season"]))
    last_season = min(2026, int(metadata["team_game_logs"]["last_season"]))
    if first_season > last_season:
        raise RuntimeError("The public source does not contain completed NBA seasons.")

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    for dataset in ("team_game_logs", "games", "team_seasons"):
        info = metadata[dataset]
        source_first = int(info["first_season"])
        source_last = int(info["last_season"])
        if int(info["seasons"]) != source_last - source_first + 1:
            raise RuntimeError(f"{dataset} metadata reports non-contiguous seasons.")
        if first_season < source_first or last_season > source_last:
            raise RuntimeError(
                f"{dataset} lacks requested season range {first_season}-{last_season}."
            )
        urls = [
            f"{BASE}/nba/stats/{dataset}/season={s}/data.parquet"
            for s in range(first_season, last_season + 1)
        ]
        url_list = ", ".join("'" + url + "'" for url in urls)
        con.execute(
            f"CREATE VIEW {dataset} AS SELECT * FROM read_parquet([{url_list}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )

    raw = con.execute("""
        SELECT
            t.season, t.game_id, t.game_date, t.team_id,
            ts.full_name AS team_name, t.opp_team_id,
            ots.full_name AS opponent_name,
            t.pts, t.poss AS source_poss, t.pace AS source_pace,
            t.fgm, t.fg3m, t.fga, t.fg3a, t.fta, t.oreb, t.tov,
            t.home, g.neutral_site, g.game_type
        FROM team_game_logs t
        JOIN games g USING (season, game_id)
        LEFT JOIN team_seasons ts
          ON ts.season = t.season AND ts.team_id = t.team_id
        LEFT JOIN team_seasons ots
          ON ots.season = t.season AND ots.team_id = t.opp_team_id
        WHERE t.season BETWEEN ? AND ?
          AND g.game_type = 'regular_season'
        ORDER BY t.season, t.game_date, t.game_id, t.team_id
    """, [first_season, last_season]).fetchdf()
    con.close()

    metadata_out = {
        "catalog_url": catalog_url,
        "first_season_end_year": first_season,
        "last_season_end_year": last_season,
        "seasons_loaded": last_season - first_season + 1,
        "season_convention": "season year is the ending year; 2026 means 2025-26",
        "datasets": {
            name: {
                "first_season": int(metadata[name]["first_season"]),
                "last_season": int(metadata[name]["last_season"]),
                "updated": metadata[name].get("updated"),
            }
            for name in ("team_game_logs", "games", "team_seasons")
        },
    }
    return raw, metadata_out


def prepare_team_game_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate possession-based team and opponent outcomes with strict pairing."""
    missing = sorted(PROFILE_REQUIRED - set(raw.columns))
    if missing:
        raise ValueError(f"NBA source is missing required fields: {missing}")
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    for col in ("pts", "fga", "fta", "oreb", "tov", "fgm", "fg3m", "fg3a",
                "source_poss", "source_pace"):
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["team_id"] = d["team_id"].astype(str)
    d["opp_team_id"] = d["opp_team_id"].astype(str)
    d["team_name"] = d["team_name"].fillna("").astype(str).str.strip()
    d = d.dropna(subset=[
        "season", "game_id", "game_date", "pts", "fga", "fta", "oreb", "tov",
        "fgm", "fg3m", "fg3a",
    ]).copy()
    if d["team_name"].eq("").any():
        raise ValueError("Some NBA team-game rows have no season-specific team name.")
    if d.duplicated(["season", "game_id", "team_id"]).any():
        raise ValueError("Duplicate season/game/team rows in the source.")
    d["formula_poss"] = d["fga"] + 0.44 * d["fta"] - d["oreb"] + d["tov"]
    d = d.loc[d["formula_poss"] > 0].copy()
    d["off_eff"] = 100.0 * d["pts"] / d["formula_poss"]
    d["home_flag"] = d["home"].fillna(False).astype(bool).astype(int)

    opponent = d[[
        "season", "game_id", "team_id", "team_name", "pts", "formula_poss"
    ]].rename(columns={
        "team_id": "paired_team_id",
        "team_name": "opponent_name_check",
        "pts": "opp_pts",
        "formula_poss": "opp_formula_poss",
    })
    paired = d.merge(
        opponent,
        left_on=["season", "game_id", "opp_team_id"],
        right_on=["season", "game_id", "paired_team_id"],
        how="left",
        validate="one_to_one",
    )
    if paired["opp_pts"].isna().any() or paired["opp_formula_poss"].isna().any():
        raise ValueError("Some team-game rows do not have exactly one opponent row.")
    if (paired["opp_pts"] <= 0).any() or (paired["opp_formula_poss"] <= 0).any():
        raise ValueError("Invalid opponent points or possession denominator.")
    if (paired["opponent_name"].astype(str) != paired["opponent_name_check"].astype(str)).any():
        raise ValueError("Opponent IDs and season-specific names do not reconcile.")
    paired["def_eff_allowed"] = 100.0 * paired["opp_pts"] / paired["opp_formula_poss"]
    paired["net_eff"] = paired["off_eff"] - paired["def_eff_allowed"]
    paired["game_total"] = paired["pts"] + paired["opp_pts"]
    paired["margin"] = paired["pts"] - paired["opp_pts"]
    paired["win"] = paired["margin"] > 0
    return paired.sort_values(["season", "team_id", "game_date", "game_id"]).reset_index(drop=True)


def _aggregate_efficiency(frame: pd.DataFrame) -> tuple[float, float]:
    poss = float(frame["formula_poss"].sum())
    opp_poss = float(frame["opp_formula_poss"].sum())
    off = 100.0 * float(frame["pts"].sum()) / poss if poss else np.nan
    defense_allowed = 100.0 * float(frame["opp_pts"].sum()) / opp_poss if opp_poss else np.nan
    return off, defense_allowed


def build_team_season_profiles(paired: pd.DataFrame) -> pd.DataFrame:
    """Create season summaries plus last-10 form and end-season opponent-adjusted ratings."""
    rating_rows = []
    for season, group in paired.groupby("season", sort=True):
        rating = fit_rating_model(group)
        if rating is None:
            continue
        model, index, n = rating["model"], rating["index"], rating["n"]
        for team_id, idx in index.items():
            rating_rows.append({
                "season": int(season),
                "team_id": str(team_id),
                "schedule_adj_offense_effect_per100": float(model.coef_[idx]),
                # Positive opponent effect means opponents scored more efficiently
                # against this team, so a larger value indicates weaker defense.
                "opponent_scoring_allowance_effect_per100": float(model.coef_[n + idx]),
                "schedule_adj_net_strength_effect_per100": float(
                    model.coef_[idx] - model.coef_[n + idx]
                ),
            })
    ratings = pd.DataFrame(rating_rows)

    records = []
    for (season, team_id, team_name), group in paired.groupby(
        ["season", "team_id", "team_name"], sort=True
    ):
        group = group.sort_values(["game_date", "game_id"])
        recent = group.tail(10)
        off, defense = _aggregate_efficiency(group)
        recent_off, recent_defense = _aggregate_efficiency(recent)
        fga = float(group["fga"].sum())
        records.append({
            "season": int(season),
            "team_id": str(team_id),
            "team_name": str(team_name),
            "games": int(group["game_id"].nunique()),
            "wins": int(group["win"].sum()),
            "losses": int(group["game_id"].nunique() - group["win"].sum()),
            "win_pct": float(group["win"].mean()),
            "points_for_per_game": float(group["pts"].mean()),
            "points_allowed_per_game": float(group["opp_pts"].mean()),
            "point_margin_per_game": float(group["margin"].mean()),
            "true_possessions_per_game": float(group["formula_poss"].mean()),
            "nba_source_pace_per48": float(group["source_pace"].mean()),
            "off_rating_per100": off,
            "def_rating_allowed_per100": defense,
            "net_rating_per100": off - defense,
            "effective_fg_pct": 100.0 * float(
                (group["fgm"].sum() + 0.5 * group["fg3m"].sum()) / fga
            ) if fga else np.nan,
            "three_point_attempt_rate_pct": 100.0 * float(group["fg3a"].sum() / fga) if fga else np.nan,
            "free_throw_attempt_rate_pct": 100.0 * float(group["fta"].sum() / fga) if fga else np.nan,
            "turnovers_per100_possessions": 100.0 * float(group["tov"].sum() / group["formula_poss"].sum()),
            "last10_games": int(recent["game_id"].nunique()),
            "last10_points_for_per_game": float(recent["pts"].mean()),
            "last10_points_allowed_per_game": float(recent["opp_pts"].mean()),
            "last10_true_possessions_per_game": float(recent["formula_poss"].mean()),
            "last10_off_rating_per100": recent_off,
            "last10_def_rating_allowed_per100": recent_defense,
            "last10_net_rating_per100": recent_off - recent_defense,
            "last10_off_rating_change_vs_season": recent_off - off,
            "last10_def_rating_change_vs_season": recent_defense - defense,
            "last10_net_rating_change_vs_season": (recent_off - recent_defense) - (off - defense),
        })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No team-season profiles were produced.")
    rating_columns = [
        "schedule_adj_offense_effect_per100",
        "opponent_scoring_allowance_effect_per100",
        "schedule_adj_net_strength_effect_per100",
    ]
    if not ratings.empty:
        result = result.merge(ratings, on=["season", "team_id"], how="left", validate="one_to_one")
    else:
        for column in rating_columns:
            result[column] = np.nan

    for season, idx in result.groupby("season").groups.items():
        part = result.loc[idx]
        result.loc[idx, "offense_rank"] = part["off_rating_per100"].rank(ascending=False, method="min")
        result.loc[idx, "defense_rank"] = part["def_rating_allowed_per100"].rank(ascending=True, method="min")
        result.loc[idx, "net_rating_rank"] = part["net_rating_per100"].rank(ascending=False, method="min")
        result.loc[idx, "pace_rank"] = part["true_possessions_per_game"].rank(ascending=False, method="min")
        result.loc[idx, "schedule_adjusted_strength_rank"] = part[
            "schedule_adj_net_strength_effect_per100"
        ].rank(ascending=False, method="min")
        for col, source in (
            ("off_rating_vs_league", "off_rating_per100"),
            ("defense_strength_vs_league", "def_rating_allowed_per100"),
            ("net_rating_vs_league", "net_rating_per100"),
        ):
            league_avg = float(part[source].mean())
            if col == "defense_strength_vs_league":
                result.loc[idx, col] = league_avg - part[source]
            else:
                result.loc[idx, col] = part[source] - league_avg
    return result.sort_values(["season", "net_rating_rank", "team_name"]).reset_index(drop=True)


def build_league_season_environment(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize scoring and pace context so PPG is not compared across eras raw."""
    game_rows = paired.drop_duplicates(["season", "game_id"])
    records = []
    for season, group in paired.groupby("season", sort=True):
        games = game_rows.loc[game_rows["season"] == season]
        off, defense = _aggregate_efficiency(group)
        records.append({
            "season": int(season),
            "games": int(games["game_id"].nunique()),
            "average_game_total": float(games["game_total"].mean()),
            "median_game_total": float(games["game_total"].median()),
            "average_team_points": float(group["pts"].mean()),
            "average_true_possessions": float(group["formula_poss"].mean()),
            "average_source_pace_per48": float(group["source_pace"].mean()),
            "league_off_rating_per100": off,
            "league_def_rating_allowed_per100": defense,
            "league_net_rating_per100": off - defense,
            "average_effective_fg_pct": 100.0 * float(
                (group["fgm"].sum() + 0.5 * group["fg3m"].sum()) / group["fga"].sum()
            ),
            "average_three_point_attempt_rate_pct": 100.0 * float(group["fg3a"].sum() / group["fga"].sum()),
            "average_free_throw_attempt_rate_pct": 100.0 * float(group["fta"].sum() / group["fga"].sum()),
            "average_turnovers_per100_possessions": 100.0 * float(group["tov"].sum() / group["formula_poss"].sum()),
        })
    return pd.DataFrame(records).sort_values("season").reset_index(drop=True)


def build_factor_ablation(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pace-only, efficiency-only, and combined point-in-time forecasts."""
    predictions = build_point_in_time(raw)
    predictions = predictions.loc[predictions["season"].isin(TEST_SEASONS)].copy()
    if predictions.empty:
        raise RuntimeError("No completed test-season predictions were generated.")
    predictions["pace_only_total"] = (
        predictions["expected_pace"] * 2.0 * predictions["league_ortg"] / 100.0
    )
    predictions["efficiency_only_total"] = (
        predictions["league_pace"] *
        (predictions["expected_home_efficiency"] + predictions["expected_away_efficiency"]) / 100.0
    )
    predictions["league_environment_total"] = (
        2.0 * predictions["league_pace"] * predictions["league_ortg"] / 100.0
    )
    predictors = {
        "league_environment_only": "league_environment_total",
        "team_form_efficiency_only": "efficiency_only_total",
        "matchup_pace_only": "pace_only_total",
        "pace_plus_opponent_adjusted_efficiency": "formula_total",
    }
    rows = []
    for season, part in predictions.groupby("season", sort=True):
        for label, column in predictors.items():
            rows.append({
                "season": int(season),
                "predictor": label,
                **metrics(part, column),
            })
    for label, column in predictors.items():
        rows.append({
            "season": "ALL_TEST_SEASONS",
            "predictor": label,
            **metrics(predictions, column),
        })
    return predictions, pd.DataFrame(rows)


def build_report(team: pd.DataFrame, league: pd.DataFrame, ablation: pd.DataFrame,
                 metadata: dict, prediction_rows: pd.DataFrame) -> str:
    all_metrics = ablation.loc[ablation["season"] == "ALL_TEST_SEASONS"].copy()
    latest = int(league["season"].max())
    latest_teams = team.loc[team["season"] == latest].sort_values(
        "schedule_adj_net_strength_effect_per100", ascending=False
    )
    lines = [
        "# NBA team form, strength and totals audit",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope and data coverage",
        "",
        f"- Source: NBA Stats team game logs mirrored by llimllib/nba_data v2 ({metadata['catalog_url']}).",
        f"- Completed seasons covered: {metadata['first_season_end_year']}–{metadata['last_season_end_year']} "
        f"({metadata['seasons_loaded']} seasons; end-year convention).",
        "- Team-game data and exact row counts are recorded in the machine-readable summary.",
        "- Current partial 2026-27 season is excluded from season-end rankings.",
        "- Preseason, All-Star, play-in, playoff and NBA Cup Final games are excluded.",
        "",
        "## Why season normalization matters",
        "",
        "Points per game alone are not a stable measure of team strength across seasons. This audit reports points per game alongside true possessions, offensive rating, defensive points allowed per 100 possessions, net rating, shooting efficiency and turnover rate. The team-season table also reports each team's last ten games relative to its season average.",
        "",
        "## League scoring and pace environment",
        "",
        league.to_markdown(index=False),
        "",
        "## Point-in-time totals factor comparison",
        "",
        all_metrics[["predictor", "games", "mae", "rmse", "bias_prediction_minus_actual", "within_5_points_pct", "within_10_points_pct"]].to_markdown(index=False),
        "",
        "The four forecasts are deliberately separated: a league-environment baseline, team-form efficiency at league pace, matchup pace at league efficiency, and the combined pace-plus-opponent-adjusted-efficiency forecast. Each prediction uses only completed games strictly before the target date; same-date games cannot affect one another.",
        "",
        "## Latest completed season — strongest schedule-adjusted team profiles",
        "",
        latest_teams[[
            "team_name", "games", "win_pct", "off_rating_per100",
            "def_rating_allowed_per100", "net_rating_per100",
            "true_possessions_per_game", "last10_net_rating_change_vs_season",
            "schedule_adj_net_strength_effect_per100",
        ]].head(10).to_markdown(index=False),
        "",
        "## Interpretation safeguards",
        "",
        "- Season-end team profiles are retrospective descriptions, not pregame inputs.",
        "- Schedule-adjusted team coefficients are ridge effects estimated from each team's final 25 regular-season games. Positive offense effect is better; higher opponent-scoring allowance is worse for defense; their difference is a relative strength indicator.",
        "- The model-comparison forecasts are point-in-time and date-grouped. The current season's partial data is not treated as a completed-season ranking.",
        "- NBA Stats may revise old game logs; date cutoffs prevent future games from entering a forecast but cannot reconstruct the exact historical stat vintage available to bettors on that date.",
        "- This is a score-prediction and team-strength audit, not an ROI or profitability test. No production model or app is modified.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, metadata = load_raw_regular_season_logs()
    paired = prepare_team_game_data(raw)
    team_profiles = build_team_season_profiles(paired)
    league_environment = build_league_season_environment(paired)
    predictions, ablation = build_factor_ablation(raw)

    team_profiles.to_csv(OUT / "team_season_strength.csv", index=False)
    league_environment.to_csv(OUT / "league_season_environment.csv", index=False)
    ablation.to_csv(OUT / "totals_factor_ablation.csv", index=False)
    predictions.to_csv(OUT / "point_in_time_matchup_features.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": metadata,
        "raw_team_game_rows": int(len(raw)),
        "paired_team_game_rows": int(len(paired)),
        "distinct_regular_season_games": int(paired[["season", "game_id"]].drop_duplicates().shape[0]),
        "seasons_in_team_profiles": sorted(int(x) for x in team_profiles["season"].unique()),
        "team_season_profiles": int(len(team_profiles)),
        "team_count_by_season": {
            str(int(season)): int(part["team_id"].nunique())
            for season, part in team_profiles.groupby("season")
        },
        "point_in_time_test_games": int(len(predictions)),
        "test_seasons": sorted(int(x) for x in predictions["season"].unique()),
        "same_day_games_excluded": True,
        "all_feature_history_dates_strictly_prior": bool(
            (pd.to_datetime(predictions["max_history_date"]) < pd.to_datetime(predictions["game_date"])).all()
        ),
        "production_modified": False,
        "output_files": [
            "team_season_strength.csv",
            "league_season_environment.csv",
            "totals_factor_ablation.csv",
            "point_in_time_matchup_features.csv",
            "team_strength_audit_summary.json",
            "team_strength_audit_report.md",
        ],
    }
    (OUT / "team_strength_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    report = build_report(team_profiles, league_environment, ablation, metadata, predictions)
    (OUT / "team_strength_audit_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nAudit summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
