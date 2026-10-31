#!/usr/bin/env python3
"""Research-only T-120 market evaluation of Points Engine V2 and the V2/V4 blend.

The V2/V4 blend coefficient is frozen from the 2020-2023 development period:
    blend = V2 + 0.584068 * (V4 - V2)

The evaluation uses the latest locally observed DraftKings snapshot no later
than T-120, with a maximum observation age of 240 minutes. Settlement always
uses that entry line and entry prices. Later pregame observations are used only
for CLV diagnostics. Current-season V4 features are reconstructed after the
game using only completed-game history strictly before its calendar date; this
is a retrospective point-in-time reconstruction, not a forecast frozen before
tip. No production code, live model, or betting ledger is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team_name_utils import normalize_team_name
from research.v4_phase1_core import (
    BASE,
    FIRST_SEASON,
    LAST_SEASON,
    build_point_in_time,
    connect_and_load,
)
from research.v4_phase3_market_residual import (
    DEFAULT_SCORE_PATCH_DIR,
    DEFAULT_SCORES,
    DEFAULT_SNAPSHOTS,
    BOOKMAKER,
    ENTRY_MINUTES,
    MAX_ENTRY_AGE_MINUTES,
    add_bets,
    build_game_dataset,
    discover_score_patch_files,
    prepare_snapshots,
    select_entry_and_last,
)

DEFAULT_OUTPUT = "outputs/research/v4_phase3_blend_market"
DEFAULT_TARGET_SEASON = 2027
DEFAULT_BETA = 0.584068
DEFAULT_BETA_SOURCE = (
    "research/v4_v2_residual_blend_results_2026-10-31.md; "
    "fit on 2020-2023 development games, frozen before 2024-2026 holdout"
)


def _bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_team_name_map(season: int) -> dict[str, str]:
    """Read franchise IDs/names for one NBA season; no hard-coded team count."""
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as response:
        datasets = json.loads(response.read().decode("utf-8"))["datasets"]
    info = datasets["team_seasons"]
    first, last = int(info["first_season"]), int(info["last_season"])
    if not first <= int(season) <= last:
        raise ValueError(
            f"Team-name source covers seasons {first}-{last}, not {season}."
        )
    url = f"{BASE}/nba/stats/team_seasons/season={int(season)}/data.parquet"
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        rows = con.execute(
            f"SELECT team_id, full_name FROM read_parquet(['{url}'], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, "
            "union_by_name=true) WHERE season = ? AND is_nba = TRUE",
            [int(season)],
        ).fetchdf()
    finally:
        con.close()
    if rows.empty:
        raise ValueError(f"No NBA team-name records found for season {season}.")
    rows["team_id"] = rows["team_id"].astype(str)
    rows["full_name"] = rows["full_name"].astype(str).str.strip()
    if rows["team_id"].duplicated().any() or rows["full_name"].eq("").any():
        raise ValueError(f"Team-name mapping for season {season} is ambiguous.")
    return dict(zip(rows["team_id"], rows["full_name"]))


def build_current_v4_predictions(
    target_season: int = DEFAULT_TARGET_SEASON,
    min_team_games: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build current-season V4 rows with strict pre-target-date history."""
    if int(target_season) <= FIRST_SEASON:
        raise ValueError("target_season must have a prior season available.")
    raw, metadata = connect_and_load(last_season=int(target_season))
    if raw.empty:
        raise ValueError("NBA advanced-stat source returned no regular-season rows.")
    available = sorted(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int).unique())
    if int(target_season) not in available:
        raise ValueError(
            f"Advanced-stat source does not yet contain target season {target_season}; "
            f"available seasons end at {max(available) if available else 'none'}."
        )

    names = load_team_name_map(int(target_season))
    all_features = build_point_in_time(
        raw,
        last_season=int(target_season),
        min_team_games=int(min_team_games),
        prediction_seasons=[int(target_season)],
    )
    current = all_features.loc[
        pd.to_numeric(all_features["season"], errors="coerce").eq(int(target_season))
    ].copy()

    neutral = (
        raw.loc[pd.to_numeric(raw["season"], errors="coerce").eq(int(target_season))]
        .groupby("game_id", sort=False)["neutral_site"]
        .first()
        .map(_bool_value)
    )
    # NBA Stats game_id values may be numeric in the raw table while the
    # point-in-time builder serializes them as strings. Normalize both sides
    # before mapping; otherwise every neutral-site flag would silently become
    # missing and be treated as non-neutral.
    neutral.index = neutral.index.astype(str)
    current["is_neutral_site"] = current["game_id"].astype(str).map(neutral).fillna(False)
    current = current.loc[~current["is_neutral_site"]].copy()

    current["home_team"] = current["home_team_id"].astype(str).map(names)
    current["away_team"] = current["away_team_id"].astype(str).map(names)
    if current[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Current V4 features contain unmapped team IDs.")
    current["game_date"] = pd.to_datetime(current["game_date"], errors="coerce").dt.normalize()
    current["max_history_date"] = pd.to_datetime(
        current["max_history_date"], errors="coerce"
    ).dt.normalize()
    if current[["game_date", "max_history_date"]].isna().any().any():
        raise ValueError("Current V4 features contain invalid dates.")
    if not (current["max_history_date"] < current["game_date"]).all():
        raise AssertionError("V4 current-season feature history includes target/future dates.")

    current["home_team_key"] = current["home_team"].map(normalize_team_name)
    current["away_team_key"] = current["away_team"].map(normalize_team_name)
    keys = ["game_date", "home_team_key", "away_team_key"]
    if current.duplicated(keys).any():
        raise ValueError("Current V4 features contain duplicate date/home/away game keys.")

    source_latest = pd.to_datetime(raw["game_date"], errors="coerce").max()
    diag = {
        "advanced_stat_source": "llimllib/nba_data v2 NBA Stats team_game_logs",
        "advanced_source_available_seasons": available,
        "target_season": int(target_season),
        "target_season_raw_team_game_rows": int(
            pd.to_numeric(raw["season"], errors="coerce").eq(int(target_season)).sum()
        ),
        "target_season_source_latest_game_date": (
            pd.Timestamp(source_latest).date().isoformat() if pd.notna(source_latest) else None
        ),
        "target_season_v4_feature_rows": int(len(current)),
        "minimum_prior_current_season_games_per_team": int(min_team_games),
        "neutral_target_games_excluded": int(
            pd.to_numeric(all_features.loc[
                pd.to_numeric(all_features["season"], errors="coerce").eq(int(target_season)),
                "season"
            ], errors="coerce").shape[0] - len(current)
        ),
        "team_ids_mapped": int(len(names)),
        "all_v4_history_strictly_prior": True,
        "historical_stat_vintages_reconstructed_not_archived": True,
        "metadata_team_game_logs_last_season": int(
            metadata["datasets"]["team_game_logs"]["last_season"]
        ),
    }
    return current, diag


def attach_v4_features(
    v2_games: pd.DataFrame,
    v4_features: pd.DataFrame,
    beta: float = DEFAULT_BETA,
) -> pd.DataFrame:
    """One-to-one date/team join; reject score mismatches and non-prior features."""
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("Blend beta must be between 0 and 1.")
    left = v2_games.copy()
    right = v4_features.copy()
    required_left = {"game_date", "home_team", "away_team", "actual_total", "points_engine_total"}
    required_right = {
        "game_date", "home_team", "away_team", "actual_total",
        "projected_total", "max_history_date",
    }
    if missing := sorted(required_left - set(left.columns)):
        raise ValueError(f"V2/market rows are missing columns: {missing}")
    if missing := sorted(required_right - set(right.columns)):
        raise ValueError(f"V4 rows are missing columns: {missing}")

    for frame in (left, right):
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
        frame["home_team_key"] = frame["home_team"].map(normalize_team_name)
        frame["away_team_key"] = frame["away_team"].map(normalize_team_name)
    right["max_history_date"] = pd.to_datetime(
        right["max_history_date"], errors="coerce"
    ).dt.normalize()
    if right[["game_date", "max_history_date"]].isna().any().any():
        raise ValueError("V4 rows contain invalid dates.")
    if not (right["max_history_date"] < right["game_date"]).all():
        raise AssertionError("V4 target-date/future history detected.")
    keys = ["game_date", "home_team_key", "away_team_key"]
    if left.duplicated(keys).any():
        raise ValueError("V2/market rows are not unique by date/home/away.")
    if right.duplicated(keys).any():
        raise ValueError("V4 rows are not unique by date/home/away.")

    right = right[keys + [
        "projected_total", "actual_total", "max_history_date",
        "home_games_prior", "away_games_prior", "season_blend_weight",
    ]].rename(columns={
        "projected_total": "v4_pace_efficiency_total",
        "actual_total": "v4_source_actual_total",
        "max_history_date": "v4_max_history_date",
        "home_games_prior": "v4_home_games_prior",
        "away_games_prior": "v4_away_games_prior",
        "season_blend_weight": "v4_season_blend_weight",
    })
    merged = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    merged["v4_feature_matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")
    merged["v4_score_consistent"] = (
        merged["v4_feature_matched"]
        & np.isclose(
            pd.to_numeric(merged["actual_total"], errors="coerce"),
            pd.to_numeric(merged["v4_source_actual_total"], errors="coerce"),
            atol=0.01,
        )
    )
    invalid = merged["v4_feature_matched"] & ~merged["v4_score_consistent"]
    merged["v4_prediction_valid"] = merged["v4_feature_matched"] & merged["v4_score_consistent"]
    if merged.loc[merged["v4_prediction_valid"], "v4_max_history_date"].ge(
        merged.loc[merged["v4_prediction_valid"], "game_date"]
    ).any():
        raise AssertionError("Joined V4 feature is not strictly prior to its target date.")

    merged["blend_beta"] = float(beta)
    merged["blend_total"] = np.where(
        merged["v4_prediction_valid"],
        merged["points_engine_total"] + float(beta) * (
            merged["v4_pace_efficiency_total"] - merged["points_engine_total"]
        ),
        np.nan,
    )
    merged.attrs["v4_score_mismatch_rows"] = int(invalid.sum())
    return merged


def _accuracy_rows(frame: pd.DataFrame, columns: list[tuple[str, str]], split: str) -> pd.DataFrame:
    rows = []
    for label, column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        actual = pd.to_numeric(frame["actual_total"], errors="coerce")
        valid = values.notna() & actual.notna()
        error = values[valid].to_numpy(dtype=float) - actual[valid].to_numpy(dtype=float)
        if not len(error):
            rows.append({
                "population": split, "predictor": label, "games": 0,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "within_5_points_pct": np.nan, "within_10_points_pct": np.nan,
            })
            continue
        absolute = np.abs(error)
        rows.append({
            "population": split, "predictor": label, "games": int(len(error)),
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(error.mean()),
            "within_5_points_pct": float(np.mean(absolute <= 5) * 100.0),
            "within_10_points_pct": float(np.mean(absolute <= 10) * 100.0),
        })
    return pd.DataFrame(rows)


def summarize_bets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, prefix in (
        ("Points Engine V2", "points_engine"),
        ("V4", "v4"),
        ("V2 + V4 blend", "blend"),
    ):
        side_col = f"{prefix}_side"
        settlement_col = f"{prefix}_settlement"
        profit_col = f"{prefix}_profit_units"
        bets = frame.loc[frame[side_col].isin(["Over", "Under"])].copy()
        wins = int(bets[settlement_col].eq("Win").sum())
        losses = int(bets[settlement_col].eq("Loss").sum())
        pushes = int(bets[settlement_col].eq("Push").sum())
        profits = pd.to_numeric(bets[profit_col], errors="coerce").fillna(0.0)
        clv = []
        probability_clv = []
        for _, bet in bets.iterrows():
            last_line = pd.to_numeric(pd.Series([bet.get("last_observed_total")]), errors="coerce").iloc[0]
            side = bet[side_col]
            if pd.notna(last_line):
                clv.append(
                    float(last_line - bet["entry_total"])
                    if side == "Over" else float(bet["entry_total"] - last_line)
                )
            last_p = bet.get(
                "last_observed_over_probability_no_vig"
                if side == "Over" else "last_observed_under_probability_no_vig"
            )
            entry_p = bet.get(
                "over_probability_no_vig"
                if side == "Over" else "under_probability_no_vig"
            )
            if pd.notna(last_p) and pd.notna(entry_p):
                probability_clv.append(float(last_p) - float(entry_p))
        n = int(len(bets))
        rows.append({
            "strategy": label,
            "evaluated_games": int(len(frame)),
            "bets": n,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate_pct_ex_pushes": (
                100.0 * wins / (wins + losses) if wins + losses else np.nan
            ),
            "net_profit_units": float(profits.sum()),
            "roi_pct_per_unit_staked": (
                100.0 * float(profits.sum()) / n if n else np.nan
            ),
            "clv_bets_with_later_snapshot": int(len(clv)),
            "mean_side_adjusted_line_clv_points": float(np.mean(clv)) if clv else np.nan,
            "positive_side_adjusted_line_clv_pct": (
                100.0 * float(np.mean(np.asarray(clv) > 0)) if clv else np.nan
            ),
            "mean_side_adjusted_no_vig_probability_clv": (
                float(np.mean(probability_clv)) if probability_clv else np.nan
            ),
        })
    return pd.DataFrame(rows)


def build_report(
    metrics: pd.DataFrame,
    bets: pd.DataFrame,
    diagnostics: dict[str, Any],
    beta: float,
) -> str:
    lines = [
        "# V2 + V4 blend versus timestamped DraftKings entry market",
        "",
        "## Protocol",
        "",
        f"- Entry market: latest locally observed DraftKings row at or before T-{diagnostics['entry_minutes']} minutes.",
        f"- Maximum entry observation age: {diagnostics['max_entry_age_minutes']} minutes.",
        "- Settlement uses the selected entry total and the selected entry Over/Under prices only.",
        "- Later pregame market observations are used only for side-adjusted CLV diagnostics.",
        "- Bet directions and thresholds use the unchanged production classifier (Lean at 2.5 points; Strong at 5 points).",
        f"- Frozen blend coefficient: beta = {beta:.6f}; source: {DEFAULT_BETA_SOURCE}.",
        "- Current-season V4 feature reconstruction uses NBA Stats data strictly before each target game date and requires at least five completed current-season games from each team.",
        "",
        "## Data coverage",
        "",
        f"- Raw odds observations: {diagnostics['input_odds_rows']:,}.",
        f"- DraftKings pregame observations after validation: {diagnostics['draftkings_pregame_rows']:,}.",
        f"- Games with a valid T-{diagnostics['entry_minutes']} entry: {diagnostics['games_with_entry_observation']:,}.",
        f"- Entry games matched to final scores and V2 forecasts: {diagnostics['games_with_score_and_v2']:,}.",
        f"- V4 feature rows available for current season {diagnostics['target_season']}: {diagnostics['target_season_v4_feature_rows']:,}.",
        f"- Settled entry games with a score-consistent V4 forecast: {diagnostics['shared_v2_v4_games']:,}.",
        f"- V4 feature misses: {diagnostics['v4_feature_missing_games']:,}; score mismatches excluded: {diagnostics['v4_score_mismatch_games']:,}.",
        "",
        "## Accuracy on common V2/V4 games",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f") if not metrics.empty else "_No settled games have a valid V4 forecast; no same-sample accuracy comparison is available._",
        "",
        "## Entry-price betting simulation on the same games",
        "",
        bets.to_markdown(index=False, floatfmt=".4f") if not bets.empty else "_No settled games have a valid V4 forecast; no blend betting simulation is available._",
        "",
        "## Important limitations",
        "",
        "- This evaluation reconstructs pregame features after game completion. The feature cutoff is strictly prior to the target game date, but these predictions were not frozen at the historical entry time. Treat results as a retrospective point-in-time reconstruction, not proof of an executable live strategy.",
        "- Historical NBA Stats values may include retrospective corrections; archived stat vintages are not available.",
        "- This sample is not sufficient to establish profitability. Report ROI, win rate and CLV as descriptive only; do not tune thresholds on these results.",
        "- The separate Ridge market-residual learner remains in warm-up until it has at least 150 earlier eligible games on strictly prior game dates.",
        "- A point-total forecast is not a calibrated Over/Under probability. No expected-value claim is made.",
        "- No raw bookmaker prices are committed to Git, and no production code, deployment, live prediction or betting ledger is changed.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_odds = pd.read_csv(args.snapshots_file)
    snapshots, snapshot_diag = prepare_snapshots(raw_odds, args.bookmaker)
    selected, select_diag = select_entry_and_last(
        snapshots, args.entry_minutes, args.max_entry_age_minutes
    )
    if selected.empty:
        raise ValueError("No games have an eligible fixed-cutoff entry snapshot.")

    patch_files = discover_score_patch_files(
        patch_dir=args.score_patch_dir, explicit_paths=args.score_patch_files
    )
    v2_games, score_diag = build_game_dataset(
        selected, args.scores_file, [str(path) for path in patch_files]
    )
    v4_features, v4_diag = build_current_v4_predictions(
        target_season=args.target_season, min_team_games=args.min_team_games
    )
    joined = attach_v4_features(v2_games, v4_features, beta=args.beta)
    valid = joined.loc[joined["v4_prediction_valid"]].copy()

    metrics = _accuracy_rows(
        valid,
        [
            ("DraftKings T-120 entry total", "entry_total"),
            ("Points Engine V2", "points_engine_total"),
            ("V4 pace + efficiency", "v4_pace_efficiency_total"),
            ("V2 + V4 blend", "blend_total"),
        ],
        "shared_v2_v4_games",
    ) if not valid.empty else pd.DataFrame(columns=[
        "population", "predictor", "games", "mae", "rmse", "bias",
        "within_5_points_pct", "within_10_points_pct",
    ])

    if not valid.empty:
        betting = add_bets(valid, "points_engine_total", "points_engine")
        betting = add_bets(betting, "v4_pace_efficiency_total", "v4")
        betting = add_bets(betting, "blend_total", "blend")
        bet_summary = summarize_bets(betting)
    else:
        betting = valid.copy()
        bet_summary = pd.DataFrame(columns=[
            "strategy", "evaluated_games", "bets", "wins", "losses", "pushes",
            "win_rate_pct_ex_pushes", "net_profit_units", "roi_pct_per_unit_staked",
            "clv_bets_with_later_snapshot", "mean_side_adjusted_line_clv_points",
            "positive_side_adjusted_line_clv_pct",
            "mean_side_adjusted_no_vig_probability_clv",
        ])

    diagnostics = {
        "status": "success",
        "api_credits_used": 0,
        "bookmaker": args.bookmaker,
        "entry_minutes": int(args.entry_minutes),
        "max_entry_age_minutes": int(args.max_entry_age_minutes),
        "input_odds_rows": int(len(raw_odds)),
        "draftkings_pregame_rows": int(snapshots.shape[0]),
        "games_with_entry_observation": int(select_diag["games_with_entry_observation"]),
        "games_without_entry_observation": int(select_diag["games_without_entry_observation"]),
        "games_entry_observation_too_old": int(select_diag["games_entry_observation_too_old"]),
        "games_with_score_and_v2": int(len(v2_games)),
        "shared_v2_v4_games": int(len(valid)),
        "v4_feature_missing_games": int((~joined["v4_feature_matched"]).sum()),
        "v4_score_mismatch_games": int(joined.attrs.get("v4_score_mismatch_rows", 0)),
        "beta": float(args.beta),
        "beta_source": DEFAULT_BETA_SOURCE,
        "target_season": int(args.target_season),
        "min_team_games": int(args.min_team_games),
        "v4_feature_reconstructed_not_frozen": True,
        "v2_history_strictly_prior": True,
        "v4_history_strictly_prior": True,
        "entry_prices_used_for_settlement": True,
        "later_observations_used_only_for_clv": True,
        "raw_prices_committed_to_git": False,
        **snapshot_diag,
        **select_diag,
        **score_diag,
        **v4_diag,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    games_path = output / "v4_phase3_blend_market_games.csv"
    metrics_path = output / "v4_phase3_blend_market_accuracy.csv"
    bets_path = output / "v4_phase3_blend_market_bets.csv"
    diagnostics_path = output / "v4_phase3_blend_market_quality.json"
    report_path = output / "v4_phase3_blend_market_report.md"
    betting.to_csv(games_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bet_summary.to_csv(bets_path, index=False)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    report_path.write_text(
        build_report(metrics, bet_summary, diagnostics, float(args.beta)),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2, default=str))
    print("\nAccuracy:")
    print(metrics.to_string(index=False) if not metrics.empty else "No matched V4 games")
    print("\nBetting:")
    print(bet_summary.to_string(index=False) if not bet_summary.empty else "No matched V4 games")
    print(f"\nReport: {report_path}")
    return betting, metrics, diagnostics


def source_check(args: argparse.Namespace) -> dict[str, Any]:
    features, diagnostics = build_current_v4_predictions(
        target_season=args.target_season, min_team_games=args.min_team_games
    )
    diagnostics["source_check_only"] = True
    diagnostics["sample_v4_rows"] = features[
        ["game_date", "home_team", "away_team", "home_games_prior",
         "away_games_prior", "projected_total", "max_history_date"]
    ].head(10).to_dict(orient="records")
    print(json.dumps(diagnostics, indent=2, default=str))
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-file", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--scores-file", default=DEFAULT_SCORES)
    parser.add_argument("--score-patch-dir", default=str(DEFAULT_SCORE_PATCH_DIR))
    parser.add_argument("--score-patch-files", nargs="*", default=None)
    parser.add_argument("--bookmaker", default=BOOKMAKER)
    parser.add_argument("--entry-minutes", type=int, default=ENTRY_MINUTES)
    parser.add_argument("--max-entry-age-minutes", type=int, default=MAX_ENTRY_AGE_MINUTES)
    parser.add_argument("--target-season", type=int, default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--min-team-games", type=int, default=5)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--check-advanced-source-only", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if not 0 <= arguments.beta <= 1:
        raise SystemExit("--beta must be between 0 and 1.")
    if arguments.check_advanced_source_only:
        source_check(arguments)
    else:
        run(arguments)
