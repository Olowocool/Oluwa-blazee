"""Streamlit panel for prospective Stage C player-strength shadow forecasts."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from live_market_testbench_engine import (
    calculate_game_components,
    load_team_id_map,
)
from research.v4_phase1_core import connect_and_load, weighted_recent_mean, PRIOR_STRENGTH_GAMES
from research.build_prospective_pregame_records import build_v5_score_history
from research.live_player_strength import (
    build_player_strength_features,
    load_calibration,
    predict_player_adjusted_total,
)
from research.prospective_player_data import build_pregame_player_state
from research.prospective_player_snapshot import latest_draftkings_quotes, utc_timestamp
from research.prospective_rotation_projection import build_target_date_rotations


def _signal(edge: float) -> str:
    if edge >= 5.0:
        return "Strong Over direction"
    if edge >= 2.5:
        return "Lean Over direction"
    if edge <= -5.0:
        return "Strong Under direction"
    if edge <= -2.5:
        return "Lean Under direction"
    return "No clear directional difference"


def _expected_possessions_from_v4_history(
    raw: pd.DataFrame, target_date: str, home_team_id: str, away_team_id: str
) -> float:
    """Reproduce V4 Phase 1's expected_possessions feature for an upcoming fixture."""
    target = pd.Timestamp(target_date).normalize()
    season = int(target.year + 1 if target.month >= 7 else target.year)
    required = {"season", "game_date", "game_id", "team_id", "fga", "fta", "oreb", "tov"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"V4 Phase 1 possession source missing fields: {sorted(missing)}")
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    d["season"] = pd.to_numeric(d.season, errors="coerce")
    d["team_id"] = d.team_id.astype(str)
    for col in ("fga", "fta", "oreb", "tov"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["game_date", "season", "fga", "fta", "oreb", "tov"])
    d["formula_poss"] = d.fga + 0.44 * d.fta - d.oreb + d.tov
    d = d[d.formula_poss.gt(0)].copy()
    current = d[(d.season.eq(season)) & d.game_date.lt(target)].copy()
    prior = d[d.season.eq(season - 1)].copy()
    if current.duplicated(["season", "game_id", "team_id"]).any():
        raise ValueError("Duplicate current-season team-game rows in V4 pace history")
    home = current[current.team_id.eq(str(home_team_id))]
    away = current[current.team_id.eq(str(away_team_id))]
    home_n, away_n = len(home), len(away)
    if min(home_n, away_n) < 5:
        raise ValueError(
            f"V4 Phase 1 pace warm-up incomplete for target teams: home={home_n}, away={away_n}; need 5 each"
        )
    prior_league = (
        float(prior.formula_poss.mean()) if not prior.empty else 100.0
    )
    league = float(current.formula_poss.mean()) if not current.empty else prior_league
    prior_by_team = {
        str(team): weighted_recent_mean(group, "formula_poss")
        for team, group in prior.groupby("team_id", sort=False)
    }
    home_now = weighted_recent_mean(home, "formula_poss")
    away_now = weighted_recent_mean(away, "formula_poss")
    home_previous = prior_by_team.get(str(home_team_id), prior_league)
    away_previous = prior_by_team.get(str(away_team_id), prior_league)
    blend = min(
        0.90,
        (0.5 * (home_n + away_n))
        / (0.5 * (home_n + away_n) + PRIOR_STRENGTH_GAMES),
    )
    home_pace = blend * home_now + (1.0 - blend) * home_previous
    away_pace = blend * away_now + (1.0 - blend) * away_previous
    expected = float(
        league + 0.5 * ((home_pace - league) + (away_pace - league))
    )
    if not pd.notna(expected) or not float("-inf") < expected < float("inf") or expected <= 0:
        raise ValueError("V4 Phase 1 expected possessions are invalid")
    return expected


def _build_predictions(captured_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not captured_rows:
        raise ValueError("Capture the public odds board before generating player forecasts.")
    archive = pd.DataFrame(captured_rows)
    now = datetime.now(timezone.utc)
    # Both publication age and collector observation age must be <=20 minutes.
    quotes = latest_draftkings_quotes(
        archive, now, max_observation_age_minutes=20.0,
        max_publication_age_minutes=20.0,
    )
    quotes["game_date"] = pd.to_datetime(
        quotes.game_date, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if quotes.game_date.isna().any():
        raise ValueError("A fresh DraftKings quote has an invalid fixture date.")

    calibration = load_calibration()
    # Use the same V4 Phase 1 possession feature definition used during Stage C
    # training; do not substitute a differently weighted pace proxy at inference.
    target_seasons = {
        int(pd.Timestamp(value).year + 1 if pd.Timestamp(value).month >= 7 else pd.Timestamp(value).year)
        for value in quotes.game_date
    }
    phase1_raw, _phase1_metadata = connect_and_load(last_season=max(target_seasons))
    team_map = load_team_id_map()
    output: list[dict] = []
    player_output: list[dict] = []

    # Build each date's player state once. Every source adapter enforces a
    # strict pre-target cutoff; target-date player logs are never inputs.
    for target_date, day_quotes in quotes.groupby("game_date", sort=True):
        state = build_pregame_player_state(target_date)
        rotation_result = build_target_date_rotations(state, day_quotes, target_date)
        score_history = build_v5_score_history(state, target_date)
        target_games = rotation_result["target_games"].set_index("game_id")
        summary_lookup = {
            (str(row["game_id"]), str(row["team_id"])): row["features"]
            for row in rotation_result["summaries"]
        }
        report = rotation_result["report_metadata"]
        report_time = utc_timestamp(report["snapshot_time_utc"])
        report_hash = str(report["sha256"])
        if not report_hash:
            raise ValueError("Official injury report is missing its content hash.")
        if pd.Timestamp(state["diagnostics"]["rapm_max_history_date"]) >= pd.Timestamp(target_date):
            raise ValueError("Current-season RAPM contains same-day/future player history.")

        v5_cache: dict = {}
        current_season = int(target.year + 1 if target.month >= 7 else target.year)
        player_names = state["player_names"]
        player_names = player_names[
            pd.to_numeric(player_names.season, errors="coerce").eq(current_season)
        ]
        player_name_lookup = dict(zip(
            player_names.player_id.astype(str), player_names.name.astype(str)
        ))
        team_name_lookup = dict(zip(
            state["team_names"].team_id.astype(str),
            state["team_names"].full_name.astype(str),
        ))
        for player in rotation_result["projections"].to_dict("records"):
            game_key = str(player["game_id"])
            team_id = str(player["team_id"])
            fixture = target_games.loc[game_key]
            side = "Home" if team_id == str(fixture["home_team_id"]) else "Away"
            player_output.append({
                "Game key": game_key,
                "Target date": str(target.date()),
                "Team": team_name_lookup.get(team_id, team_id),
                "Side": side,
                "Player": player_name_lookup.get(
                    str(player["player_id"]), f'Unknown player {player["player_id"]}'
                ),
                "Player ID": str(player["player_id"]),
                "Rotation rank": int(player["rotation_rank"]),
                "Expected minutes": round(float(player["expected_minutes"]), 3),
                "Baseline minutes": round(float(player["baseline_minutes"]), 3),
                "Availability probability": round(float(player["availability_probability"]), 4),
                "Official status": str(player["status"]),
                "Offensive RAPM": round(float(player.get("o_rapm", 0.0)), 4),
                "Defensive RAPM": round(float(player.get("d_rapm", 0.0)), 4),
                "Offensive minutes contribution": round(float(player.get("o_delta_contribution", 0.0)), 5),
                "Defensive minutes contribution": round(float(player.get("d_delta_contribution", 0.0)), 5),
            })
        for quote in day_quotes.to_dict("records"):
            game_key = str(quote["game_key"])
            if game_key not in target_games.index:
                raise ValueError(f"Target team mapping missing for {game_key}.")
            fixture = target_games.loc[game_key]
            if isinstance(fixture, pd.DataFrame):
                raise ValueError(f"Duplicate target fixture mapping for {game_key}.")
            home_id = str(fixture["home_team_id"])
            away_id = str(fixture["away_team_id"])
            home_features = summary_lookup.get((game_key, home_id))
            away_features = summary_lookup.get((game_key, away_id))
            if home_features is None or away_features is None:
                raise ValueError(f"Full-rotation player features missing for {game_key}.")

            commence = utc_timestamp(quote["commence_time_utc"])
            observed = utc_timestamp(quote["observed_at_utc"])
            published = utc_timestamp(quote["published_as_of_utc"])
            if report_time >= commence or observed >= commence or published >= commence:
                raise ValueError(f"Pregame timestamp gate failed for {game_key}.")
            if observed > now or published > observed + pd.Timedelta(minutes=5):
                raise ValueError(f"Market timestamp ordering failed for {game_key}.")

            prediction = calculate_game_components(
                str(quote["home_team"]), str(quote["away_team"]), target_date,
                score_history, None, team_map,
                use_pace=False, h2h_mode="off", rest_mode="off",
                v5_cache=v5_cache,
            )
            if not prediction["point_in_time_cutoff"]:
                raise ValueError(f"V5 history cutoff failed for {game_key}.")
            expected_possessions = _expected_possessions_from_v4_history(
                phase1_raw, target_date, home_id, away_id
            )

            features = build_player_strength_features(
                home_features, away_features, expected_possessions
            )
            totals = predict_player_adjusted_total(
                prediction["raw_total"], features, calibration
            )
            market_line = float(quote["total_line"])
            edge = round(totals["player_adjusted_total"] - market_line, 2)
            tip_lagos = commence.tz_convert("Africa/Lagos").strftime("%d %b %Y %H:%M")
            output.append({
                "Game": f'{quote["away_team"]} @ {quote["home_team"]}',
                "Tip (Lagos)": tip_lagos,
                "Frozen V5 total": totals["frozen_v5_total"],
                "Player-adjusted total": totals["player_adjusted_total"],
                "Player adjustment": totals["player_adjustment"],
                "DraftKings total": market_line,
                "DraftKings Over odds (American)": int(float(quote["over_odds_american"])),
                "DraftKings Under odds (American)": int(float(quote["under_odds_american"])),
                "Player total − DK": edge,
                "Directional signal (not a bet)": _signal(edge),
                "Decision": "Shadow test only — No Bet",
                "Expected possessions": round(float(expected_possessions), 3),
                "Expected possessions source": "V4 Phase 1 point-in-time formula possessions",
                "Home rotation players": int(
                    next(x["players"] for x in rotation_result["summaries"]
                         if str(x["game_id"]) == game_key and str(x["team_id"]) == home_id)
                ),
                "Away rotation players": int(
                    next(x["players"] for x in rotation_result["summaries"]
                         if str(x["game_id"]) == game_key and str(x["team_id"]) == away_id)
                ),
                "Home projected minutes": round(float(
                    rotation_result["projections"].loc[
                        (rotation_result["projections"].game_id.astype(str) == game_key)
                        & (rotation_result["projections"].team_id.astype(str) == home_id),
                        "expected_minutes",
                    ].sum()
                ), 3),
                "Away projected minutes": round(float(
                    rotation_result["projections"].loc[
                        (rotation_result["projections"].game_id.astype(str) == game_key)
                        & (rotation_result["projections"].team_id.astype(str) == away_id),
                        "expected_minutes",
                    ].sum()
                ), 3),
                "Official injury report UTC": report_time.isoformat(),
                "Official injury report SHA256": report_hash,
                "RAPM history through": state["diagnostics"]["rapm_max_history_date"],
                "Odds published UTC": published.isoformat(),
                "Odds observed UTC": observed.isoformat(),
                "Odds observation age minutes": round(
                    (now - observed).total_seconds() / 60.0, 3
                ),
                "Calibration training games": int(calibration["training_games"]),
                "Calibration training through": calibration["training_last_date"],
                "Forecast generated UTC": now.isoformat(),
                "Game key": game_key,
            })

    result = pd.DataFrame(output)
    if result.empty:
        raise ValueError("No eligible player-strength forecasts were generated.")
    if result["Game key"].duplicated().any():
        raise RuntimeError("Duplicate game forecasts were generated.")
    player_detail = pd.DataFrame(player_output)
    if player_detail.empty:
        raise ValueError("No individual full-rotation player detail was produced.")
    if player_detail.duplicated(["Game key", "Player ID"]).any():
        raise RuntimeError("Duplicate player rows in the live full-rotation evidence.")
    return (
        result.sort_values("Tip (Lagos)").reset_index(drop=True),
        player_detail.sort_values(["Game key", "Side", "Rotation rank"]).reset_index(drop=True),
    )


def render_player_strength_panel(captured_market_rows: list[dict]) -> None:
    st.divider()
    st.subheader("Stage C player-strength live-market test")
    st.write(
        "Uses point-in-time RAPM, official injury status, individual full-rotation "
        "minutes and the frozen V5 baseline. It captures a shadow forecast only; "
        "it does not place wagers or write to the paper-trading ledger."
    )
    try:
        calibration = load_calibration()
        st.caption(
            f'Calibration ready · {int(calibration["training_games"]):,} prior games · '
            f'trained through {calibration["training_last_date"]} · '
            f'{len(calibration["features"])} RAPM/injury features'
        )
    except FileNotFoundError as exc:
        st.warning(
            "Player-strength calibration is not available yet. The live forecast "
            f"button will become available after the calibration workflow completes. ({exc})"
        )
        return
    except Exception as exc:
        st.error(f"Calibration validation failed: {type(exc).__name__}: {exc}")
        return

    generate = st.button(
        "Generate player-strength live projections",
        key="generate_player_strength_live_projections",
        type="primary",
        use_container_width=True,
        disabled=not bool(captured_market_rows),
    )
    if generate:
        with st.spinner(
            "Checking fresh DraftKings prices, current-season player data, the official "
            "injury report, full rotations and point-in-time V5 baselines…"
        ):
            try:
                predictions, player_detail = _build_predictions(captured_market_rows)
                st.session_state["player_strength_live_predictions"] = predictions
                st.session_state["player_strength_live_player_detail"] = player_detail
                st.session_state["player_strength_live_error"] = None
            except Exception as exc:
                st.session_state["player_strength_live_predictions"] = None
                st.session_state["player_strength_live_player_detail"] = None
                st.session_state["player_strength_live_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

    error = st.session_state.get("player_strength_live_error")
    if error:
        st.error(f"Player-strength forecast stopped safely: {error}")
        st.caption(
            "No V5-only fallback is presented as a player-adjusted forecast. "
            "Refresh the market snapshot or retry after the missing official source is available."
        )
    result = st.session_state.get("player_strength_live_predictions")
    if isinstance(result, pd.DataFrame) and not result.empty:
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download player-strength live forecast evidence (CSV)",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="nba_player_strength_live_shadow_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info(
            "These are point-total shadow forecasts, not calibrated Over/Under probabilities. "
            "Directional differences are for prospective evaluation only. All decisions remain No Bet."
        )
        player_detail = st.session_state.get("player_strength_live_player_detail")
        if isinstance(player_detail, pd.DataFrame) and not player_detail.empty:
            with st.expander("Individual player rotation and strength detail"):
                st.dataframe(player_detail, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download full player rotation evidence (CSV)",
                    data=player_detail.to_csv(index=False).encode("utf-8"),
                    file_name="nba_full_rotation_player_strength_evidence.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
