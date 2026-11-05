"""Streamlit panel for prospective Stage C player-strength shadow forecasts."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from live_market_testbench_engine import (
    calculate_game_components,
    load_pace_history,
    load_team_id_map,
)
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


def _build_predictions(captured_rows: list[dict]) -> pd.DataFrame:
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
    pace_history = load_pace_history()
    team_map = load_team_id_map()
    output: list[dict] = []

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
                score_history, pace_history, team_map,
                use_pace=False, h2h_mode="off", rest_mode="off",
                v5_cache=v5_cache,
            )
            if not prediction["point_in_time_cutoff"]:
                raise ValueError(f"V5 history cutoff failed for {game_key}.")
            expected_possessions = prediction["pace"].get("expected_matchup_pace_l10")
            if not prediction["pace"].get("available") or expected_possessions is None:
                raise ValueError(f"Expected possessions unavailable for {game_key}.")

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
                "Player total − DK": edge,
                "Directional signal (not a bet)": _signal(edge),
                "Decision": "Shadow test only — No Bet",
                "Expected possessions": round(float(expected_possessions), 3),
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
    return result.sort_values("Tip (Lagos)").reset_index(drop=True)


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
                st.session_state["player_strength_live_predictions"] = (
                    _build_predictions(captured_market_rows)
                )
                st.session_state["player_strength_live_error"] = None
            except Exception as exc:
                st.session_state["player_strength_live_predictions"] = None
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
