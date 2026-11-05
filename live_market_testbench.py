from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from research.collect_public_odds_snapshots import PAGE_URL, parse_odds_page
from research.objective5_market_comparison import run as run_objective5
from research.objective6_bet_selection import run as run_objective6
from research.objective7_profitability_validation import run as run_objective7
from live_market_testbench_engine import PACE_LAST10_BETA, apply_live_market_freshness, apply_player_strength_calibration, calculate_game_components, evaluate_pace_challenger, load_pace_history, load_score_history, load_team_id_map

CAPTURE_MAX_PAGE_AGE_MINUTES = 75  # DunkAPI public board refreshes hourly.
LIVE_MARKET_MAX_PAGE_AGE_MINUTES = 20  # Keep the stricter gate for live-market eligibility.

st.set_page_config(page_title="NBA Totals Research Test Bench", page_icon="🏀", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def cached_pace_history():
    """Cache the public possession-based pace source for one hour per app process."""
    return load_pace_history()

st.title("NBA Totals — Real-Game Prediction & Research Bench")
st.caption("Upcoming NBA game projections using frozen V5 scoring, live sportsbook totals, and chronological validation. Research only.")
st.warning(
    "No wagers are placed. No production model, live app, or paper-trading ledger is changed. "
    "A live market quote is not automatically an eligible T−120 entry price, and this bench does not claim profitability."
)

if "captured_market_rows" not in st.session_state:
    st.session_state.captured_market_rows = []
if "last_capture_meta" not in st.session_state:
    st.session_state.last_capture_meta = None
if "last_validation" not in st.session_state:
    st.session_state.last_validation = None
if "last_pace_validation" not in st.session_state:
    st.session_state.last_pace_validation = None
if "last_pace_rows" not in st.session_state:
    st.session_state.last_pace_rows = None
if "player_diagnostics" not in st.session_state:
    st.session_state.player_diagnostics = None

with st.sidebar:
    st.subheader("Live market capture")
    st.write("Source: DunkAPI public NBA odds board. This does not use paid DunkAPI API credits.")
    capture = st.button("Capture current market snapshot", type="primary", use_container_width=True)
    st.caption("The public board refreshes hourly. Capture archives the latest published page up to 75 minutes old; quotes older than 20 minutes are clearly marked as archive-only and must not be treated as fresh live-market inputs. Objective 5 separately enforces its T−120 entry rule.")
    st.divider()
    st.subheader("Research rules")
    st.write("Bookmaker: DraftKings")
    st.write("Entry cutoff: T−120 minutes")
    st.write("Maximum entry age: 240 minutes")
    st.write("Residual model: Ridge, alpha 25")
    st.write("Minimum prior training: 150 games")
    st.write("Minimum prior OOS calibration errors: 100")
    st.write("Objective 7 minimum: 100 settled bets across 30 dates")

if capture:
    try:
        response = requests.get(
            PAGE_URL,
            headers={
                "User-Agent": "OluwaBlazee-NBA-Research-TestBench/1.0",
                "Accept": "text/html",
                "Cache-Control": "no-cache, no-store, max-age=0",
                "Pragma": "no-cache",
            },
            timeout=30,
        )
        response.raise_for_status()
        now = datetime.now(timezone.utc)
        rows, meta = parse_odds_page(
            response.text,
            retrieved_at=now,
            max_page_age_minutes=CAPTURE_MAX_PAGE_AGE_MINUTES,
        )
        existing = {(r.get("game_key"), r.get("bookmaker"), r.get("observed_at_utc")) for r in st.session_state.captured_market_rows}
        added = [r for r in rows if (r.get("game_key"), r.get("bookmaker"), r.get("observed_at_utc")) not in existing]
        st.session_state.captured_market_rows.extend(added)
        st.session_state.last_capture_meta = meta
        st.success(f"Archived {len(added)} bookmaker observations across {meta.get('games_seen', 0)} listed games.")
        if float(meta.get("page_age_minutes", 0)) > LIVE_MARKET_MAX_PAGE_AGE_MINUTES:
            st.warning(
                f"This page was published {meta['page_age_minutes']} minutes ago. "
                f"It has been saved for the research archive, but it is older than the "
                f"{LIVE_MARKET_MAX_PAGE_AGE_MINUTES}-minute live-market freshness limit. "
                "Do not treat it as a fresh live-market quote."
            )
    except Exception as exc:
        st.error(f"Market capture failed: {type(exc).__name__}: {exc}")

tab_live, tab_validation, tab_players, tab_guide = st.tabs(["Live market test", "Objectives 5–7 validation", "Player strength diagnostics", "How to use this bench"])

with tab_live:
    st.subheader("Upcoming game predictions")
    st.write("Capture the public odds board to compare the V5 and pace predictions with current sportsbook totals. The bundled historical score file is used automatically.")
    captured = pd.DataFrame(st.session_state.captured_market_rows)
    if captured.empty:
        st.info("No observations captured in this session yet. Use “Capture current market snapshot” in the sidebar.")
    else:
        captured["observed_at_utc"] = pd.to_datetime(captured["observed_at_utc"], utc=True, errors="coerce")
        latest = captured.sort_values("observed_at_utc").groupby(["game_key", "bookmaker"], as_index=False).tail(1)
        latest = apply_live_market_freshness(
            latest, max_age_minutes=LIVE_MARKET_MAX_PAGE_AGE_MINUTES
        )
        display = latest[[
            "game_date", "commence_time_utc", "away_team", "home_team", "bookmaker",
            "total_line", "over_odds_american", "under_odds_american", "observed_at_utc",
            "Observation age (min)", "Publication age (min)", "Live quote eligible",
        ]].copy()
        st.dataframe(display.sort_values(["game_date", "commence_time_utc", "bookmaker"]), use_container_width=True, hide_index=True)
        if st.session_state.last_capture_meta:
            st.caption(
                f"Published as of {st.session_state.last_capture_meta.get('published_as_of_utc')} · "
                f"Observed {st.session_state.last_capture_meta.get('observed_at_utc')} · "
                f"Page age {st.session_state.last_capture_meta.get('page_age_minutes')} minutes · "
                f"Live-market freshness limit {LIVE_MARKET_MAX_PAGE_AGE_MINUTES} minutes"
            )
        st.download_button(
            "Download captured market archive (CSV)",
            data=captured.to_csv(index=False).encode("utf-8"),
            file_name="dunkapi_public_odds_snapshots.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Prediction stack: V5 baseline + possession-based pace challenger")
    score_upload_live = st.file_uploader("Optional: upload a newer historical scores CSV (otherwise the bundled project history is used)", type=["csv"], key="score_live")
    if not captured.empty:
        try:
            if score_upload_live is not None:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                    handle.write(score_upload_live.getvalue())
                    score_path = handle.name
                history = load_score_history(score_path)
                st.caption("Using your uploaded score-history override.")
            else:
                history = load_score_history()
                st.caption(f"Using bundled project score history: {len(history):,} games through {history.game_date.max().date()}.")
            team_map = load_team_id_map()
            try:
                pace_history = cached_pace_history()
            except Exception as pace_exc:
                pace_history = None
                st.warning(f"Pace source unavailable; showing V5 baseline without a pace adjustment. {type(pace_exc).__name__}: {pace_exc}")
            latest_rows = latest[
                latest["commence_time_utc"].gt(pd.Timestamp.now(tz="UTC"))
            ].copy()
            if latest_rows.empty:
                st.warning("No upcoming fixtures are present in the captured market archive.")
            elif not latest_rows["Live quote eligible"].any():
                st.warning(
                    "Showing model-only projections for upcoming fixtures. The archived market quotes are stale; "
                    "market comparisons and directional signals are disabled until both timestamps pass the 20-minute gate."
                )
            projections = []
            cache = {"v3": {}, "v4": {}}
            for game_key, group in latest_rows.groupby("game_key", sort=False):
                first = group.iloc[0]
                fresh_group = group[group["Live quote eligible"]]
                quotes = []
                for _, q in fresh_group.iterrows():
                    def to_decimal(price):
                        x = float(price)
                        return 1.0 + (100.0 / abs(x) if x < 0 else x / 100.0)
                    quotes.append({
                        "bookmaker": q["bookmaker"],
                        "total_line": float(q["total_line"]),
                        "over_odds_decimal": to_decimal(q["over_odds_american"]),
                        "under_odds_decimal": to_decimal(q["under_odds_american"]),
                        "last_update_utc": str(q["published_as_of_utc"]),
                    })
                commence = pd.to_datetime(first["commence_time_utc"], utc=True)
                target_date = commence.tz_convert("America/New_York").date().isoformat()
                prediction = calculate_game_components(
                    str(first["home_team"]), str(first["away_team"]), target_date,
                    history, pace_history, team_map, use_pace=True, h2h_mode="off",
                    rest_mode="off", v5_cache=cache,
                )
                market_median = float(np.median([q["total_line"] for q in quotes])) if quotes else np.nan
                # Fair market probabilities are calculated only from paired Over/Under
                # prices at the same most-common fresh total line. They are market estimates,
                # not probabilities generated or calibrated by our model.
                if quotes:
                    line_counts = pd.Series([q["total_line"] for q in quotes]).value_counts()
                    consensus_line = float(sorted(
                        line_counts[line_counts == line_counts.max()].index.tolist()
                    )[0])
                    same_line_quotes = [
                        q for q in quotes if np.isclose(float(q["total_line"]), consensus_line)
                    ]
                else:
                    consensus_line = None
                    same_line_quotes = []
                fair_over = []
                for q in same_line_quotes:
                    inv_over = 1.0 / float(q["over_odds_decimal"])
                    inv_under = 1.0 / float(q["under_odds_decimal"])
                    fair_over.append(inv_over / (inv_over + inv_under))
                market_fair_over = float(np.median(fair_over)) if fair_over else np.nan
                market_fair_under = 1.0 - market_fair_over if np.isfinite(market_fair_over) else np.nan
                pace = prediction["pace"]
                pace_adjusted_total = float(prediction["composed"]["selected_prediction"])
                difference = float(pace_adjusted_total - market_median) if np.isfinite(market_median) else np.nan
                if not quotes:
                    signal = "Market stale — model only"
                elif difference >= 5.0:
                    signal = "Strong Over"
                elif difference >= 2.5:
                    signal = "Lean Over"
                elif difference <= -5.0:
                    signal = "Strong Under"
                elif difference <= -2.5:
                    signal = "Lean Under"
                else:
                    signal = "No clear edge"
                v5 = prediction["v5"]
                projections.append({
                    "Game": str(first["away_team"]) + " @ " + str(first["home_team"]),
                    "Tip (Lagos)": commence.tz_convert("Africa/Lagos").strftime("%d %b %H:%M"),
                    "V5 away baseline points": round(float(v5["projected_away_points"]), 1),
                    "V5 home baseline points": round(float(v5["projected_home_points"]), 1),
                    "V5 baseline total": round(float(prediction["raw_total"]), 1),
                    "Pace adjustment (pts)": round(float(prediction["composed"]["pace_adjustment_used"]), 2),
                    "Matchup pace": round(float(pace["expected_matchup_pace_l10"]), 2) if pace.get("available") else None,
                    "League pace": round(float(pace["league_pace_prior"]), 2) if pace.get("available") else None,
                    "Pace data status": "Available" if pace.get("available") else pace.get("reason", "Unavailable"),
                    "V5 + pace total": round(pace_adjusted_total, 1),
                    "Market quote status": "Fresh" if quotes else "Archive only",
                    "Market median": round(market_median, 1) if np.isfinite(market_median) else None,
                    "Consensus price line": consensus_line,
                    "Market fair Over %": round(100.0 * market_fair_over, 1) if np.isfinite(market_fair_over) else None,
                    "Market fair Under %": round(100.0 * market_fair_under, 1) if np.isfinite(market_fair_under) else None,
                    "Books at price line": len(same_line_quotes),
                    "V5+pace − market": round(difference, 1) if np.isfinite(difference) else None,
                    "V5 model probability": "Not calibrated",
                    "Residual-adjusted total": "Unavailable — Objective 5 gate",
                    "Bet decision": "No Bet — stale market" if not quotes else "No Bet — evidence gate",
                    "Model signal": signal,
                    "Books": len(quotes),
                    "Archived books": len(group),
                })
            result_frame = pd.DataFrame(projections)
            if not result_frame.empty:
                result_frame = result_frame.sort_values("V5+pace − market", ascending=False)
                st.dataframe(result_frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download prediction sheet (CSV)",
                    data=result_frame.to_csv(index=False).encode("utf-8"),
                    file_name="nba_real_game_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No prediction sheet was produced because the captured archive contains no upcoming fixtures.")
            st.info(
                "Prediction-layer status: V5 is the frozen baseline; V5+pace is the possession-based challenger. "
                "Pace adjustment is displayed separately and is not a calibrated betting probability. Market fair Over/Under percentages are derived from paired bookmaker prices at the consensus line; "
                "they are not our model probabilities. The residual-adjusted total, calibrated model probabilities, "
                "expected value and staking decision remain disabled because Objective 5 has not passed its "
                "timestamped historical-entry-data gate. The safe decision is No Bet until those gates pass."
            )
        except Exception as exc:
            st.error(f"V5 projection could not be calculated: {type(exc).__name__}: {exc}")
    else:
        st.caption("Capture the current odds board to generate projected team scores, pace-adjusted totals, market differences and directional Over/Under signals. The bundled score history is used by default.")

with tab_validation:
    st.subheader("Run the complete historical validation")
    st.write("The bundled historical score file and verified recent score patches are used automatically. You can optionally upload a newer score file; the market archive can be downloaded from the Live market tab and combined across sessions.")
    score_upload = st.file_uploader("Optional: override the bundled historical NBA scores CSV", type=["csv"], key="score_validation")
    bundled_score_path = Path(__file__).resolve().parent / "data" / "historical_nba_scores.csv"
    score_history_available = score_upload is not None or bundled_score_path.is_file()
    snapshot_upload = st.file_uploader("Market snapshots archive CSV", type=["csv"], key="snapshots_validation")
    captured_frame = pd.DataFrame(st.session_state.captured_market_rows)
    if snapshot_upload is not None:
        archive = pd.read_csv(snapshot_upload)
        if not captured_frame.empty:
            archive = pd.concat([archive, captured_frame], ignore_index=True)
    else:
        archive = captured_frame.copy()

    if not archive.empty:
        st.caption(
            f"Snapshot rows loaded: {len(archive):,} · "
            f"Unique games: {archive.game_key.nunique():,} · "
            f"Distinct game dates: {pd.to_datetime(archive.game_date, errors='coerce').dt.date.nunique():,}"
        )
        st.dataframe(archive.tail(10), use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined archive",
            data=archive.to_csv(index=False).encode("utf-8"),
            file_name="dunkapi_public_odds_snapshots.csv",
            mime="text/csv",
        )
    else:
        st.warning("No market archive loaded. Capture snapshots in the Live market tab or upload your saved archive.")

    run_pace_validation = st.button(
        "Run historical V5 vs V5 + pace evaluation",
        disabled=(not score_history_available),
        use_container_width=True,
    )
    if run_pace_validation:
        with st.spinner("Building point-in-time possession features and evaluating the pace challenger…"):
            try:
                if score_upload is not None:
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                        handle.write(score_upload.getvalue())
                        score_path_for_pace = handle.name
                    pace_scores = load_score_history(score_path_for_pace)
                else:
                    pace_scores = load_score_history()
                pace_source = cached_pace_history()
                pace_team_map = load_team_id_map()
                pace_rows, pace_summary = evaluate_pace_challenger(
                    pace_scores, pace_source, pace_team_map,
                    bootstrap_resamples=2000, seed=73,
                )
                st.session_state.last_pace_validation = pace_summary
                st.session_state.last_pace_rows = pace_rows
                st.success("Chronological V5 vs V5+pace evaluation completed.")
            except Exception as exc:
                st.error(f"Pace challenger evaluation stopped: {type(exc).__name__}: {exc}")
                st.caption("No fallback coefficient or non-point-in-time feature was substituted.")

    if st.session_state.get("last_pace_validation"):
        st.markdown("**Pace challenger — chronological historical comparison**")
        pace_summary = st.session_state.last_pace_validation
        pace_ci = pace_summary.get("mae_change_date_cluster_bootstrap_95_ci", [float("nan"), float("nan")])
        pace_cols = st.columns(4)
        pace_cols[0].metric("Evaluated games", f"{pace_summary.get('evaluation_games', 0):,}")
        pace_cols[1].metric("V5 baseline MAE", f"{pace_summary.get('v5_baseline', {}).get('mae', float('nan')):.2f}")
        pace_cols[2].metric("V5 + pace MAE", f"{pace_summary.get('v5_plus_pace', {}).get('mae', float('nan')):.2f}")
        pace_cols[3].metric("MAE change (pace − V5)", f"{pace_summary.get('mae_change_pace_minus_v5', float('nan')):+.2f}")
        st.caption(
            f"Date-cluster 95% CI for MAE change: [{pace_ci[0]:+.2f}, {pace_ci[1]:+.2f}] · "
            f"Evaluation dates: {pace_summary.get('evaluation_dates', 0):,} · "
            "Negative MAE change favours V5+pace; a CI crossing zero is inconclusive."
        )
        comparison = pd.DataFrame([
            {"Model": "Frozen V5", **pace_summary.get("v5_baseline", {})},
            {"Model": "V5 + pace", **pace_summary.get("v5_plus_pace", {})},
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.json(pace_summary)
        if st.session_state.get("last_pace_rows") is not None:
            st.download_button(
                "Download game-level pace evaluation (CSV)",
                data=st.session_state.last_pace_rows.to_csv(index=False).encode("utf-8"),
                file_name="v5_vs_pace_challenger_evaluation.csv",
                mime="text/csv",
                use_container_width=True,
            )

    run_validation = st.button(
        "Run Objectives 5, 6 and 7",
        type="primary",
        disabled=(not score_history_available or archive.empty),
        use_container_width=True,
    )
    if run_validation:
        with st.spinner("Running chronological market, probability, bet-selection and profitability checks…"):
            try:
                with tempfile.TemporaryDirectory(prefix="nba_totals_testbench_") as temp:
                    root = Path(temp)
                    score_path = root / "historical_nba_scores.csv"
                    snapshot_path = root / "market_snapshots.csv"
                    if score_upload is not None:
                        score_path.write_bytes(score_upload.getvalue())
                    else:
                        score_path.write_bytes(bundled_score_path.read_bytes())
                    archive.to_csv(snapshot_path, index=False)
                    score_patch_dir = Path(__file__).resolve().parent / "research" / "data"
                    out5, out6, out7 = root / "objective5", root / "objective6", root / "objective7"
                    summary5 = run_objective5(Namespace(
                        snapshots_file=str(snapshot_path),
                        scores_file=str(score_path),
                        score_patch_dir=str(score_patch_dir),
                        score_patch_files=None,
                        output_dir=str(out5),
                        bookmaker="DraftKings",
                        entry_minutes=120,
                        max_entry_age_minutes=240,
                        min_train_games=150,
                        min_calibration_games=100,
                        ridge_alpha=25.0,
                    ))
                    summary6 = run_objective6(Namespace(
                        input_dir=str(out5),
                        output_dir=str(out6),
                        bankroll_units=100.0,
                        minimum_ev=0.02,
                    ))
                    summary7 = run_objective7(Namespace(
                        objective5_dir=str(out5),
                        objective6_dir=str(out6),
                        output_dir=str(out7),
                    ))
                    st.session_state.last_validation = {
                        "objective5": summary5,
                        "objective6": summary6,
                        "objective7": summary7,
                    }
                st.success("All three objective runners completed.")
            except Exception as exc:
                st.error(f"Validation stopped safely: {type(exc).__name__}: {exc}")
                st.caption("No production service or betting ledger was changed.")

    if st.session_state.get("last_validation"):
        result = st.session_state.last_validation
        st.markdown("**Objective 5 — Market and probability validation**")
        st.json(result["objective5"])
        st.markdown("**Objective 6 — Bet selection (research only)**")
        st.json(result["objective6"])
        st.markdown("**Objective 7 — Independent profitability validation**")
        st.json(result["objective7"])

with tab_players:
    st.subheader("Point-in-time player strength and full-rotation diagnostics")
    st.write(
        "This panel rebuilds player RAPM from strictly prior lineup stints, maps the official "
        "target-date NBA injury report, and projects every eligible player across each team's "
        "240 regulation minutes."
    )
    st.warning(
        "The player calibration may show an unvalidated shadow estimate so its parameter impact is visible. "
        "It is eligible as a forecast only after passing chronological out-of-sample validation. "
        "V5+pace and V5+player are separate challengers; do not add their adjustments together without "
        "a separately validated combined model."
    )
    player_market = pd.DataFrame(st.session_state.captured_market_rows)
    if player_market.empty:
        st.info("Capture a market snapshot first to load the upcoming fixtures.")
    else:
        target_dates = sorted(pd.to_datetime(player_market.game_date, errors="coerce").dropna().dt.date.astype(str).unique())
        player_target_date = st.selectbox("Player diagnostics target date", target_dates)
        run_player_diagnostics = st.button(
            "Build player-strength and full-rotation diagnostics",
            type="primary",
            use_container_width=True,
        )
        if run_player_diagnostics:
            with st.spinner("Rebuilding point-in-time RAPM and mapping the official injury report…"):
                try:
                    from research.prospective_player_data import build_pregame_player_state
                    from research.prospective_rotation_projection import build_target_date_rotations
                    if score_upload_live is not None:
                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as score_handle:
                            score_handle.write(score_upload_live.getvalue())
                            player_score_path = score_handle.name
                        player_score_history = load_score_history(player_score_path)
                    else:
                        player_score_history = load_score_history()
                    target_fixtures = player_market.copy()
                    target_fixtures["game_date"] = pd.to_datetime(target_fixtures.game_date, errors="coerce").dt.normalize()
                    target_fixtures = target_fixtures[target_fixtures.game_date.eq(pd.Timestamp(player_target_date))]
                    target_fixtures = target_fixtures[["game_key", "game_date", "away_team", "home_team"]].drop_duplicates()
                    player_state = build_pregame_player_state(player_target_date)
                    rotation_result = build_target_date_rotations(player_state, target_fixtures, player_target_date)
                    player_rows = rotation_result["projections"].copy()
                    player_name_map = player_state["player_names"][["player_id", "name"]].drop_duplicates("player_id")
                    team_name_map = player_state["team_names"][["team_id", "full_name"]].drop_duplicates("team_id")
                    player_rows = player_rows.merge(player_name_map, on="player_id", how="left").merge(
                        team_name_map, on="team_id", how="left"
                    )
                    player_rows["minute_change"] = player_rows.expected_minutes - player_rows.baseline_minutes
                    player_rows["offensive_rapm_minute_delta"] = player_rows.o_delta_contribution
                    player_rows["defensive_rapm_minute_delta"] = player_rows.d_delta_contribution
                    summaries = pd.DataFrame(rotation_result["summaries"])
                    team_map = load_team_id_map()
                    try:
                        pace_source = cached_pace_history()
                    except Exception as pace_exc:
                        pace_source = None
                        st.warning(f"Pace source unavailable; player diagnostics will show RAPM features without a pace-derived raw signal. {type(pace_exc).__name__}: {pace_exc}")
                    game_features = []
                    v5_cache = {"v3": {}, "v4": {}}
                    for fixture in target_fixtures.itertuples(index=False):
                        target_game = rotation_result["target_games"]
                        target_game = target_game[target_game.game_id.astype(str).eq(str(fixture.game_key))]
                        if target_game.empty:
                            raise RuntimeError(f"Missing mapped target fixture {fixture.game_key}")
                        ids = target_game.iloc[0]
                        home = summaries[(summaries.game_id.astype(str).eq(str(fixture.game_key))) &
                                         summaries.team_id.astype(str).eq(str(ids.home_team_id))]
                        away = summaries[(summaries.game_id.astype(str).eq(str(fixture.game_key))) &
                                         summaries.team_id.astype(str).eq(str(ids.away_team_id))]
                        if len(home) != 1 or len(away) != 1:
                            raise RuntimeError(f"Missing home/away player-strength features for {fixture.game_key}")
                        model_parts = calculate_game_components(
                            fixture.home_team, fixture.away_team, player_target_date,
                            player_score_history, pace_source, team_map, use_pace=True,
                            h2h_mode="off", rest_mode="off", v5_cache=v5_cache,
                        )
                        pace = model_parts["pace"]
                        home_features = home.iloc[0]["features"]
                        away_features = away.iloc[0]["features"]
                        player_feature_values = {
                            "home_off_delta": float(home_features["off_delta"]),
                            "home_def_delta": float(home_features["def_delta"]),
                            "away_off_delta": float(away_features["off_delta"]),
                            "away_def_delta": float(away_features["def_delta"]),
                            "home_minutes_changed": float(home_features["minutes_changed"]),
                            "away_minutes_changed": float(away_features["minutes_changed"]),
                            "home_reported_out_minutes": float(home_features["reported_out_minutes"]),
                            "away_reported_out_minutes": float(away_features["reported_out_minutes"]),
                            "home_questionable_minutes": float(home_features["questionable_minutes"]),
                            "away_questionable_minutes": float(away_features["questionable_minutes"]),
                            "home_reported_players_without_prior_team_history": float(home_features["reported_players_without_prior_team_history"]),
                            "away_reported_players_without_prior_team_history": float(away_features["reported_players_without_prior_team_history"]),
                            "home_players_excluded_by_opposing_report": float(home_features["players_excluded_by_opposing_report"]),
                            "away_players_excluded_by_opposing_report": float(away_features["players_excluded_by_opposing_report"]),
                        }
                        total_feature_delta = sum(
                            player_feature_values[key]
                            for key in ("home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta")
                        )
                        raw_signal = (
                            float(pace["expected_matchup_pace_l10"]) / 100.0 * total_feature_delta
                            if pace.get("available") else None
                        )
                        player_calibration = apply_player_strength_calibration(
                            float(model_parts["raw_total"]), player_feature_values, player_target_date
                        )
                        player_adjusted_total = (
                            player_calibration["player_adjusted_total"]
                            if player_calibration.get("available") else None
                        )
                        game_features.append({
                            "Game": f"{fixture.away_team} @ {fixture.home_team}",
                            **player_feature_values,
                            "V5 baseline total": round(float(model_parts["raw_total"]), 1),
                            "Projected matchup pace": pace.get("expected_matchup_pace_l10"),
                            "Pace adjustment (pts)": round(float(model_parts["composed"]["pace_adjustment_used"]), 2),
                            "V5 + pace total": round(float(model_parts["composed"]["selected_prediction"]), 1),
                            "Home offensive delta": player_feature_values["home_off_delta"],
                            "Home defensive delta": player_feature_values["home_def_delta"],
                            "Away offensive delta": player_feature_values["away_off_delta"],
                            "Away defensive delta": player_feature_values["away_def_delta"],
                            "Raw RAPM rotation signal (pts; uncalibrated)": raw_signal,
                            "Player adjustment (pts)": (
                                round(float(player_adjusted_total - model_parts["raw_total"]), 2)
                                if player_adjusted_total is not None else None
                            ),
                            "V5 + player strength total": (
                                round(float(player_adjusted_total), 1)
                                if player_adjusted_total is not None else None
                            ),
                            "Player calibration status": (
                                "Validated challenger" if player_calibration.get("forecast_eligible")
                                else (
                                    "Unvalidated shadow estimate — not used in decisions"
                                    if player_calibration.get("available")
                                    else player_calibration.get("reason", "Unavailable")
                                )
                            ),
                            "Player OOS games": (player_calibration.get("validation") or {}).get("games"),
                            "Player OOS dates": (player_calibration.get("validation") or {}).get("dates"),
                            "Player MAE change vs V5": (player_calibration.get("validation") or {}).get("mae_change_player_minus_v5"),
                            "Player date-cluster 95% CI": (player_calibration.get("validation") or {}).get("date_cluster_ci_95"),
                        })
                    st.session_state.player_diagnostics = {
                        "diagnostics": player_state["diagnostics"],
                        "injury_report": rotation_result["report_metadata"],
                        "game_features": pd.DataFrame(game_features),
                        "players": player_rows,
                    }
                except Exception as exc:
                    st.error(f"Player-strength diagnostics stopped safely: {type(exc).__name__}: {exc}")
                    st.caption("No player-adjusted total was emitted; missing or incomplete inputs were not imputed.")

        if st.session_state.get("player_diagnostics"):
            diagnostics = st.session_state.player_diagnostics
            st.markdown("**Pregame source and RAPM diagnostics**")
            st.json(diagnostics["diagnostics"])
            calibration_path = Path(__file__).resolve().parent / "research" / "live_player_strength_calibration.json"
            if calibration_path.is_file():
                calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
                calibration_features = calibration_payload.get("features", [])
                live_feature_columns = set(diagnostics["game_features"].columns)
                feature_contract_matches = bool(calibration_features) and set(calibration_features).issubset(live_feature_columns)
                if not feature_contract_matches:
                    st.warning(
                        "The stored calibration artifact predates the current live player-feature contract. "
                        "Raw RAPM and rotation contributions are shown, but no player-adjusted total is emitted "
                        "until the corrected historical recalibration is published."
                    )
                else:
                    st.markdown("**Player calibration parameter audit**")
                    st.caption(
                        f"Artifact status: {calibration_payload.get('status', 'unknown')} · "
                        f"Forecast enabled: {bool(calibration_payload.get('forecast_enabled', False))} · "
                        f"Training cutoff exclusive: {calibration_payload.get('training_cutoff_exclusive', 'unknown')} · "
                        "Ridge coefficients are standardized feature weights, not direct points-per-unit."
                    )
                    parameter_frame = pd.DataFrame({
                        "Feature": calibration_features,
                        "Standardized Ridge coefficient": calibration_payload.get("ridge", {}).get("coefficients", []),
                    })
                    if not parameter_frame.empty:
                        parameter_frame["Absolute coefficient"] = parameter_frame["Standardized Ridge coefficient"].abs()
                        st.dataframe(
                            parameter_frame.sort_values("Absolute coefficient", ascending=False),
                            use_container_width=True, hide_index=True,
                        )
                    validation = calibration_payload.get("validation") or {}
                    if validation:
                        player_ci = validation.get("date_cluster_ci_95", [float("nan"), float("nan")])
                        player_cols = st.columns(4)
                        player_cols[0].metric("Player OOS games", f"{validation.get('games', 0):,}")
                        player_cols[1].metric("OOS dates", f"{validation.get('dates', 0):,}")
                        player_cols[2].metric(
                            "MAE change vs V5",
                            f"{validation.get('mae_change_player_minus_v5', float('nan')):+.2f}",
                        )
                        player_cols[3].metric(
                            "Validation gate",
                            "PASS" if validation.get("gate_passed") else "SHADOW",
                        )
                        st.caption(
                            f"Date-cluster 95% CI for player MAE change: "
                            f"[{player_ci[0]:+.2f}, {player_ci[1]:+.2f}] · "
                            "Negative MAE change favours the player challenger."
                        )
                        st.json(validation)
            st.markdown("**Game-level player-strength contributions**")
            st.dataframe(diagnostics["game_features"], use_container_width=True, hide_index=True)
            st.markdown("**Full individual player rotation**")
            display_players = diagnostics["players"][[
                "game_id", "full_name", "name", "rotation_rank", "status",
                "availability_probability", "baseline_minutes", "expected_minutes",
                "minute_change", "o_rapm", "d_rapm",
                "offensive_rapm_minute_delta", "defensive_rapm_minute_delta",
            ]].rename(columns={
                "game_id": "Game", "full_name": "Team", "name": "Player",
                "rotation_rank": "Rotation rank", "status": "Official status",
                "availability_probability": "Availability probability",
                "baseline_minutes": "Baseline minutes", "expected_minutes": "Expected minutes",
                "minute_change": "Minute change", "o_rapm": "Offensive RAPM (pts/100 poss)",
                "d_rapm": "Defensive RAPM (pts/100 poss)",
                "offensive_rapm_minute_delta": "Offensive RAPM × minute delta (pts/100 poss)",
                "defensive_rapm_minute_delta": "Defensive RAPM × minute delta (pts/100 poss)",
            })
            st.dataframe(display_players, use_container_width=True, hide_index=True)
            st.download_button(
                "Download full player rotation diagnostics (CSV)",
                data=display_players.to_csv(index=False).encode("utf-8"),
                file_name="pregame_player_strength_rotation_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab_guide:
    st.subheader("Operating instructions")
    st.markdown(
        """
        1. Open Real game predictions and capture the public odds board periodically. Capture at least one observation no later than 120 minutes before each game, then capture again later before tip.
        2. Download the market archive after each session. Keep your own copy because the free web service may restart and clear session data.
        3. The bundled historical NBA score file and committed score patches load automatically; upload a newer score file only if you need to override them.
        4. Use the Player strength diagnostics tab to inspect RAPM, expected minutes, official availability and the separately validated player challenger.
        5. Run the three market objectives together. Objective 5 enforces DraftKings, T−120, the 240-minute maximum entry age, 150 prior training games and 100 prior out-of-sample calibration errors.
        6. Objective 6 only produces candidate decisions if the development evidence gate passes. Otherwise it returns No Bet.
        7. Objective 7 settles only frozen holdout decisions. It requires at least 100 settled bets across 30 distinct dates before any positive-profitability evidence can be considered.
        """
    )
    st.warning(
        "This app uses the public DunkAPI odds-board page and does not place bets. "
        "Snapshots captured after the T−120 cutoff cannot be treated as entry prices. "
        "Do not interpret model-minus-market differences or small samples as an edge."
    )
