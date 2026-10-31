from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from integrated_totals_lab import (
    append_integrated_observations,
    evaluate_market_game,
    make_integrated_observation,
)
from live_market_testbench_engine import (
    fetch_current_totals_markets,
    load_pace_history,
    load_score_history,
    load_team_id_map,
)

st.title("NBA Totals Research Lab")
st.caption(
    "Integrated comparison inside the isolated research copy of the main app. "
    "Production remains unchanged."
)
st.warning(
    "Research environment only. The Main App column runs the existing official "
    "totals path. V5 and its components are challengers, not production-approved "
    "replacements. No wagers are placed."
)

with st.sidebar:
    st.header("Research market connection")
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        api_key = st.text_input(
            "The Odds API key",
            type="password",
            help="Used for live market retrieval in this session only.",
            key="integrated_lab_odds_key",
        ).strip()
    region = st.selectbox(
        "Market region",
        ["us", "uk", "eu", "au"],
        index=0,
        key="integrated_lab_region",
    )

    st.divider()
    st.header("Research challenger configuration")
    use_pace = st.toggle(
        "Use possession-derived last-10 pace",
        value=True,
        key="integrated_lab_pace",
    )
    h2h_choice = st.selectbox(
        "Research H2H adjustment",
        ["Off", "Legacy fixed 15%", "Learned era-adjusted"],
        index=1,
        key="integrated_lab_h2h",
    )
    rest_choice = st.selectbox(
        "Research rest / schedule adjustment",
        ["Off", "Legacy fixed 0.50", "Learned shared coefficient", "Learned schedule-density"],
        index=1,
        key="integrated_lab_rest",
    )
    refresh = st.button(
        "Refresh live market",
        type="primary",
        use_container_width=True,
        key="integrated_lab_refresh",
    )

h2h_mode = {
    "Off": "off",
    "Legacy fixed 15%": "legacy_15pct",
    "Learned era-adjusted": "learned_era",
}[h2h_choice]
rest_mode = {
    "Off": "off",
    "Legacy fixed 0.50": "legacy_fixed_050",
    "Learned shared coefficient": "learned_shared_0416",
    "Learned schedule-density": "learned_schedule_density",
}[rest_choice]
configuration_label = f"pace={use_pace}; h2h={h2h_choice}; rest={rest_choice}"

if not api_key:
    st.info(
        "Enter your Odds API key in the sidebar, or add ODDS_API_KEY to this "
        "separate Render service's environment, to retrieve live market totals."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def get_integrated_score_history():
    return load_score_history()


@st.cache_resource(show_spinner="Loading possession-based pace history…")
def get_integrated_pace_history():
    return load_pace_history()


@st.cache_data(ttl=90, show_spinner=False)
def get_integrated_market_snapshot(secret_key: str, region_code: str):
    return fetch_current_totals_markets(secret_key, region_code)


if refresh:
    get_integrated_market_snapshot.clear()

try:
    with st.spinner("Retrieving current pre-game totals markets…"):
        markets, market_meta = get_integrated_market_snapshot(api_key, region)
except Exception as exc:
    st.error(f"Market retrieval failed: {type(exc).__name__}: {exc}")
    st.stop()

if not markets:
    st.warning(
        "No future game with a complete same-book Over/Under total pair was "
        "returned. The provider may not have posted a total yet, or the game "
        "may already have started."
    )
    st.caption(f'Market capture (UTC): {market_meta.get("captured_at_utc", "unknown")}')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Upcoming totals markets", len(markets))
c2.metric("Market events returned", market_meta.get("events_received", 0))
c3.metric("Odds API requests used", market_meta.get("requests_used") or "Not reported")
c4.metric("Requests remaining", market_meta.get("requests_remaining") or "Not reported")
st.caption(
    f'Market capture (UTC): {market_meta.get("captured_at_utc", "unknown")} · '
    f"Region: {region.upper()}"
)

with st.spinner("Loading the shared score history and possession pace source…"):
    try:
        score_history = get_integrated_score_history()
        pace_history = get_integrated_pace_history() if use_pace else None
        team_id_map = load_team_id_map()
    except Exception as exc:
        st.error(f"Historical feature load failed: {type(exc).__name__}: {exc}")
        st.stop()

history_max = pd.to_datetime(score_history["game_date"], errors="coerce").max()
st.caption(
    f"Shared historical score rows: {len(score_history):,} · "
    f"history through: {history_max.date() if pd.notna(history_max) else 'unknown'} · "
    f'possession source rows: {pace_history.get("team_game_rows", 0):,}'
    if pace_history else
    f"Shared historical score rows: {len(score_history):,} · "
    f"history through: {history_max.date() if pd.notna(history_max) else 'unknown'} · "
    "possession pace disabled"
)

if pace_history:
    st.caption(
        f'Pace seasons loaded: {pace_history.get("seasons_loaded", [])} · '
        "possession records are filtered strictly before each target game date."
    )

st.markdown(
    """
    **How this comparison works:** Both engines receive the same home and away
    teams, Lagos target date, score-history dataframe and market median. The
    Main App projection is generated by the production totals prediction
    function with experimental adjustments disabled, matching the main app's
    official path. Research variants use V5 as their raw baseline.
    """
)

v5_cache = {"v3": {}, "v4": {}}
results = []
errors = []
progress = st.progress(0, text="Running main-app baseline and V5 challenger…")

for index, market in enumerate(markets):
    try:
        result = evaluate_market_game(
            market,
            score_history,
            pace_history,
            team_id_map,
            use_pace=use_pace,
            h2h_mode=h2h_mode,
            rest_mode=rest_mode,
            v5_cache=v5_cache,
        )
        result["market"] = market
        results.append(result)
    except Exception as exc:
        errors.append({
            "game": f'{market.get("away_team")} @ {market.get("home_team")}',
            "error": f"{type(exc).__name__}: {exc}",
        })
    progress.progress(
        (index + 1) / len(markets),
        text=f"Processed {index + 1}/{len(markets)} games",
    )

if errors:
    with st.expander(f"Games not processed ({len(errors)})"):
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

if not results:
    st.error("No games could be evaluated. See the game errors above.")
    st.stop()

summary_rows = []
for result in results:
    market = result["market"]
    main = result["main_prediction"]
    research = result["research_prediction"]
    variants = research["composed"]["predictions"]
    selected = result["variant_signals"]["Selected test-bench configuration"]
    line = result["market_line_used_for_both"]
    tip = pd.to_datetime(market["commence_time_utc"], utc=True).tz_convert(
        "Africa/Lagos"
    ).strftime("%d %b %H:%M")
    summary_rows.append({
        "Game": f'{market["away_team"]} @ {market["home_team"]}',
        "Tip (Lagos)": tip,
        "Shared market": line,
        "Books": market["bookmaker_count"],
        "Main app total": main["projected_total"],
        "Main app edge": main["edge"],
        "Main app signal": main["recommendation"],
        "V5 raw": variants["V5 raw"],
        "V5 + pace": variants["V5 + learned last-10 pace"],
        "V5 + H2H": variants["V5 + legacy H2H 15%"],
        "V5 + rest": variants["V5 + legacy rest 0.50"],
        "V5 + all legacy": variants["V5 + pace + legacy H2H + legacy rest"],
        "Selected research": selected["projected_total"],
        "Research edge": selected["edge"],
        "Research signal": selected["signal"],
        "Selected − main": round(selected["projected_total"] - main["projected_total"], 2),
        "Same inputs": result["same_inputs"],
    })

st.subheader("Main app versus research challenger")
st.dataframe(
    pd.DataFrame(summary_rows).round(2),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "The Main app signal is the existing totals model's provisional edge label. "
    "Research signals use the same edge thresholds solely for comparison; they "
    "are not approved betting recommendations."
)

st.subheader("Game-by-game reconciliation")
for result in results:
    market = result["market"]
    main = result["main_prediction"]
    research = result["research_prediction"]
    selected = result["variant_signals"]["Selected test-bench configuration"]
    line = result["market_line_used_for_both"]
    label = (
        f'{market["away_team"]} @ {market["home_team"]} · '
        f'market {line:.1f} · main {main["projected_total"]:.1f} · '
        f'research {selected["projected_total"]:.1f}'
    )
    with st.expander(label, expanded=False):
        a, b, c, d = st.columns(4)
        a.metric("Shared market total", f"{line:.2f}")
        b.metric("Main app official total", f'{main["projected_total"]:.2f}')
        c.metric("V5 raw total", f'{research["raw_total"]:.2f}')
        d.metric("Selected research total", f'{selected["projected_total"]:.2f}')

        st.markdown("**Main app official prediction**")
        st.write({
            "home points": main["projected_home_points"],
            "away points": main["projected_away_points"],
            "edge": main["edge"],
            "recommendation": main["recommendation"],
            "Points Engine H2H adjustment": main["points_engine_h2h_adjustment"],
            "Points Engine H2H games": main["points_engine_h2h_games_used"],
            "rest adjustment already included in baseline": main["rest_adjustment"],
            "experimental adjustment applied": main["experimental_adjustments_applied"],
            "experimental total diagnostic (not official)": main["experimental_projected_total"],
            "history rows used after target-date cutoff": main["history_rows"],
        })

        st.markdown("**V5 and component reconciliation**")
        component_rows = []
        for name, signal in result["variant_signals"].items():
            component_rows.append({
                "Variant": name,
                "Projected total": signal["projected_total"],
                "Edge vs shared market": signal["edge"],
                "Signal (research only)": signal["signal"],
            })
        st.dataframe(
            pd.DataFrame(component_rows).round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.write({
            "pace adjustment": research["pace"].get("adjustment", 0.0),
            "pace feature available": research["pace"].get("available", False),
            "pace reason": research["pace"].get("reason", ""),
            "legacy H2H adjustment": research["h2h"].get("legacy_adjustment", 0.0),
            "learned H2H adjustment": research["h2h"].get("learned_era_adjustment", 0.0),
            "H2H meetings used": research["h2h"].get("games", 0),
            "legacy rest adjustment": research["rest_adjustments"]["legacy_fixed_050"],
            "learned shared rest adjustment": research["rest_adjustments"]["learned_shared_0416"],
            "schedule-density adjustment": research["rest_adjustments"]["learned_schedule_density"],
            "home rest days": research["home_rest"]["rest_days"],
            "away rest days": research["away_rest"]["rest_days"],
            "main-app history cutoff check": result["main_history_cutoff_ok"],
            "research strict prior-date input check": research["point_in_time_cutoff"],
            "both engines use same inputs": result["same_inputs"],
        })

        quotes = pd.DataFrame(market["bookmaker_quotes"])
        if not quotes.empty:
            st.markdown("**Bookmaker quotes used to calculate the common median**")
            st.dataframe(quotes, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Matched paper-observation ledger")
st.write(
    "Save the main app baseline, every V5 variant, and the identical market "
    "snapshot together. This creates research records only; no stake, wager, "
    "or execution fields are recorded."
)

if st.button("Save this slate to the matched research ledger", use_container_width=True):
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = [
        make_integrated_observation(result, observed_at, configuration_label)
        for result in results
    ]
    saved = append_integrated_observations(rows)
    st.success(
        f'Saved {saved.get("rows_added", 0)} matched observations to {saved.get("path")}.'
    )

ledger_path = Path("outputs/integrated_totals_lab/paper_observations.csv")
if ledger_path.exists():
    try:
        ledger = pd.read_csv(ledger_path)
        st.caption(f"Matched observations recorded locally: {len(ledger):,}")
        st.dataframe(ledger.tail(25), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not read the local observation ledger: {exc}")
else:
    st.info("No matched paper observations have been saved in this instance yet.")

with st.expander("Interpretation and validation status"):
    st.markdown(
        """
        - The Main App value is the unchanged production Points Engine output.
          It includes the main app's H2H and rest calculations as part of that baseline.
        - V5 raw is a distinct model: 55% V3 plus 45% V4-90. It excludes pace,
          H2H and rest before the challenger adjustments are applied.
        - Pace uses possession-derived last-10 pace and the development-fitted
          beta of 0.9673. Its holdout MAE improvement was small and its bias increased.
        - H2H and rest challenger variants are exposed for research. Their
          historical audits did not establish dependable gains.
        - Every historical input is restricted to games strictly before the
          target date. The page displays a per-game same-input/cutoff check.
        - The market median is passed to both engines. This removes the previous
          comparison problem where different bookmaker lines could create
          different edges and recommendations.
        - Free-host filesystems are ephemeral. Export the ledger regularly;
          persistent database storage is not configured in this environment.
        - Historical point-prediction accuracy does not by itself establish ROI.
        """
    )
