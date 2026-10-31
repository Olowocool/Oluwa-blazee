from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from live_market_testbench_engine import (
    append_paper_observations,
    calculate_game_components,
    fetch_current_totals_markets,
    load_pace_history,
    load_score_history,
    load_team_id_map,
    make_paper_observation,
)

st.set_page_config(
    page_title="NBA Totals Research Test Bench",
    page_icon="🏀",
    layout="wide",
)

st.title("NBA Totals — Live Market Test Bench")
st.caption(
    "Research-only comparison of the validated V5 raw baseline and its "
    "pace, H2H and rest/schedule-density components."
)

st.warning(
    "This is a paper-testing research bench. It does not place wagers, "
    "recommend bets, size stakes or claim profitability. The four component "
    "adjustments are not approved for production."
)

with st.sidebar:
    st.header("Market connection")
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
            help="Used only by this Streamlit session to fetch current NBA totals. "
                 "It is not displayed in the results or written to the paper ledger.",
        ).strip()
    region = st.selectbox("Market region", ["us", "uk", "eu", "au"], index=0)

    st.divider()
    st.header("Component configuration")
    use_pace = st.toggle("Use possession-derived last-10 pace", value=True)
    h2h_choice = st.selectbox(
        "Head-to-head adjustment",
        ["Off", "Legacy fixed 15%", "Learned era-adjusted"],
        index=1,
        help="The H2H audit found only small, uncertain gains. Compare rather than assume.",
    )
    rest_choice = st.selectbox(
        "Rest / schedule adjustment",
        ["Off", "Legacy fixed 0.50", "Learned shared coefficient", "Learned schedule-density"],
        index=1,
        help="Rest variants did not establish a dependable holdout gain.",
    )
    refresh = st.button("Refresh live market", type="primary", use_container_width=True)

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

if not api_key:
    st.info(
        "Add ODDS_API_KEY to the Render service environment, or enter it in the "
        "password field in the sidebar, to load live totals markets."
    )
    st.stop()

@st.cache_data(show_spinner=False)
def get_score_history():
    return load_score_history()

@st.cache_resource(show_spinner="Loading possession-based pace history…")
def get_pace_history():
    return load_pace_history()

@st.cache_data(ttl=90, show_spinner=False)
def get_market_snapshot(secret_key: str, region_code: str):
    # The key is used only as a private cache input and is never rendered.
    return fetch_current_totals_markets(secret_key, region_code)

if refresh:
    get_market_snapshot.clear()

try:
    with st.spinner("Connecting to The Odds API and retrieving current pre-game totals…"):
        markets, market_meta = get_market_snapshot(api_key, region)
except Exception as exc:
    st.error(f"Market retrieval failed: {type(exc).__name__}: {exc}")
    st.stop()

if not markets:
    st.warning(
        "The market feed returned no future games with a complete Over/Under "
        "totals pair from at least one bookmaker. This can happen after tip-off "
        "or when the provider has not posted a totals market."
    )
    st.caption(f"Captured: {market_meta.get('captured_at_utc', 'unknown')}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Upcoming markets", len(markets))
col2.metric("Market events returned", market_meta.get("events_received", 0))
col3.metric("Odds API requests used", market_meta.get("requests_used") or "Not reported")
col4.metric("Requests remaining", market_meta.get("requests_remaining") or "Not reported")
st.caption(
    "Market capture (UTC): "
    f"{market_meta.get('captured_at_utc', 'unknown')} · Region: {region.upper()}"
)

with st.spinner("Loading historical score history and team pace features…"):
    try:
        score_history = get_score_history()
        pace_history = get_pace_history() if use_pace else None
        team_id_map = load_team_id_map()
    except Exception as exc:
        st.error(f"Historical feature load failed: {type(exc).__name__}: {exc}")
        st.stop()

st.caption(
    f"Historical score rows: {len(score_history):,} · "
    f"Score history through: {pd.to_datetime(score_history.game_date).max().date()} · "
    f"Possession source rows: {pace_history.get('team_game_rows', 0):,} "
    f"across seasons {pace_history.get('seasons_loaded', []) if pace_history else 'not loaded'}"
    if pace_history else
    f"Historical score rows: {len(score_history):,} · "
    f"Score history through: {pd.to_datetime(score_history.game_date).max().date()} · "
    "Pace feature disabled"
)

if use_pace and pace_history:
    source_updated = pace_history.get("source_updated", {})
    st.caption(
        "Pace source metadata: team-game logs "
        f"{source_updated.get('team_game_logs', 'unknown')} · games "
        f"{source_updated.get('games', 'unknown')}"
    )

v5_cache = {"v3": {}, "v4": {}}
results = []
errors = []
progress = st.progress(0, text="Calculating V5 and component comparisons…")

for index, market in enumerate(markets):
    try:
        result = calculate_game_components(
            market["home_team"],
            market["away_team"],
            market["game_date_wat"],
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
    progress.progress((index + 1) / len(markets), text=f"Processed {index + 1}/{len(markets)} games")

if errors:
    with st.expander(f"Games not processed ({len(errors)})"):
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

if not results:
    st.error("No games could be evaluated. See the error details above.")
    st.stop()

summary_rows = []
for result in results:
    market = result["market"]
    variants = result["composed"]["predictions"]
    median = float(market["market_median_total"])
    summary_rows.append({
        "Game": f'{market["away_team"]} @ {market["home_team"]}',
        "Tip (Lagos)": pd.to_datetime(market["commence_time_utc"]).tz_convert("Africa/Lagos").strftime("%d %b %H:%M"),
        "Market total": median,
        "Books": market["bookmaker_count"],
        "V5 raw": variants["V5 raw"],
        "V5 + pace": variants["V5 + learned last-10 pace"],
        "V5 + legacy H2H": variants["V5 + legacy H2H 15%"],
        "V5 + legacy rest": variants["V5 + legacy rest 0.50"],
        "All legacy components": variants["V5 + pace + legacy H2H + legacy rest"],
        "Selected configuration": result["composed"]["selected_prediction"],
        "Selected − market": result["composed"]["selected_prediction"] - median,
    })

st.subheader("Model versus current market")
st.dataframe(
    pd.DataFrame(summary_rows).round(2),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "The selected-minus-market column is a model/market difference only. "
    "It is not a bet recommendation or a probability of an Over/Under result."
)

st.subheader("Game-level component diagnostics")
for result in results:
    market = result["market"]
    title = f'{market["away_team"]} @ {market["home_team"]} — market {market["market_median_total"]:.1f}'
    with st.expander(title, expanded=False):
        a, b, c, d = st.columns(4)
        a.metric("V5 raw total", f'{result["raw_total"]:.2f}')
        a.metric("Pace adjustment", f'{result["pace"].get("adjustment", 0.0):+.2f}')
        b.metric("Legacy H2H adjustment", f'{result["h2h"].get("legacy_adjustment", 0.0):+.2f}')
        b.metric("Learned H2H adjustment", f'{result["h2h"].get("learned_era_adjustment", 0.0):+.2f}')
        c.metric("Legacy rest adjustment", f'{result["rest_adjustments"]["legacy_fixed_050"]:+.2f}')
        c.metric("Schedule-density adjustment", f'{result["rest_adjustments"]["learned_schedule_density"]:+.2f}')
        d.metric("H2H meetings", result["h2h"].get("games", 0))
        d.metric("Selected projection", f'{result["composed"]["selected_prediction"]:.2f}')

        st.write(
            f'Home rest: {result["home_rest"]["rest_days"]} day(s); '
            f'Away rest: {result["away_rest"]["rest_days"]} day(s). '
            f'Last home game: {result["home_rest"]["last_game_date"]}; '
            f'last away game: {result["away_rest"]["last_game_date"]}.'
        )
        if result["pace"].get("available"):
            st.write(
                f'Pace gap versus league prior: {result["pace"]["pace_gap_l10"]:+.2f}; '
                f'expected matchup pace: {result["pace"]["expected_matchup_pace_l10"]:.2f}; '
                f'coefficient: {result["pace"]["beta"]:.4f}.'
            )
        else:
            st.warning(result["pace"].get("reason", "Pace feature unavailable."))

        quote_frame = pd.DataFrame(market["bookmaker_quotes"])
        if not quote_frame.empty:
            st.markdown("**Current book totals quotes**")
            st.dataframe(
                quote_frame.rename(columns={
                    "bookmaker": "Book",
                    "total_line": "Total",
                    "over_odds_decimal": "Over price (decimal)",
                    "under_odds_decimal": "Under price (decimal)",
                    "last_update_utc": "Book update (UTC)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Component switches used**")
        st.write({
            "pace": use_pace,
            "head_to_head": h2h_choice,
            "rest_schedule": rest_choice,
            "strict_prior_date_history": result["point_in_time_cutoff"],
        })

st.divider()
st.subheader("Paper observation ledger")
st.write(
    "Save the current market observation and all model variants with a timestamp. "
    "This creates a research record only; no wagers or stake amounts are recorded."
)

if st.button("Save current slate as paper observations", use_container_width=True):
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    observations = [
        make_paper_observation(result["market"], result, observed_at)
        for result in results
    ]
    for row in observations:
        row["selected_configuration"] = (
            f"pace={use_pace}; h2h={h2h_choice}; rest={rest_choice}"
        )
    saved = append_paper_observations(observations)
    st.success(f"Saved {saved.get('rows_added', 0)} paper observations to {saved.get('path')}.")

ledger_path = "outputs/live_market_testbench/paper_observations.csv"
if os.path.exists(ledger_path):
    ledger = pd.read_csv(ledger_path)
    st.caption(f"Local paper observations recorded: {len(ledger):,}")
    st.dataframe(ledger.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("No paper observations have been saved on this app instance yet.")

st.divider()
with st.expander("Research interpretation and limitations"):
    st.markdown(
        """
        - **Raw baseline:** V5 = 55% V3 + 45% V4-90.
        - **Pace:** possession-based last-10 pace, with the development-fitted coefficient 0.9673. The research holdout improved MAE but increased underprediction bias.
        - **H2H:** the legacy 15% adjustment and a smaller era-adjusted shrinkage alternative are exposed separately. Historical testing did not establish a dependable gain.
        - **Rest:** the legacy 0.50 rule, shared learned coefficient and schedule-density variant are exposed separately. Historical testing did not establish a dependable gain.
        - All history-derived inputs are filtered to dates strictly before each game.
        - Market lines and prices are provider observations at retrieval time; they can change before tip-off.
        - Free-host filesystem storage can be ephemeral. Export paper observations regularly; a durable database is a separate deployment decision.
        - Historical prediction accuracy is not betting profitability. This bench has not established ROI or closing-line value.
        """
    )
