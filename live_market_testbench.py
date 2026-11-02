from __future__ import annotations

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
from live_market_testbench_engine import calculate_game_components, load_score_history, load_team_id_map

st.set_page_config(page_title="NBA Totals Research Test Bench", page_icon="🏀", layout="wide")
st.title("NBA Totals — Integrated Research Test Bench")
st.caption("Live market observation + frozen V5 baseline + Objectives 5, 6 and 7. Research only.")
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

with st.sidebar:
    st.subheader("Live market capture")
    st.write("Source: DunkAPI public NBA odds board. This does not use paid DunkAPI API credits.")
    capture = st.button("Capture current market snapshot", type="primary", use_container_width=True)
    st.caption("Capture repeatedly before tip-off. Objective 5 selects only valid DraftKings observations at least 120 minutes before tip.")
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
        response = requests.get(PAGE_URL, headers={"User-Agent": "OluwaBlazee-NBA-Research-TestBench/1.0"}, timeout=30)
        response.raise_for_status()
        now = datetime.now(timezone.utc)
        rows, meta = parse_odds_page(response.text, retrieved_at=now)
        existing = {(r.get("game_key"), r.get("bookmaker"), r.get("observed_at_utc")) for r in st.session_state.captured_market_rows}
        added = [r for r in rows if (r.get("game_key"), r.get("bookmaker"), r.get("observed_at_utc")) not in existing]
        st.session_state.captured_market_rows.extend(added)
        st.session_state.last_capture_meta = meta
        st.success(f"Captured {len(added)} bookmaker observations across {meta.get('games_seen', 0)} listed games.")
    except Exception as exc:
        st.error(f"Market capture failed: {type(exc).__name__}: {exc}")

tab_live, tab_validation, tab_guide = st.tabs(["Live market test", "Objectives 5–7 validation", "How to use this bench"])

with tab_live:
    st.subheader("Latest observed market")
    captured = pd.DataFrame(st.session_state.captured_market_rows)
    if captured.empty:
        st.info("No observations captured in this session yet. Use “Capture current market snapshot” in the sidebar.")
    else:
        captured["observed_at_utc"] = pd.to_datetime(captured["observed_at_utc"], utc=True, errors="coerce")
        latest = captured.sort_values("observed_at_utc").groupby(["game_key", "bookmaker"], as_index=False).tail(1)
        display = latest[["game_date", "commence_time_utc", "away_team", "home_team", "bookmaker", "total_line", "over_odds_american", "under_odds_american", "observed_at_utc"]].copy()
        st.dataframe(display.sort_values(["game_date", "commence_time_utc", "bookmaker"]), use_container_width=True, hide_index=True)
        if st.session_state.last_capture_meta:
            st.caption(
                f"Published as of {st.session_state.last_capture_meta.get('published_as_of_utc')} · "
                f"Observed {st.session_state.last_capture_meta.get('observed_at_utc')} · "
                f"Page age {st.session_state.last_capture_meta.get('page_age_minutes')} minutes"
            )
        st.download_button(
            "Download captured market archive (CSV)",
            data=captured.to_csv(index=False).encode("utf-8"),
            file_name="dunkapi_public_odds_snapshots.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("V5 live matchup projections")
    score_upload_live = st.file_uploader("Upload historical NBA scores CSV to calculate V5 projections", type=["csv"], key="score_live")
    if score_upload_live is not None and not captured.empty:
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                handle.write(score_upload_live.getvalue())
                score_path = handle.name
            history = load_score_history(score_path)
            team_map = load_team_id_map()
            latest_rows = captured.sort_values("observed_at_utc").groupby(["game_key", "bookmaker"], as_index=False).tail(1)
            projections = []
            cache = {"v3": {}, "v4": {}}
            for game_key, group in latest_rows.groupby("game_key", sort=False):
                first = group.iloc[0]
                quotes = []
                for _, q in group.iterrows():
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
                target_date = commence.tz_convert("Africa/Lagos").date().isoformat()
                prediction = calculate_game_components(
                    str(first["home_team"]), str(first["away_team"]), target_date,
                    history, None, team_map, use_pace=False, h2h_mode="off",
                    rest_mode="off", v5_cache=cache,
                )
                market_median = float(np.median([q["total_line"] for q in quotes]))
                projections.append({
                    "Game": f'{first["away_team"]} @ {first["home_team"]}',
                    "Tip (Lagos)": commence.tz_convert("Africa/Lagos").strftime("%d %b %H:%M"),
                    "Market median": round(market_median, 1),
                    "Books": len(quotes),
                    "V5 raw total": round(float(prediction["raw_total"]), 2),
                    "V5 − market": round(float(prediction["raw_total"] - market_median), 2),
                })
            st.dataframe(pd.DataFrame(projections), use_container_width=True, hide_index=True)
            st.caption("V5 is the frozen raw baseline. Pace, H2H and rest adjustments are deliberately off in this comparison.")
        except Exception as exc:
            st.error(f"V5 projection could not be calculated: {type(exc).__name__}: {exc}")
    elif score_upload_live is None:
        st.caption("Upload the score history file to compare frozen V5 projections with the captured live market.")

with tab_validation:
    st.subheader("Run the complete historical validation")
    st.write("Upload the historical score file and the saved market-snapshot archive. The archive can be downloaded from the Live market tab and combined across sessions.")
    score_upload = st.file_uploader("Historical NBA scores CSV", type=["csv"], key="score_validation")
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

    run_validation = st.button(
        "Run Objectives 5, 6 and 7",
        type="primary",
        disabled=(score_upload is None or archive.empty),
        use_container_width=True,
    )
    if run_validation:
        with st.spinner("Running chronological market, probability, bet-selection and profitability checks…"):
            try:
                with tempfile.TemporaryDirectory(prefix="nba_totals_testbench_") as temp:
                    root = Path(temp)
                    score_path = root / "historical_nba_scores.csv"
                    snapshot_path = root / "market_snapshots.csv"
                    score_path.write_bytes(score_upload.getvalue())
                    archive.to_csv(snapshot_path, index=False)
                    empty_patch_dir = root / "empty_score_patches"
                    empty_patch_dir.mkdir()
                    out5, out6, out7 = root / "objective5", root / "objective6", root / "objective7"
                    summary5 = run_objective5(Namespace(
                        snapshots_file=str(snapshot_path),
                        scores_file=str(score_path),
                        score_patch_dir=str(empty_patch_dir),
                        score_patch_files=[],
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

with tab_guide:
    st.subheader("Operating instructions")
    st.markdown(
        """
        1. Open Live market test and capture the public odds board periodically. Capture at least one observation no later than 120 minutes before each game, then capture again later before tip.
        2. Download the market archive after each session. Keep your own copy because the free web service may restart and clear session data.
        3. Upload your historical NBA score CSV and the combined market archive under Objectives 5–7 validation.
        4. Run the three objectives together. Objective 5 enforces DraftKings, T−120, the 240-minute maximum entry age, 150 prior training games and 100 prior out-of-sample calibration errors.
        5. Objective 6 only produces candidate decisions if the development evidence gate passes. Otherwise it returns No Bet.
        6. Objective 7 settles only frozen holdout decisions. It requires at least 100 settled bets across 30 distinct dates before any positive-profitability evidence can be considered.
        """
    )
    st.warning(
        "This app uses the public DunkAPI odds-board page and does not place bets. "
        "Snapshots captured after the T−120 cutoff cannot be treated as entry prices. "
        "Do not interpret model-minus-market differences or small samples as an edge."
    )
