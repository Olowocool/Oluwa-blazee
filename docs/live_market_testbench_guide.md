# NBA Totals Live-Market Test Bench

## Purpose and isolation

This is a separate Streamlit research app on branch research/live-market-testbench. It does not edit frontend.py, app.py, points_engine.py, or the live production prediction route.

It lets the user compare the validated V5 raw total with separately selectable pace, H2H and rest/schedule-density adjustments against current pre-game bookmaker totals. It only records paper observations. It has no bet-placement function, stake field, bankroll calculation or wager recommendation.

## Models exposed

### Baseline

V5 is the 55% V3 + 45% V4-90 point projection. Its raw total excludes pace, H2H and rest.

### Pace

The current research challenger uses:
- Team PBP possessions converted to 48-minute pace as PBP possessions multiplied by 48 and divided by elapsed game minutes.
- A prior-season fallback, with current-season form shrunk according to available games.
- The matchup pace gap measured against the point-in-time league pace.
- Fixed research coefficient: **0.9673 points per pace point**, fit on the 2021–2023 development seasons.

This option defaults on in the UI because pace was the strongest individual challenger in the historical tests. It remains a research feature because it increased negative prediction bias and was not production-approved.

### Head-to-head

The UI exposes:
- Off.
- Legacy: (last-10 H2H average total − V5 raw total) multiplied by 0.15.
- Learned era-adjusted: last-10 H2H totals moved into the target-date scoring environment, shrunk by n / (n + 5), then multiplied by the fixed development coefficient **0.036718**.

The H2H audit found negligible/uncertain gains, so neither rule is represented as a proven improvement.

### Rest and schedule density

The UI exposes:
- Off.
- Legacy combined rest: 0.50 multiplied by (home_rest_days + away_rest_days − 2).
- Learned shared coefficient: **0.416157** times the same combined rest delta.
- Learned schedule-density ridge: home/away rest deltas plus 3-in-4 and 4-in-6 game flags, using development-fit coefficients:
  - home rest delta: 0.642787
  - away rest delta: 0.073308
  - home 3-in-4: 0.315415
  - away 3-in-4: 0.462948
  - home 4-in-6: -0.065809
  - away 4-in-6: 0.432959

Rest and schedule-density changes did not establish reliable improvement on the later validation seasons.

## Market feed

The app calls The Odds API current NBA endpoint for the totals market, in decimal price format. It:
- Keeps only future, not-yet-started events.
- Requires a complete Over and Under pair from the same bookmaker and the same total line.
- Shows the median current total across books and a per-book quote table.
- Keeps Over and Under decimal prices paired with their bookmaker and line.
- Validates market update timestamps against the retrieval clock and tip-off.

The user can set ODDS_API_KEY in the app service environment. If it is not set, the sidebar offers a password field for the current session. The key is not printed, persisted to the repository or written into the paper ledger.

## Point-in-time data controls

- V5 receives the target game date and internally filters historical score data strictly before it.
- H2H history is filtered to meetings before the target date.
- Rest and schedule-density features use team games before the target date.
- Pace team histories and league priors are also filtered to dates strictly before the target date, even when the downloaded source contains later records.
- Market parsing excludes games that have already commenced and incomplete/mismatched totals quotes.

## Paper observations

The button **Save current slate as paper observations** records:
- Observation time, game/event and scheduled tip.
- Current median market total, bookmaker count and the per-book quote JSON.
- V5 raw, pace-adjusted, legacy H2H, legacy rest, all-legacy-component and selected combined projections.
- The selected component configuration and diagnostic adjustments.

It does not record or calculate a stake, bet, payout, win/loss or ROI. Game results and closing lines need a separate grading process before any profitability inference.

The default ledger path is outputs/live_market_testbench/paper_observations.csv. Render free-service filesystems can be ephemeral across deploys/restarts. Export the file regularly. A durable database is intentionally not auto-created by this test-bench deployment.

## Local run

    pip install -r requirements-testbench.txt
    streamlit run live_market_testbench.py --server.port 8501 --server.address 0.0.0.0

Set ODDS_API_KEY in the process environment or enter it in the sidebar.

## Render service

Create a separate Python web service linked to:
- Repository: https://github.com/Olowocool/Oluwa-blazee
- Branch: research/live-market-testbench
- Build command: pip install -r requirements-testbench.txt
- Start command: streamlit run live_market_testbench.py --server.port $PORT --server.address 0.0.0.0

Set ODDS_API_KEY as a secret environment variable in Render or enter it through the app's password field. Do not add the key to source control.

## Validation

GitHub Actions workflow: .github/workflows/live-market-testbench-validation.yml.

It compiles the app and V3/V4/V5 modules, tests market quote integrity, strict point-in-time H2H/rest inputs, adjustment arithmetic and paper-ledger writes, verifies the historical score file, smoke-tests a V5 future-date prediction, and confirms the Streamlit CLI is available.

No current market observations are committed to the repository by the CI workflow.
