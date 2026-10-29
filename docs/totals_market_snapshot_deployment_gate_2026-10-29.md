# Deployment review gate for timestamped NBA totals data — 2026-10-29

## Decision

**Do not merge/deploy the timestamped market-ledger change as a durable paper-trading data collector until storage and service configuration are confirmed.** The code is passing CI, but code tests do not prove the hosted app's storage survives a restart or that the deployed service is the same service configured by the repository's Render manifest.

This review covers repository configuration only. It does not have access to the Streamlit Community Cloud dashboard, Render service dashboard, or hosted environment-variable/secrets settings.

## 1. Odds API secret

frontend.py reads ODDS_API_KEY from the process environment and then from st.secrets. The key must be configured in the deployed Streamlit app's environment/secrets settings. Never put the real key into Git, render.yaml, a committed .streamlit/secrets.toml, test fixtures, or logs.

Without a working key, the live snapshot path receives no current odds and cannot record new market snapshots.

## 2. Data durability — merge blocker for prospective collection

The new market ledger defaults to the local path data/live_totals_market_snapshots.csv. Saved picks and prediction logs also use local CSV paths (totals_history.csv and outputs/prediction_history.csv).

Local filesystem writes must not be treated as a durable historical archive on Streamlit Community Cloud. A restart or redeploy can discard runtime-created files; only files included in the repository are restored automatically. Do not rely on local runtime CSVs to survive deployments.

Before calling this a persistent forward paper-trading collector, move the ledger and saved-pick records to a durable external database/storage service (or use an explicitly configured persistent disk on a host that supports it). That storage needs to retain append-only snapshot history, allow conflict-safe inserts, and keep entry prices linked to the saved prediction. Do not commit live market-price history into public Git as a workaround.

## 3. Render configuration mismatch

The repository render.yaml currently declares service name fastapi-example, autoDeploy: true, and start command uvicorn app:app --host 0.0.0.0 --port $PORT. Meanwhile, frontend.py points its API client at https://oluwa-blazee-new.onrender.com.

The repository alone cannot prove whether that manifest manages the currently used Render service. Do not sync or rename the Render Blueprint based only on the URL: Render may create or attempt to manage a service based on the manifest. Confirm in the Render dashboard whether the live backend uses this repository, which branch it deploys, whether it auto-deploys on commits, and whether this YAML is the Blueprint currently attached to that service.

Render reference: https://render.com/docs/blueprint-spec and https://render.com/docs/deploys

## 4. Pre-merge deployment checklist

- Confirm the Streamlit app points to Olowocool/Oluwa-blazee, branch main, and entry point frontend.py.
- Confirm ODDS_API_KEY is present in the deployed app's secret/environment settings, without revealing the value.
- Choose and configure durable storage for data/live_totals_market_snapshots.csv, totals_history.csv, and the prediction audit log. Test that a write survives an app restart/redeploy.
- Confirm the exact Render service and linked branch, and whether a merge to main automatically deploys it. The repository's render.yaml is a template-style configuration and does not match the API hostname in frontend.py.
- After durable storage is addressed, run a controlled live-observation test; verify the ledger row, the saved pick's timestamps and line/prices, and a restart/reload. Do not make a bet from this test.
- Keep PR #23 and the evaluator PR #24 as drafts until the storage/deployment path is verified.

## What has been validated

The market capture, timestamp provenance, invalid-line guards, and regression tests passed the repository CI for PR #23. This proves the tested Python behavior; it does not verify provider credentials, real hosted network access, persistent storage, or the actual Render/Streamlit dashboard settings.
