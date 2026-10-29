# PostgreSQL persistence rollout gate

## Current state

The app's existing storage workflows write CSV files. This branch adds an isolated,
optional PostgreSQL adapter in `postgres_storage.py`, a PostgreSQL driver dependency,
and configuration tests. It does **not** switch the frontend to PostgreSQL yet.

## Configuration

- No `DATABASE_URL`: CSV mode remains the current behavior.
- `DATABASE_URL` configured: PostgreSQL adapter functions can connect using psycopg 3.
- Never commit the connection string or paste it into GitHub issues/chat. Configure it
  as a secret/environment variable in the hosting service.
- Provision a managed PostgreSQL database before enabling the adapter.

## Tables

- `nba_totals_market_snapshots`: append-only observations, timestamps, bookmaker,
  total line, both sides' prices, source event ID, source and raw audit payload.
- `nba_totals_picks`: prediction, contemporaneous market provenance, recommendation,
  stake, and settlement fields.

Snapshot observations are append-only; the schema does not overwrite older lines.
The schema's positive-line constraints reject zero/negative market totals.

## Required follow-up before production

1. Wire the adapter into the market snapshot writer and totals-pick tracker on a
   separate integration commit. Preserve the current CSV fallback when no database
   URL is configured.
2. Define a single authoritative read/write path for pick settlement. Do not let the
   dashboard read PostgreSQL while grading continues to update only CSV.
3. Add migration/version handling and database-backed integration tests against a
   disposable PostgreSQL instance.
4. Add an explicit, reviewed one-time CSV import with duplicate detection and a
   row-count reconciliation report. Do not silently import on every app startup.
5. Configure the database URL as a host secret, deploy to staging, then verify writes
   survive an app restart and that timestamp/line provenance remains valid.
6. Confirm the deployed Streamlit app and Render API are using the expected repository,
   branch and service before enabling the database path.

## Release rule

This branch is groundwork, not production-ready persistence. Keep the PR in draft until
integration, import/reconciliation, tests, and restart validation pass. No bets should
be automatically placed as part of this change.
