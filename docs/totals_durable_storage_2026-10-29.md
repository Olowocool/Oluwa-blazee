# Durable storage for NBA totals research — 2026-10-29

## Decision

The local CSV implementation is useful for development but must not be treated as a persistent hosted market archive. This branch adds an opt-in PostgreSQL store for:
- timestamped live totals market observations (append-only, conflict-safe);
- saved totals picks and their settlement updates;
- prediction audit rows.

When PostgreSQL is configured, the database is the source of truth and the local CSV paths are mirrors used for existing UI/report compatibility. When it is not configured, the app keeps its local CSV development behavior unless durable storage is explicitly required. This opt-in mode must not be considered production durable until the hosted secret is set and a write/restart/read-back test passes.

## Configure the provider secret

Create a PostgreSQL database with network access from the Streamlit app. Keep its connection URL private. In the Streamlit app's **Secrets** settings (or the hosting environment), configure:

```toml
NBA_TOTALS_DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"
NBA_TOTALS_REQUIRE_DURABLE_STORAGE = "true"
```

Replace the placeholder URL with the actual provider-issued connection URL. Do not put real credentials in GitHub, this documentation, a committed `.streamlit/secrets.toml`, or logs. The configured database role must be allowed to create the `nba_totals_records` table and index on first connection, or an administrator must pre-create them.

`NBA_TOTALS_REQUIRE_DURABLE_STORAGE = "true"` makes missing connection configuration fail closed rather than silently falling back to ephemeral local CSV files. In that mode, saving picks, logging predictions, grading results, snapshot capture, and learning-data construction should report storage failures when the database is unavailable.

## Database design

The module creates `nba_totals_records` with:
- `record_type` and stable `record_key` as the primary key;
- `payload JSONB` containing the record fields;
- database timestamps for insert/update auditing.

Snapshot identity is the Eastern market date, teams, bookmaker, and UTC capture timestamp. Existing snapshot line, price, tip-off, provider-update, source, and event metadata are immutable: repeating an identical observation is idempotent, while a conflicting observation at the same identity fails rather than rewriting the historic market.

Saved picks use a stable `pick_id` so settlement updates update the same record. Prediction audits use `prediction_id`. Legacy local rows are deterministically identified during migration when they do not yet have IDs. The app's CSV files are hydrated from the database and are never treated as the durable source while PostgreSQL is configured.

## Verification checklist

1. Configure both secrets on the actual Streamlit app; do not share the URL in chat or logs.
2. Confirm the app reports PostgreSQL durable storage as configured.
3. Save one controlled paper pick with a valid live market snapshot; verify the row is present in the database and the CSV mirror.
4. Restart/redeploy the app, then verify that the pick and snapshot return from PostgreSQL and the CSV mirror is recreated.
5. Confirm the evaluator can match the pick's entry observation. Do not submit a real bet during this test.
6. Only after the controlled test passes should the service be treated as collecting durable prospective research data.

## Known limits

- This repository change cannot create a hosted PostgreSQL instance or configure the user's Streamlit secret; those are external account actions.
- PostgreSQL connection and schema initialization have unit/integration coverage in CI using a disposable database. A green CI run does not prove that the production URL, network allowlist, credentials, SSL options, or service dashboard are correct.
- The existing Render service mismatch remains unresolved. `frontend.py` calls `https://oluwa-blazee-new.onrender.com`, while the repository `render.yaml` says `fastapi-example`. Do not change or deploy Render settings until the actual Render dashboard's linked repository, branch, service, Blueprint association and auto-deploy configuration are verified.
- Probabilities remain disabled, experimental adjustments remain diagnostic, and live betting remains unauthorized by this research work.
