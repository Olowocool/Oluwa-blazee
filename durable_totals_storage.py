"""Opt-in PostgreSQL persistence for NBA totals research ledgers.

Set NBA_TOTALS_DATABASE_URL in the host's secret/environment settings to enable
PostgreSQL. Set NBA_TOTALS_REQUIRE_DURABLE_STORAGE=true to refuse silent CSV-only
fallback when the database URL is missing. CSVs remain working mirrors and
local-development fallbacks; they are not the durable source when PostgreSQL is
configured.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

TABLE_NAME = "nba_totals_records"
MARKET_SNAPSHOT_IMMUTABLE_FIELDS = (
    "total_line",
    "over_odds",
    "under_odds",
    "market_last_update_utc",
)


def _secret_value(key: str) -> str:
    value = os.getenv(key, "")
    if value:
        return str(value).strip()
    try:
        import streamlit as st
        value = st.secrets.get(key, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""


def get_database_url() -> str:
    """Read the PostgreSQL URL without ever logging or returning it to UI text."""
    return _secret_value("NBA_TOTALS_DATABASE_URL")


def durable_storage_required() -> bool:
    value = _secret_value("NBA_TOTALS_REQUIRE_DURABLE_STORAGE").casefold()
    return value in {"1", "true", "yes", "y", "required"}


def database_is_configured() -> bool:
    return bool(get_database_url())


def _url_or_fail() -> str | None:
    url = get_database_url()
    if url:
        return url
    if durable_storage_required():
        raise RuntimeError(
            "Durable NBA totals storage is required, but NBA_TOTALS_DATABASE_URL "
            "is not configured. Add it to the host's secret/environment settings."
        )
    return None


def _connect():
    url = _url_or_fail()
    if not url:
        return None
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL storage is configured but psycopg is not installed. "
            "Install the repository requirements."
        ) from exc
    # Do not include the DSN in any exception or log message.
    try:
        return psycopg.connect(url, connect_timeout=5)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to the configured NBA totals database. "
            "The durable write was not completed; CSV-only fallback was refused."
        ) from exc


def _ensure_schema(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            record_type TEXT NOT NULL,
            record_key TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (record_type, record_key)
        )
        """
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_type_created "
        f"ON {TABLE_NAME} (record_type, created_at_utc)"
    )


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    # Handle NumPy/Pandas scalar values without requiring those packages here.
    if hasattr(value, "item"):
        try:
            return _safe_json_value(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    text = str(value)
    return None if text in {"<NA>", "NaT", "nan", "None"} else text


def _canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(
        _safe_json_value(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_key(record_type: str, record: dict[str, Any]) -> str:
    """Return a stable key for each append-only observation or mutable audit row."""
    if record_type == "market_snapshot":
        teams = sorted([
            str(record.get("home_team", "")).strip().casefold(),
            str(record.get("away_team", "")).strip().casefold(),
        ])
        identity = [
            str(record.get("game_date", "")).strip(),
            "|".join(teams),
            str(record.get("bookmaker", "")).strip().casefold(),
            str(record.get("snapshot_timestamp_utc", "")).strip(),
        ]
        return _stable_hash("\x1f".join(identity))

    id_field = {
        "totals_pick": "pick_id",
        "prediction_audit": "prediction_id",
    }.get(record_type)
    if id_field and str(record.get(id_field) or "").strip():
        return str(record[id_field]).strip()

    # Legacy CSV rows receive a deterministic ID, which is also written back to
    # the mirror so settlement updates retain the same identity across restarts.
    return "legacy-" + _stable_hash(_canonical_json(record))


def _ensure_legacy_ids(record_type: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if record_type not in {"totals_pick", "prediction_audit"}:
        return rows
    field = "pick_id" if record_type == "totals_pick" else "prediction_id"
    prepared = []
    for original in rows:
        row = dict(original)
        if not str(row.get(field) or "").strip():
            row[field] = "legacy-" + _stable_hash(_canonical_json(row))
        prepared.append(row)
    return prepared


def _payload_from_database(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Database record payload is not a JSON object.")


def _values_equal(left: Any, right: Any) -> bool:
    left_clean = "" if left is None else str(left).strip()
    right_clean = "" if right is None else str(right).strip()
    try:
        a, b = float(left_clean), float(right_clean)
        if math.isfinite(a) and math.isfinite(b):
            return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError, OverflowError):
        pass
    return left_clean == right_clean


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(
    path: str | Path,
    records: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
    prior_fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    if not records and not fieldnames and not prior_fieldnames:
        return
    columns: list[str] = []
    for name in (prior_fieldnames or []) + (fieldnames or []):
        if name not in columns:
            columns.append(name)
    for record in records:
        for name in record:
            if name not in columns:
                columns.append(name)
    if not columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            tmp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def fetch_records(record_type: str) -> list[dict[str, Any]]:
    """Fetch records in insertion order; returns [] only when storage isn't configured."""
    url = _url_or_fail()
    if not url:
        return []
    connection = _connect()
    assert connection is not None
    with connection:
        _ensure_schema(connection)
        rows = connection.execute(
            f"SELECT payload::text FROM {TABLE_NAME} "
            "WHERE record_type = %s ORDER BY created_at_utc, record_key",
            (record_type,),
        ).fetchall()
    return [_payload_from_database(row[0]) for row in rows]


def persist_records(
    record_type: str,
    records: Iterable[dict[str, Any]],
    *,
    immutable: bool = False,
    conflict_fields: tuple[str, ...] = (),
) -> dict[str, int]:
    """Insert/update records transactionally. Immutable conflicts abort the batch."""
    url = _url_or_fail()
    if not url:
        return {"inserted": 0, "updated": 0, "unchanged": 0}
    prepared = _ensure_legacy_ids(
        record_type,
        [dict(record) for record in records],
    )
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    connection = _connect()
    assert connection is not None
    with connection:
        _ensure_schema(connection)
        for record in prepared:
            key = record_key(record_type, record)
            payload = _canonical_json(record)
            existing_row = connection.execute(
                f"SELECT payload::text FROM {TABLE_NAME} "
                "WHERE record_type = %s AND record_key = %s FOR UPDATE",
                (record_type, key),
            ).fetchone()
            if existing_row:
                existing = _payload_from_database(existing_row[0])
                if immutable:
                    conflicts = [
                        field for field in conflict_fields
                        if not _values_equal(existing.get(field), record.get(field))
                    ]
                    if conflicts:
                        raise ValueError(
                            "Conflicting immutable market observation for the same "
                            f"record identity; fields changed: {conflicts}. Existing database data was not overwritten."
                        )
                    counts["unchanged"] += 1
                    continue
                connection.execute(
                    f"UPDATE {TABLE_NAME} SET payload = %s::jsonb, updated_at_utc = NOW() "
                    "WHERE record_type = %s AND record_key = %s",
                    (payload, record_type, key),
                )
                counts["updated"] += 1
            else:
                connection.execute(
                    f"INSERT INTO {TABLE_NAME} (record_type, record_key, payload) "
                    "VALUES (%s, %s, %s::jsonb)",
                    (record_type, key, payload),
                )
                counts["inserted"] += 1
    return counts


def sync_csv_mirror(
    record_type: str,
    path: str | Path,
    *,
    fieldnames: list[str] | None = None,
) -> bool:
    """Hydrate the local CSV from PostgreSQL, importing local history only as needed.

    With a database URL configured, the database is the durable source of truth.
    Missing local rows are imported to support first-time migration, but existing
    database rows are never overwritten by a stale CSV mirror.
    """
    url = _url_or_fail()
    if not url:
        return False

    prior_fieldnames, local_rows = _read_csv(path)
    local_rows = _ensure_legacy_ids(record_type, local_rows)
    db_rows = fetch_records(record_type)

    if not db_rows and local_rows:
        persist_records(
            record_type,
            local_rows,
            immutable=(record_type == "market_snapshot"),
            conflict_fields=MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
        )
        db_rows = fetch_records(record_type)
    elif db_rows and local_rows:
        db_keys = {record_key(record_type, row) for row in db_rows}
        missing_local = [
            row for row in local_rows
            if record_key(record_type, row) not in db_keys
        ]
        if missing_local:
            persist_records(
                record_type,
                missing_local,
                immutable=(record_type == "market_snapshot"),
                conflict_fields=MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
            )
        if record_type == "market_snapshot":
            # Even an already-present identity must have identical line/prices.
            persist_records(
                record_type,
                local_rows,
                immutable=True,
                conflict_fields=MARKET_SNAPSHOT_IMMUTABLE_FIELDS,
            )
        db_rows = fetch_records(record_type)

    _write_csv(path, db_rows, fieldnames=fieldnames, prior_fieldnames=prior_fieldnames)
    return True


def storage_status() -> dict[str, Any]:
    """Return a safe status summary without exposing connection details."""
    configured = database_is_configured()
    required = durable_storage_required()
    return {
        "database_configured": configured,
        "durable_storage_required": required,
        "mode": "postgresql" if configured else "required-but-missing" if required else "local-csv",
        "durable": configured,
    }
