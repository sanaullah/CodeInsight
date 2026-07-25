"""Model-call ledger operations over the canonical SQLite schema."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from .run_ledger import SqliteRunLedger


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteModelCallRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger

    def start(
        self,
        *,
        model_call_id: str,
        run_id: str,
        wave_id: str,
        task_id: str,
        attempt_id: str,
        provider: str,
        model: str,
        request_hash: str,
        correlation: dict[str, str],
    ) -> None:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO model_calls(
                    model_call_id, run_id, wave_id, task_id, attempt_id, provider, model,
                    request_hash, status, usage_json, trace_correlation_json,
                    started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', '{}', ?, ?)
                ON CONFLICT(task_id, request_hash, provider, model) DO UPDATE SET
                    attempt_id = excluded.attempt_id, status = 'running',
                    started_at = excluded.started_at,
                    completed_at = NULL
                """,
                (
                    model_call_id,
                    run_id,
                    wave_id,
                    task_id,
                    attempt_id,
                    provider,
                    model,
                    request_hash,
                    json.dumps(correlation, sort_keys=True, separators=(",", ":")),
                    timestamp,
                ),
            )

        self.ledger.write_transaction(operation)

    def finish(
        self,
        model_call_id: str,
        *,
        status: str,
        usage: dict[str, int | float] | None = None,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                UPDATE model_calls
                SET status = ?, usage_json = ?, completed_at = ?
                WHERE model_call_id = ?
                """,
                (
                    status,
                    json.dumps(usage or {}, sort_keys=True, separators=(",", ":")),
                    _now(),
                    model_call_id,
                ),
            )

        self.ledger.write_transaction(operation)
