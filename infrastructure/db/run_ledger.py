"""Data operations for durable run lifecycle state.

This module deliberately contains no DDL or pragma configuration. The canonical
database module owns schema and connection behavior.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from domain.contracts import RunStage

from .database import database_connection, initialize_database, open_database

_T = TypeVar("_T")
_ACTIVE_STATUSES = ("queued", "running")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteRunLedger:
    """Authoritative run and event repository for one application database."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        event_history_limit: int = 200,
        busy_retry_attempts: int = 5,
    ) -> None:
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be greater than zero")
        if busy_retry_attempts < 1:
            raise ValueError("busy_retry_attempts must be greater than zero")
        self.database_path = Path(database_path).expanduser().resolve()
        self.event_history_limit = event_history_limit
        self.busy_retry_attempts = busy_retry_attempts
        initialize_database(self.database_path)
        self._writer_lock = threading.RLock()
        self._writer = open_database(
            self.database_path, allow_cross_thread=True
        )
        self._finalizer = weakref.finalize(self, self._writer.close)
        self._closed = False

    def close(self) -> None:
        with self._writer_lock:
            if not self._closed:
                self._finalizer()
                self._closed = True

    def __enter__(self) -> SqliteRunLedger:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _write(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        with self._writer_lock:
            if self._closed:
                raise RuntimeError("run ledger is closed")
            for attempt in range(self.busy_retry_attempts):
                try:
                    self._writer.execute("BEGIN IMMEDIATE")
                    try:
                        result = operation(self._writer)
                        self._writer.commit()
                        return result
                    except Exception:
                        self._writer.rollback()
                        raise
                except sqlite3.OperationalError as exc:
                    if (
                        "locked" not in str(exc).lower()
                        or attempt + 1 >= self.busy_retry_attempts
                    ):
                        raise
                    time.sleep(0.005 * (2**attempt))
        raise RuntimeError("unreachable database retry state")

    def write_transaction(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a repository operation through the ledger's serialized writer."""

        return self._write(operation)

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        created_at: str | None = None,
    ) -> None:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence "
            "FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO events(run_id, sequence, event_type, data_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(row["sequence"]),
                event_type,
                json.dumps(data, sort_keys=True, separators=(",", ":")),
                created_at or _utc_now(),
            ),
        )

    def create_run(
        self,
        *,
        run_id: str,
        submission_key: str,
        request: dict[str, Any],
        mode: str = "deep",
        budget: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        timestamp = created_at or _utc_now()

        def operation(connection: sqlite3.Connection) -> str:
            existing = connection.execute(
                "SELECT run_id FROM runs WHERE submission_key = ?",
                (submission_key,),
            ).fetchone()
            if existing is not None:
                return str(existing["run_id"])
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, submission_key, status, mode, request_json,
                    budget_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    submission_key,
                    mode,
                    json.dumps(request, sort_keys=True, separators=(",", ":")),
                    json.dumps(budget or {}, sort_keys=True, separators=(",", ":")),
                    timestamp,
                    timestamp,
                ),
            )
            self._append_event(
                connection, run_id, "analysis_queued", {}, created_at=timestamp
            )
            return run_id

        persisted_run_id = self._write(operation)
        record = self.get_run(persisted_run_id)
        if record is None:
            raise RuntimeError("persisted run could not be read")
        return record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            return self._record(connection, row)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            if not rows:
                return []
            placeholders = ",".join("?" for _ in rows)
            event_rows = connection.execute(
                f"""
                WITH ranked AS (
                    SELECT sequence, event_type, data_json, created_at, run_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY run_id ORDER BY sequence DESC
                           ) AS rank
                    FROM events
                    WHERE run_id IN ({placeholders})
                )
                SELECT sequence, event_type, data_json, created_at, run_id
                FROM ranked
                WHERE rank <= ?
                ORDER BY run_id, sequence
                """,
                (*[row["run_id"] for row in rows], self.event_history_limit),
            ).fetchall()
            events_by_run: dict[str, list[dict[str, Any]]] = {
                str(row["run_id"]): [] for row in rows
            }
            for event in event_rows:
                events_by_run[str(event["run_id"])].append(self._event_record(event))
            return [
                self._record_from_row(row, events_by_run[str(row["run_id"])])
                for row in rows
            ]

    def list_queued_run_ids(self) -> list[str]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
            return [str(row["run_id"]) for row in rows]

    def count_active(self) -> int:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM runs WHERE status IN (?, ?)",
                _ACTIVE_STATUSES,
            ).fetchone()
            return int(row["count"])

    def mark_running(self, run_id: str) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'running',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE run_id = ? AND status = 'queued'
                """,
                (timestamp, timestamp, run_id),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection, run_id, "analysis_started", {}, created_at=timestamp
            )
            return True

        return self._write(operation)

    def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            active = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND status IN (?, ?)",
                (run_id, *_ACTIVE_STATUSES),
            ).fetchone()
            if active is None:
                return
            self._append_event(connection, run_id, event_type, data)

        self._write(operation)

    def set_stage(self, run_id: str, stage: RunStage) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs SET current_stage = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (stage.value, timestamp, run_id),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection,
                run_id,
                "stage_changed",
                {"stage": stage.value},
                created_at=timestamp,
            )
            return True

        return self._write(operation)

    def succeed(self, run_id: str, result: dict[str, Any]) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'succeeded', result_json = ?, completed_at = ?,
                    updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (
                    json.dumps(result, sort_keys=True, separators=(",", ":")),
                    timestamp,
                    timestamp,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection, run_id, "analysis_completed", {}, created_at=timestamp
            )
            return True

        return self._write(operation)

    def needs_attention(self, run_id: str, result: dict[str, Any]) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'needs_attention', result_json = ?,
                    completed_at = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (
                    json.dumps(result, sort_keys=True, separators=(",", ":")),
                    timestamp,
                    timestamp,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection,
                run_id,
                "analysis_needs_attention",
                {"failed_task_count": result.get("failed_task_count", 0)},
                created_at=timestamp,
            )
            return True

        return self._write(operation)

    def fail(self, run_id: str, error: str) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'failed', error = ?, completed_at = ?, updated_at = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (error, timestamp, timestamp, run_id, *_ACTIVE_STATUSES),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection,
                run_id,
                "analysis_failed",
                {"error": error},
                created_at=timestamp,
            )
            return True

        return self._write(operation)

    def cancel(self, run_id: str) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'cancelled', cancellation_requested = 1,
                    completed_at = ?, updated_at = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (timestamp, timestamp, run_id, *_ACTIVE_STATUSES),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection, run_id, "analysis_cancelled", {}, created_at=timestamp
            )
            return True

        return self._write(operation)

    def recover_interrupted(self) -> int:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> int:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE status = 'running' "
                "ORDER BY created_at"
            ).fetchall()
            for row in rows:
                run_id = str(row["run_id"])
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'queued', started_at = NULL, updated_at = ?
                    WHERE run_id = ? AND status = 'running'
                    """,
                    (timestamp, run_id),
                )
                self._append_event(
                    connection,
                    run_id,
                    "analysis_recovered",
                    {"previous_status": "running"},
                    created_at=timestamp,
                )
            return len(rows)

        return self._write(operation)

    def requeue_interrupted(self, run_id: str) -> bool:
        timestamp = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'queued', started_at = NULL, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (timestamp, run_id),
            )
            if cursor.rowcount != 1:
                return False
            self._append_event(
                connection,
                run_id,
                "analysis_interrupted",
                {"reason": "service_shutdown"},
                created_at=timestamp,
            )
            return True

        return self._write(operation)

    def _record(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        event_rows = connection.execute(
            """
            SELECT sequence, event_type, data_json, created_at
            FROM events
            WHERE run_id = ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (row["run_id"], self.event_history_limit),
        ).fetchall()
        events = [self._event_record(event) for event in reversed(event_rows)]
        return self._record_from_row(row, events)

    @staticmethod
    def _event_record(event: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": int(event["sequence"]),
            "event_type": str(event["event_type"]),
            "timestamp": str(event["created_at"]),
            "data": json.loads(event["data_json"]),
        }

    @staticmethod
    def _record_from_row(
        row: sqlite3.Row, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "status": str(row["status"]),
            "request": json.loads(row["request_json"]),
            "current_stage": row["current_stage"],
            "snapshot_id": row["snapshot_id"],
            "mode": str(row["mode"]),
            "created_at": str(row["created_at"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "events": events,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
        }
