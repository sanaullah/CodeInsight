"""Durable task queue operations for the native workflow scheduler.

Schema creation and connection pragmas intentionally remain in
``infrastructure.db.database``. This repository only performs data operations.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from domain.contracts import AnalysisTask, RoleSpec

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class TaskLease:
    task_id: str
    run_id: str
    wave_id: str
    role_id: str | None
    task_type: str
    attempt_id: str
    attempt_number: int
    input: dict[str, Any]
    budget: dict[str, Any]
    routing_policy: dict[str, Any]
    lease_owner: str
    lease_expires_at: datetime


class SqliteTaskRepository:
    """Atomic task leasing and completion against the application ledger."""

    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def create_wave(
        self,
        *,
        run_id: str,
        wave_id: str,
        wave_number: int,
        rationale: str,
        budget: dict[str, Any],
    ) -> None:
        timestamp = _timestamp(_utc_now())

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO waves(
                    wave_id, run_id, wave_number, rationale, status,
                    budget_json, created_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                ON CONFLICT(run_id, wave_number) DO NOTHING
                """,
                (
                    wave_id,
                    run_id,
                    wave_number,
                    rationale,
                    json.dumps(budget, sort_keys=True, separators=(",", ":")),
                    timestamp,
                ),
            )

        self.ledger.write_transaction(operation)

    def enqueue(self, task: AnalysisTask) -> bool:
        """Persist a task once, keyed by its stable idempotency key."""

        timestamp = _timestamp(_utc_now())
        input_data = {
            "immutable_input_ids": list(task.immutable_input_ids),
            "configuration_hash": task.configuration_hash,
        }
        budget = {
            "token_budget": task.token_budget,
            "cost_budget_usd": task.cost_budget_usd,
            "time_budget_seconds": task.time_budget_seconds,
            "tool_call_budget": task.tool_call_budget,
        }
        routing = {"model_policy": task.model_policy}

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                INSERT INTO tasks(
                    task_id, run_id, wave_id, role_id, task_type, priority,
                    status, idempotency_key, input_json, budget_json,
                    routing_policy_json, attempt_count, max_attempts,
                    cancellation_requested, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, 0, ?, 0, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    task.task_id,
                    task.run_id,
                    task.wave_id,
                    task.role_id,
                    task.task_type,
                    task.priority,
                    task.idempotency_key,
                    json.dumps(input_data, sort_keys=True, separators=(",", ":")),
                    json.dumps(budget, sort_keys=True, separators=(",", ":")),
                    json.dumps(routing, sort_keys=True, separators=(",", ":")),
                    task.max_attempts,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)

    def add_role(self, *, run_id: str, wave_id: str, role: RoleSpec) -> bool:
        timestamp = _timestamp(_utc_now())

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                INSERT INTO role_specs(
                    role_id, run_id, wave_id, contract_json,
                    contract_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(role_id) DO NOTHING
                """,
                (
                    role.role_id,
                    run_id,
                    wave_id,
                    role.model_dump_json(),
                    role.schema_version,
                    timestamp,
                ),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)

    def lease(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: float,
        run_id: str | None = None,
        now: datetime | None = None,
    ) -> list[TaskLease]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        leased_at = now or _utc_now()
        leased_at_text = _timestamp(leased_at)
        expires_at = leased_at + timedelta(seconds=lease_seconds)
        expires_at_text = _timestamp(expires_at)

        def operation(connection: sqlite3.Connection) -> list[TaskLease]:
            parameters: list[Any] = [leased_at_text]
            run_filter = ""
            if run_id is not None:
                run_filter = "AND tasks.run_id = ?"
                parameters.append(run_id)
            parameters.append(limit)
            rows = connection.execute(
                f"""
                SELECT tasks.*
                FROM tasks
                JOIN runs ON runs.run_id = tasks.run_id
                WHERE tasks.status IN ('queued', 'retry_wait')
                  AND tasks.available_at <= ?
                  AND tasks.cancellation_requested = 0
                  AND runs.status = 'running'
                  AND runs.cancellation_requested = 0
                  {run_filter}
                ORDER BY tasks.priority, tasks.created_at, tasks.task_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            leases: list[TaskLease] = []
            for row in rows:
                attempt_number = int(row["attempt_count"]) + 1
                cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'leased', attempt_count = ?,
                        lease_owner = ?, lease_expires_at = ?, updated_at = ?
                    WHERE task_id = ?
                      AND status IN ('queued', 'retry_wait')
                      AND cancellation_requested = 0
                    """,
                    (
                        attempt_number,
                        worker_id,
                        expires_at_text,
                        leased_at_text,
                        row["task_id"],
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                attempt_id = uuid4().hex
                connection.execute(
                    """
                    INSERT INTO task_attempts(
                        attempt_id, task_id, attempt_number, worker_id,
                        status, started_at
                    ) VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (
                        attempt_id,
                        row["task_id"],
                        attempt_number,
                        worker_id,
                        leased_at_text,
                    ),
                )
                leases.append(
                    TaskLease(
                        task_id=str(row["task_id"]),
                        run_id=str(row["run_id"]),
                        wave_id=str(row["wave_id"]),
                        role_id=str(row["role_id"]) if row["role_id"] else None,
                        task_type=str(row["task_type"]),
                        attempt_id=attempt_id,
                        attempt_number=attempt_number,
                        input=json.loads(row["input_json"]),
                        budget=json.loads(row["budget_json"]),
                        routing_policy=json.loads(row["routing_policy_json"]),
                        lease_owner=worker_id,
                        lease_expires_at=expires_at,
                    )
                )
            return leases

        return self.ledger.write_transaction(operation)

    def succeed(
        self,
        lease: TaskLease,
        *,
        output_artifact_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> bool:
        timestamp = _timestamp(_utc_now())

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'succeeded', output_artifact_id = ?,
                    completed_at = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (
                    output_artifact_id,
                    timestamp,
                    timestamp,
                    lease.task_id,
                    lease.lease_owner,
                ),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                UPDATE task_attempts
                SET status = 'succeeded', completed_at = ?, usage_json = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (
                    timestamp,
                    json.dumps(usage or {}, sort_keys=True, separators=(",", ":")),
                    lease.attempt_id,
                ),
            )
            return True

        return self.ledger.write_transaction(operation)

    def fail(
        self,
        lease: TaskLease,
        *,
        error: str,
        retry_delay_seconds: float = 0,
        retryable: bool = True,
    ) -> str:
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        now = _utc_now()
        timestamp = _timestamp(now)
        available_at = _timestamp(now + timedelta(seconds=retry_delay_seconds))
        error_json = json.dumps({"message": error}, separators=(",", ":"))

        def operation(connection: sqlite3.Connection) -> str:
            row = connection.execute(
                """
                SELECT attempt_count, max_attempts, cancellation_requested
                FROM tasks
                WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (lease.task_id, lease.lease_owner),
            ).fetchone()
            if row is None:
                return "stale"
            retry = (
                retryable
                and
                not bool(row["cancellation_requested"])
                and int(row["attempt_count"]) < int(row["max_attempts"])
            )
            status = "retry_wait" if retry else "failed"
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, available_at = ?, updated_at = ?,
                    completed_at = ?, error_json = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (
                    status,
                    available_at,
                    timestamp,
                    None if retry else timestamp,
                    error_json,
                    lease.task_id,
                    lease.lease_owner,
                ),
            )
            connection.execute(
                """
                UPDATE task_attempts
                SET status = 'failed', completed_at = ?, error_json = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (timestamp, error_json, lease.attempt_id),
            )
            return status

        return self.ledger.write_transaction(operation)

    def cancel_task(self, lease: TaskLease, *, reason: str) -> bool:
        timestamp = _timestamp(_utc_now())
        error_json = json.dumps({"message": reason}, separators=(",", ":"))

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'cancelled', cancellation_requested = 1,
                    completed_at = ?, updated_at = ?, error_json = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (
                    timestamp,
                    timestamp,
                    error_json,
                    lease.task_id,
                    lease.lease_owner,
                ),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                UPDATE task_attempts
                SET status = 'cancelled', completed_at = ?, error_json = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (timestamp, error_json, lease.attempt_id),
            )
            return True

        return self.ledger.write_transaction(operation)

    def interrupt_task(self, lease: TaskLease, *, reason: str) -> bool:
        """Release a controlled-shutdown lease so another worker can resume it."""

        timestamp = _timestamp(_utc_now())
        error_json = json.dumps({"message": reason}, separators=(",", ":"))

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'retry_wait', available_at = ?, updated_at = ?,
                    error_json = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
                  AND cancellation_requested = 0
                """,
                (
                    timestamp,
                    timestamp,
                    error_json,
                    lease.task_id,
                    lease.lease_owner,
                ),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                UPDATE task_attempts
                SET status = 'interrupted', completed_at = ?, error_json = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (timestamp, error_json, lease.attempt_id),
            )
            return True

        return self.ledger.write_transaction(operation)

    def request_run_cancellation(self, run_id: str) -> int:
        timestamp = _timestamp(_utc_now())
        error_json = json.dumps(
            {"message": "run cancellation requested"}, separators=(",", ":")
        )

        def operation(connection: sqlite3.Connection) -> int:
            connection.execute(
                "UPDATE runs SET cancellation_requested = 1, updated_at = ? "
                "WHERE run_id = ?",
                (timestamp, run_id),
            )
            return self.ledger.terminalize_active_tasks(
                connection,
                run_id=run_id,
                timestamp=timestamp,
                error_json=error_json,
            )

        return self.ledger.write_transaction(operation)

    def is_cancellation_requested(self, task_id: str) -> bool:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT tasks.cancellation_requested AS task_cancel,
                       runs.cancellation_requested AS run_cancel
                FROM tasks JOIN runs ON runs.run_id = tasks.run_id
                WHERE tasks.task_id = ?
                """,
                (task_id,),
            ).fetchone()
            return row is None or bool(row["task_cancel"]) or bool(row["run_cancel"])

    def recover_expired(self, *, now: datetime | None = None) -> int:
        timestamp = _timestamp(now or _utc_now())
        error_json = json.dumps({"message": "task lease expired"}, separators=(",", ":"))

        def operation(connection: sqlite3.Connection) -> int:
            rows = connection.execute(
                """
                SELECT task_id, attempt_count, max_attempts
                FROM tasks
                WHERE status = 'leased' AND lease_expires_at <= ?
                """,
                (timestamp,),
            ).fetchall()
            for row in rows:
                retry = int(row["attempt_count"]) < int(row["max_attempts"])
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = ?, available_at = ?, updated_at = ?,
                        completed_at = ?, error_json = ?, lease_owner = NULL,
                        lease_expires_at = NULL
                    WHERE task_id = ? AND status = 'leased'
                    """,
                    (
                        "retry_wait" if retry else "failed",
                        timestamp,
                        timestamp,
                        None if retry else timestamp,
                        error_json,
                        row["task_id"],
                    ),
                )
                connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = 'interrupted', completed_at = ?, error_json = ?
                    WHERE task_id = ? AND status = 'running'
                    """,
                    (timestamp, error_json, row["task_id"]),
                )
            return len(rows)

        return self.ledger.write_transaction(operation)

    def task_counts(self, run_id: str) -> dict[str, int]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM tasks WHERE run_id = ? GROUP BY status
                """,
                (run_id,),
            ).fetchall()
            return {str(row["status"]): int(row["count"]) for row in rows}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "task_id": str(row["task_id"]),
                "run_id": str(row["run_id"]),
                "status": str(row["status"]),
                "attempt_count": int(row["attempt_count"]),
                "max_attempts": int(row["max_attempts"]),
                "cancellation_requested": bool(row["cancellation_requested"]),
                "lease_owner": row["lease_owner"],
                "error": json.loads(row["error_json"]) if row["error_json"] else None,
            }
