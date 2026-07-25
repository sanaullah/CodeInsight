"""Immutable redacted specialist-prompt artifacts in the canonical ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from .run_ledger import SqliteRunLedger

DEFAULT_PROMPT_RETENTION_DAYS = 90
PROMPT_RETENTION_POLICY = "application-default"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqlitePromptArtifactRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger

    def record(
        self,
        *,
        prompt_artifact_id: str,
        run_id: str,
        wave_id: str,
        role_id: str,
        task_id: str,
        prompt_template: str,
        prompt_version: int,
        prompt_text: str,
        request_hash: str,
        retention_days: int = DEFAULT_PROMPT_RETENTION_DAYS,
    ) -> None:
        """Insert once and reject any attempted identifier reuse with different data."""

        redaction = {
            "provider_secrets_included": False,
            "source_content_included": False,
            "user_prompt_persisted": False,
        }
        values = (
            prompt_artifact_id,
            run_id,
            wave_id,
            role_id,
            task_id,
            prompt_template,
            prompt_version,
            prompt_text,
            request_hash,
            json.dumps(redaction, sort_keys=True, separators=(",", ":")),
            PROMPT_RETENTION_POLICY,
            retention_days,
            _now(),
        )

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR IGNORE INTO specialist_prompt_artifacts(
                    prompt_artifact_id, run_id, wave_id, role_id, task_id,
                    prompt_template, prompt_version, prompt_text, request_hash,
                    source_content_included, redaction_json, retention_policy,
                    retention_days, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                values,
            )
            stored = connection.execute(
                """
                SELECT run_id, wave_id, role_id, task_id, prompt_template,
                       prompt_version, prompt_text, request_hash, redaction_json,
                       retention_policy, retention_days
                FROM specialist_prompt_artifacts
                WHERE prompt_artifact_id = ?
                """,
                (prompt_artifact_id,),
            ).fetchone()
            expected = values[1:9] + values[9:12]
            actual = tuple(stored[key] for key in stored.keys()) if stored else ()
            if actual != expected:
                raise RuntimeError("immutable specialist prompt artifact conflict")

        self.ledger.write_transaction(operation)
