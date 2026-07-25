"""Durable optimistic-concurrency notes for immutable architecture components."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteComponentAnnotationRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger

    def get(self, snapshot_id: str, component_id: str) -> dict[str, Any] | None:
        with database_connection(self.ledger.database_path) as connection:
            row = connection.execute(
                """
                SELECT annotations.annotation_id, annotations.snapshot_id,
                       components.component_id, annotations.note,
                       annotations.version, annotations.actor,
                       annotations.created_at, annotations.updated_at
                FROM architecture_component_annotations AS annotations
                JOIN semantic_components AS components
                  ON components.snapshot_id = annotations.snapshot_id
                 AND components.stable_key = annotations.component_stable_key
                WHERE components.snapshot_id = ? AND components.component_id = ?
                """,
                (snapshot_id, component_id),
            ).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        snapshot_id: str,
        component_id: str,
        *,
        note: str,
        expected_version: int,
        actor: str = "local-user",
    ) -> dict[str, Any] | None:
        normalized_note = note.strip()
        if not normalized_note or len(normalized_note) > 4000:
            raise ValueError("annotation note must contain 1 to 4000 characters")
        timestamp = _now()

        def operation(connection):
            component = connection.execute(
                """
                SELECT stable_key FROM semantic_components
                WHERE snapshot_id = ? AND component_id = ?
                """,
                (snapshot_id, component_id),
            ).fetchone()
            if component is None:
                return None
            stable_key = str(component["stable_key"])
            current = connection.execute(
                """
                SELECT annotation_id, version, created_at
                FROM architecture_component_annotations
                WHERE snapshot_id = ? AND component_stable_key = ?
                """,
                (snapshot_id, stable_key),
            ).fetchone()
            current_version = int(current["version"]) if current else 0
            if current_version != expected_version:
                raise RuntimeError("component annotation changed; refresh and retry")
            annotation_id = (
                str(current["annotation_id"]) if current else f"annotation-{uuid4()}"
            )
            created_at = str(current["created_at"]) if current else timestamp
            next_version = current_version + 1
            connection.execute(
                """
                INSERT INTO architecture_component_annotations(
                    annotation_id, snapshot_id, component_stable_key, note, version,
                    actor, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id, component_stable_key) DO UPDATE SET
                    note = excluded.note,
                    version = excluded.version,
                    actor = excluded.actor,
                    updated_at = excluded.updated_at
                """,
                (
                    annotation_id,
                    snapshot_id,
                    stable_key,
                    normalized_note,
                    next_version,
                    actor,
                    created_at,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO architecture_component_annotation_events(
                    annotation_id, snapshot_id, component_stable_key, note, version,
                    actor, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    snapshot_id,
                    stable_key,
                    normalized_note,
                    next_version,
                    actor,
                    timestamp,
                ),
            )
            return True

        changed = self.ledger.write_transaction(operation)
        return self.get(snapshot_id, component_id) if changed else None
