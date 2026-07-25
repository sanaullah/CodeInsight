"""Metadata persistence for immutable content-addressed artifacts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infrastructure.artifacts.store import StoredArtifact

from .run_ledger import SqliteRunLedger


class SqliteArtifactRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def register(self, artifact: StoredArtifact) -> None:
        created_at = datetime.now(UTC).isoformat()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, content_hash, artifact_kind, storage_path,
                    byte_size, media_type, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
                ON CONFLICT(content_hash, artifact_kind) DO NOTHING
                """,
                (
                    artifact.artifact_id,
                    artifact.content_hash,
                    artifact.artifact_kind,
                    artifact.storage_path,
                    artifact.byte_size,
                    artifact.media_type,
                    created_at,
                ),
            )

        self.ledger.write_transaction(operation)
