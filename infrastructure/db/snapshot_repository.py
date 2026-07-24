"""Persistence and graph queries for immutable repository snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from domain.contracts import RepositorySnapshot
from infrastructure.artifacts.store import StoredArtifact

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteSnapshotRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def persist(
        self,
        snapshot: RepositorySnapshot,
        *,
        display_name: str,
        artifacts: Sequence[StoredArtifact],
        files: Sequence[dict[str, Any]],
        symbols: Sequence[dict[str, Any]],
        edges: Sequence[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> bool:
        """Atomically persist a completed snapshot; return false when cached."""

        now = _utc_now()

        def operation(connection: sqlite3.Connection) -> bool:
            existing = connection.execute(
                "SELECT snapshot_id FROM repository_snapshots WHERE identity_hash = ?",
                (snapshot.identity_hash,),
            ).fetchone()
            if existing is not None:
                return False
            connection.execute(
                """
                INSERT INTO projects(
                    project_id, canonical_path, display_name, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(canonical_path) DO UPDATE SET
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot.project_id,
                    snapshot.canonical_path,
                    display_name,
                    now,
                    now,
                ),
            )
            for artifact in artifacts:
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
                        now,
                    ),
                )
            connection.execute(
                """
                INSERT INTO repository_snapshots(
                    snapshot_id, project_id, identity_hash, configuration_hash,
                    scanner_version, git_repository, base_commit, head_commit,
                    dirty, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.project_id,
                    snapshot.identity_hash,
                    snapshot.configuration_hash,
                    snapshot.scanner_version,
                    snapshot.git_repository,
                    snapshot.base_commit,
                    snapshot.head_commit,
                    int(snapshot.dirty),
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    snapshot.created_at.isoformat(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO files(
                    file_id, snapshot_id, relative_path, content_hash, language,
                    classification, support_tier, byte_size, line_count,
                    artifact_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["file_id"],
                        snapshot.snapshot_id,
                        item["relative_path"],
                        item["content_hash"],
                        item["language"],
                        item["classification"],
                        item["support_tier"],
                        item["byte_size"],
                        item["line_count"],
                        item["artifact_id"],
                        json.dumps(
                            item.get("metadata", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    for item in files
                ],
            )
            connection.executemany(
                """
                INSERT INTO symbols(
                    symbol_id, snapshot_id, file_id, qualified_name, symbol_kind,
                    start_line, end_line, signature, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["symbol_id"],
                        snapshot.snapshot_id,
                        item["file_id"],
                        item["qualified_name"],
                        item["symbol_kind"],
                        item["start_line"],
                        item["end_line"],
                        item.get("signature"),
                        item["confidence"],
                    )
                    for item in symbols
                ],
            )
            connection.executemany(
                """
                INSERT INTO edges(
                    edge_id, snapshot_id, source_id, target_id, edge_kind,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["edge_id"],
                        snapshot.snapshot_id,
                        item["source_id"],
                        item["target_id"],
                        item["edge_kind"],
                        item["confidence"],
                        json.dumps(
                            item.get("metadata", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    for item in edges
                ],
            )
            return True

        return self.ledger.write_transaction(operation)

    def get_by_identity(self, identity_hash: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT repository_snapshots.*, projects.canonical_path
                FROM repository_snapshots
                JOIN projects USING(project_id)
                WHERE identity_hash = ?
                """,
                (identity_hash,),
            ).fetchone()
            return self._snapshot_record(row) if row else None

    def get(self, snapshot_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT repository_snapshots.*, projects.canonical_path
                FROM repository_snapshots
                JOIN projects USING(project_id)
                WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            return self._snapshot_record(row) if row else None

    def list_files(self, snapshot_id: str) -> list[dict[str, Any]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT file_id, relative_path, content_hash, language,
                       classification, support_tier, byte_size, line_count,
                       artifact_id, metadata_json
                FROM files WHERE snapshot_id = ? ORDER BY relative_path
                """,
                (snapshot_id,),
            ).fetchall()
            return [
                {
                    **{key: row[key] for key in row.keys() if key != "metadata_json"},
                    "metadata": json.loads(row["metadata_json"]),
                }
                for row in rows
            ]

    def neighborhood(
        self,
        snapshot_id: str,
        seed_file_ids: Sequence[str],
        *,
        depth: int = 1,
        max_files: int = 500,
    ) -> list[str]:
        if depth < 0:
            raise ValueError("depth cannot be negative")
        if max_files < 1:
            raise ValueError("max_files must be greater than zero")
        visited = set(seed_file_ids)
        frontier = set(seed_file_ids)
        with database_connection(self.database_path) as connection:
            for _ in range(depth):
                if not frontier or len(visited) >= max_files:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = connection.execute(
                    f"""
                    WITH file_edges(source_id, target_id) AS (
                        SELECT source_id, target_id
                        FROM edges
                        WHERE snapshot_id = ? AND edge_kind = 'imports'
                        UNION
                        SELECT source_symbol.file_id, target_symbol.file_id
                        FROM edges
                        JOIN symbols AS source_symbol
                          ON source_symbol.symbol_id = edges.source_id
                        JOIN symbols AS target_symbol
                          ON target_symbol.symbol_id = edges.target_id
                        WHERE edges.snapshot_id = ? AND edges.edge_kind = 'calls'
                    )
                    SELECT source_id, target_id FROM file_edges
                    WHERE (
                        source_id IN ({placeholders})
                        OR target_id IN ({placeholders})
                      )
                    """,
                    (snapshot_id, snapshot_id, *frontier, *frontier),
                ).fetchall()
                next_frontier = {
                    str(identifier)
                    for row in rows
                    for identifier in (row["source_id"], row["target_id"])
                    if identifier not in visited
                }
                remaining = max_files - len(visited)
                frontier = set(sorted(next_frontier)[:remaining])
                visited.update(frontier)
        return sorted(visited)

    @staticmethod
    def _snapshot_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "snapshot_id": str(row["snapshot_id"]),
            "project_id": str(row["project_id"]),
            "canonical_path": str(row["canonical_path"]),
            "identity_hash": str(row["identity_hash"]),
            "configuration_hash": str(row["configuration_hash"]),
            "scanner_version": str(row["scanner_version"]),
            "git_repository": row["git_repository"],
            "base_commit": row["base_commit"],
            "head_commit": row["head_commit"],
            "dirty": bool(row["dirty"]),
            "metadata": json.loads(row["metadata_json"]),
            "created_at": str(row["created_at"]),
        }
