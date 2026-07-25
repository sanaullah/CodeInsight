"""Persistence for evidence-derived semantic architecture projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from domain.architecture import SemanticArchitectureProjection

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class SqliteSemanticArchitectureRepository:
    """Atomically replace and read one snapshot's versioned semantic projection."""

    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def replace(self, projection: SemanticArchitectureProjection) -> dict[str, int]:
        snapshot_id = projection.snapshot_id

        def operation(connection):
            exists = connection.execute(
                "SELECT 1 FROM repository_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            if exists is None:
                raise ValueError("repository snapshot does not exist")
            connection.execute(
                """
                DELETE FROM semantic_boundary_memberships
                WHERE boundary_id IN (
                    SELECT boundary_id FROM semantic_boundaries
                    WHERE snapshot_id = ?
                )
                """,
                (snapshot_id,),
            )
            for table in (
                "semantic_finding_links",
                "semantic_provenance",
                "semantic_memberships",
                "semantic_endpoints",
                "semantic_resources",
                "semantic_relations",
                "semantic_boundaries",
                "semantic_components",
            ):
                connection.execute(f"DELETE FROM {table} WHERE snapshot_id = ?", (snapshot_id,))
            connection.executemany(
                """
                INSERT INTO semantic_components(
                    component_id, snapshot_id, stable_key, name, component_kind,
                    support_tier, completeness, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.component_id,
                        snapshot_id,
                        item.stable_key,
                        item.name,
                        item.component_kind.value,
                        item.support_tier.value,
                        item.completeness.value,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.components
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_boundaries(
                    boundary_id, snapshot_id, stable_key, name, boundary_kind,
                    support_tier, completeness, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.boundary_id,
                        snapshot_id,
                        item.stable_key,
                        item.name,
                        item.boundary_kind,
                        item.support_tier.value,
                        item.completeness.value,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.boundaries
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_memberships(
                    membership_id, snapshot_id, component_id, file_id, symbol_id,
                    membership_kind, confidence, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.membership_id,
                        snapshot_id,
                        item.component_id,
                        item.file_id,
                        item.symbol_id,
                        item.membership_kind,
                        item.confidence,
                        _json(item.provenance),
                    )
                    for item in projection.memberships
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_boundary_memberships(boundary_id, component_id)
                VALUES (?, ?)
                """,
                [
                    (boundary.boundary_id, component_id)
                    for boundary in projection.boundaries
                    for component_id in boundary.component_ids
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_resources(
                    resource_id, snapshot_id, component_id, stable_key,
                    resource_kind, name, locator, support_tier, completeness,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.resource_id,
                        snapshot_id,
                        item.component_id,
                        item.stable_key,
                        item.resource_kind,
                        item.name,
                        item.locator,
                        item.support_tier.value,
                        item.completeness.value,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.resources
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_endpoints(
                    endpoint_id, snapshot_id, component_id, stable_key, protocol,
                    method, route, direction, support_tier, completeness,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.endpoint_id,
                        snapshot_id,
                        item.component_id,
                        item.stable_key,
                        item.protocol,
                        item.method,
                        item.route,
                        item.direction,
                        item.support_tier.value,
                        item.completeness.value,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.endpoints
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_relations(
                    relation_id, snapshot_id, source_component_id,
                    target_component_id, stable_key, relation_kind, transport,
                    is_async, support_tier, completeness, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.relation_id,
                        snapshot_id,
                        item.source_component_id,
                        item.target_component_id,
                        item.stable_key,
                        item.relation_kind.value,
                        item.transport,
                        int(item.is_async),
                        item.support_tier.value,
                        item.completeness.value,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.relations
                ],
            )
            connection.executemany(
                """
                INSERT INTO semantic_provenance(
                    provenance_id, snapshot_id, entity_kind, entity_id, file_id,
                    start_line, end_line, derivation, extractor_version,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.provenance_id,
                        snapshot_id,
                        item.entity_kind,
                        item.entity_id,
                        item.file_id,
                        item.start_line,
                        item.end_line,
                        item.derivation,
                        item.extractor_version,
                        item.confidence,
                        _json(item.metadata),
                    )
                    for item in projection.provenance
                ],
            )
            counts = {
                "components": len(projection.components),
                "memberships": len(projection.memberships),
                "boundaries": len(projection.boundaries),
                "resources": len(projection.resources),
                "endpoints": len(projection.endpoints),
                "relations": len(projection.relations),
                "provenance": len(projection.provenance),
            }
            completeness = {item.completeness.value for item in projection.components}
            if not completeness:
                status = "unknown"
            elif completeness == {"unsupported"}:
                status = "unsupported"
            elif completeness == {"complete"}:
                status = "complete"
            else:
                status = "partial"
            connection.execute(
                """
                INSERT INTO semantic_projection_state(
                    snapshot_id, extractor_version, status, counts_json,
                    generated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    extractor_version = excluded.extractor_version,
                    status = excluded.status,
                    counts_json = excluded.counts_json,
                    generated_at = excluded.generated_at
                """,
                (
                    snapshot_id,
                    projection.extractor_version,
                    status,
                    _json(counts),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return counts

        return self.ledger.write_transaction(operation)

    def projection_state(self, snapshot_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT extractor_version, status, counts_json, generated_at
                FROM semantic_projection_state WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "extractor_version": str(row["extractor_version"]),
                "status": str(row["status"]),
                "counts": json.loads(row["counts_json"]),
                "generated_at": str(row["generated_at"]),
            }

    def projection_counts(self, snapshot_id: str) -> dict[str, int] | None:
        with database_connection(self.database_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM repository_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            if exists is None:
                return None
            tables = {
                "components": "semantic_components",
                "memberships": "semantic_memberships",
                "boundaries": "semantic_boundaries",
                "resources": "semantic_resources",
                "endpoints": "semantic_endpoints",
                "relations": "semantic_relations",
                "provenance": "semantic_provenance",
            }
            return {
                key: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE snapshot_id = ?",
                        (snapshot_id,),
                    ).fetchone()[0]
                )
                for key, table in tables.items()
            }

    def component(self, component_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM semantic_components WHERE component_id = ?
                """,
                (component_id,),
            ).fetchone()
            if row is None:
                return None
            files = connection.execute(
                """
                SELECT files.file_id, files.relative_path, files.language,
                       files.line_count, memberships.membership_kind,
                       memberships.confidence
                FROM semantic_memberships AS memberships
                LEFT JOIN files USING(file_id)
                WHERE memberships.component_id = ?
                ORDER BY files.relative_path
                """,
                (component_id,),
            ).fetchall()
            return {
                **{
                    key: row[key]
                    for key in row.keys()
                    if key != "metadata_json"
                },
                "metadata": json.loads(row["metadata_json"]),
                "files": [dict(item) for item in files],
            }
