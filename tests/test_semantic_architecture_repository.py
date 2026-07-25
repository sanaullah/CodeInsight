from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.architecture import (
    ArchitectureCompleteness,
    ArchitectureSupportTier,
    ComponentKind,
    ProvenanceSpan,
    RelationKind,
    SemanticArchitectureProjection,
    SemanticBoundary,
    SemanticComponent,
    SemanticEndpoint,
    SemanticMembership,
    SemanticRelation,
    SemanticResource,
)
from domain.contracts import RepositorySnapshot
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.component_annotation_repository import (
    SqliteComponentAnnotationRepository,
)
from infrastructure.db.database import SCHEMA_VERSION, database_connection
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.semantic_architecture_repository import (
    SqliteSemanticArchitectureRepository,
)
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository


def _seed_snapshot(tmp_path: Path) -> tuple[SqliteRunLedger, str, str]:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    artifacts = FilesystemArtifactStore(tmp_path / "artifacts")
    stored = artifacts.put(b"from fastapi import FastAPI\n", artifact_kind="source")
    snapshot_id = "snapshot-1"
    file_id = "file-1"
    snapshot = RepositorySnapshot(
        snapshot_id=snapshot_id,
        project_id="project-1",
        canonical_path=str(tmp_path / "repo"),
        identity_hash="identity-1",
        configuration_hash="configuration-1",
        scanner_version="test",
        created_at=datetime.now(UTC),
    )
    assert SqliteSnapshotRepository(ledger).persist(
        snapshot,
        display_name="repo",
        artifacts=[stored],
        files=[
            {
                "file_id": file_id,
                "relative_path": "app.py",
                "content_hash": stored.content_hash,
                "language": "python",
                "classification": "source",
                "support_tier": "parsed",
                "byte_size": stored.byte_size,
                "line_count": 1,
                "artifact_id": stored.artifact_id,
            }
        ],
        symbols=[],
        edges=[],
        metadata={},
    )
    return ledger, snapshot_id, file_id


def _projection(snapshot_id: str, file_id: str) -> SemanticArchitectureProjection:
    service = SemanticComponent(
        component_id="component-service",
        snapshot_id=snapshot_id,
        stable_key="service:api",
        name="API",
        component_kind=ComponentKind.SERVICE,
        support_tier=ArchitectureSupportTier.EXACT,
        completeness=ArchitectureCompleteness.PARTIAL,
        confidence=0.95,
    )
    store = SemanticComponent(
        component_id="component-store",
        snapshot_id=snapshot_id,
        stable_key="datastore:sqlite",
        name="SQLite",
        component_kind=ComponentKind.DATASTORE,
        support_tier=ArchitectureSupportTier.INFERRED,
        completeness=ArchitectureCompleteness.PARTIAL,
        confidence=0.8,
    )
    return SemanticArchitectureProjection(
        snapshot_id=snapshot_id,
        extractor_version="semantic-v1",
        components=(service, store),
        memberships=(
            SemanticMembership(
                membership_id="membership-1",
                snapshot_id=snapshot_id,
                component_id=service.component_id,
                file_id=file_id,
                membership_kind="declares",
                confidence=1,
            ),
        ),
        boundaries=(
            SemanticBoundary(
                boundary_id="boundary-1",
                snapshot_id=snapshot_id,
                stable_key="boundary:application",
                name="Application",
                boundary_kind="application",
                support_tier=ArchitectureSupportTier.INFERRED,
                completeness=ArchitectureCompleteness.PARTIAL,
                confidence=0.8,
                component_ids=(service.component_id, store.component_id),
            ),
        ),
        resources=(
            SemanticResource(
                resource_id="resource-1",
                snapshot_id=snapshot_id,
                component_id=store.component_id,
                stable_key="resource:sqlite:ledger",
                resource_kind="database",
                name="Application ledger",
                locator="codeinsight.db",
                support_tier=ArchitectureSupportTier.EXACT,
                completeness=ArchitectureCompleteness.COMPLETE,
                confidence=1,
            ),
        ),
        endpoints=(
            SemanticEndpoint(
                endpoint_id="endpoint-1",
                snapshot_id=snapshot_id,
                component_id=service.component_id,
                stable_key="endpoint:get:/health",
                protocol="http",
                method="GET",
                route="/health",
                direction="inbound",
                support_tier=ArchitectureSupportTier.EXACT,
                completeness=ArchitectureCompleteness.COMPLETE,
                confidence=1,
            ),
        ),
        relations=(
            SemanticRelation(
                relation_id="relation-1",
                snapshot_id=snapshot_id,
                source_component_id=service.component_id,
                target_component_id=store.component_id,
                stable_key="data_access:api:sqlite",
                relation_kind=RelationKind.DATA_ACCESS,
                transport="sqlite",
                support_tier=ArchitectureSupportTier.INFERRED,
                completeness=ArchitectureCompleteness.PARTIAL,
                confidence=0.8,
            ),
        ),
        provenance=(
            ProvenanceSpan(
                provenance_id="provenance-1",
                entity_kind="component",
                entity_id=service.component_id,
                file_id=file_id,
                relative_path="app.py",
                start_line=1,
                end_line=1,
                derivation="fastapi-import",
                extractor_version="semantic-v1",
                confidence=0.95,
            ),
        ),
    )


def test_semantic_schema_is_canonical_and_indexed(tmp_path: Path) -> None:
    ledger, _, _ = _seed_snapshot(tmp_path)
    with database_connection(ledger.database_path) as connection:
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert version == SCHEMA_VERSION == 8
    assert {
        "semantic_components",
        "semantic_memberships",
        "semantic_boundaries",
        "semantic_boundary_memberships",
        "semantic_resources",
        "semantic_endpoints",
        "semantic_relations",
        "semantic_provenance",
        "semantic_finding_links",
        "architecture_component_annotations",
        "architecture_component_annotation_events",
    } <= tables
    assert "idx_semantic_relations_source" in indexes
    assert "idx_semantic_provenance_entity" in indexes
    assert "idx_architecture_annotations_snapshot" in indexes


def test_projection_replace_is_atomic_idempotent_and_queryable(tmp_path: Path) -> None:
    ledger, snapshot_id, file_id = _seed_snapshot(tmp_path)
    repository = SqliteSemanticArchitectureRepository(ledger)
    projection = _projection(snapshot_id, file_id)

    assert repository.replace(projection)["components"] == 2
    annotations = SqliteComponentAnnotationRepository(ledger)
    annotation = annotations.upsert(
        snapshot_id,
        "component-service",
        note="Preserve this local context.",
        expected_version=0,
    )
    assert annotation is not None
    assert repository.replace(projection) == repository.projection_counts(snapshot_id)
    preserved = annotations.get(snapshot_id, "component-service")
    assert preserved is not None
    assert preserved["note"] == "Preserve this local context."
    with database_connection(ledger.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM architecture_component_annotation_events"
        ).fetchone()[0] == 1
    assert repository.projection_state(snapshot_id)["extractor_version"] == "semantic-v1"
    component = repository.component("component-service")
    assert component is not None
    assert component["stable_key"] == "service:api"
    assert component["files"][0]["relative_path"] == "app.py"

    with database_connection(ledger.database_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_projection_rejects_unknown_snapshot_and_rolls_back(tmp_path: Path) -> None:
    ledger, snapshot_id, file_id = _seed_snapshot(tmp_path)
    repository = SqliteSemanticArchitectureRepository(ledger)
    projection = _projection(snapshot_id, file_id)
    repository.replace(projection)
    missing = projection.model_copy(update={"snapshot_id": "missing"})

    with pytest.raises(ValueError, match="snapshot"):
        repository.replace(missing)
    assert repository.projection_counts(snapshot_id)["components"] == 2


def test_semantic_foreign_keys_reject_cross_snapshot_membership(tmp_path: Path) -> None:
    ledger, snapshot_id, file_id = _seed_snapshot(tmp_path)
    projection = _projection(snapshot_id, file_id)
    invalid = projection.model_copy(
        update={
            "memberships": (
                projection.memberships[0].model_copy(
                    update={"file_id": "missing-file"}
                ),
            )
        }
    )

    with pytest.raises(sqlite3.IntegrityError):
        SqliteSemanticArchitectureRepository(ledger).replace(invalid)
    assert (
        SqliteSemanticArchitectureRepository(ledger).projection_counts(snapshot_id)[
            "components"
        ]
        == 0
    )


def test_semantic_contracts_reject_unbounded_or_targetless_values() -> None:
    with pytest.raises(ValueError, match="file_id"):
        SemanticMembership(
            membership_id="membership",
            snapshot_id="snapshot",
            component_id="component",
            membership_kind="declares",
            confidence=1,
        )
    with pytest.raises(ValueError, match="end_line"):
        ProvenanceSpan(
            provenance_id="provenance",
            entity_kind="component",
            entity_id="component",
            file_id="file",
            start_line=2,
            end_line=1,
            derivation="test",
            extractor_version="test",
            confidence=1,
        )
