"""Canonical creation and migration source for the CodeInsight SQLite ledger.

This is the only module allowed to define application tables or connection
pragmas. Every durable repository uses the same database file and calls the
connection helpers here.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 8
DEFAULT_BUSY_TIMEOUT_MS = 5_000

_MIGRATION_1 = (
    """
    CREATE TABLE projects (
        project_id TEXT PRIMARY KEY,
        canonical_path TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        artifact_kind TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        media_type TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(content_hash, artifact_kind)
    )
    """,
    """
    CREATE TABLE repository_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(project_id),
        identity_hash TEXT NOT NULL UNIQUE,
        configuration_hash TEXT NOT NULL,
        scanner_version TEXT NOT NULL,
        git_repository TEXT,
        base_commit TEXT,
        head_commit TEXT,
        dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0, 1)),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE files (
        file_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id),
        relative_path TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        language TEXT,
        classification TEXT NOT NULL,
        support_tier TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        line_count INTEGER NOT NULL CHECK (line_count >= 0),
        artifact_id TEXT REFERENCES artifacts(artifact_id),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, relative_path)
    )
    """,
    """
    CREATE TABLE symbols (
        symbol_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id),
        file_id TEXT NOT NULL REFERENCES files(file_id),
        qualified_name TEXT NOT NULL,
        symbol_kind TEXT NOT NULL,
        start_line INTEGER NOT NULL CHECK (start_line >= 1),
        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
        signature TEXT,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1)
    )
    """,
    """
    CREATE TABLE edges (
        edge_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id),
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        edge_kind TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        submission_key TEXT NOT NULL UNIQUE,
        snapshot_id TEXT REFERENCES repository_snapshots(snapshot_id),
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled',
                       'needs_attention')
        ),
        current_stage TEXT,
        mode TEXT NOT NULL DEFAULT 'deep',
        request_json TEXT NOT NULL,
        budget_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        error TEXT,
        cancellation_requested INTEGER NOT NULL DEFAULT 0
            CHECK (cancellation_requested IN (0, 1)),
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_stages (
        run_stage_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        stage_version TEXT NOT NULL,
        status TEXT NOT NULL,
        input_json TEXT NOT NULL DEFAULT '{}',
        output_artifact_id TEXT REFERENCES artifacts(artifact_id),
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        lease_owner TEXT,
        lease_expires_at TEXT,
        started_at TEXT,
        completed_at TEXT,
        error_json TEXT,
        UNIQUE(run_id, stage, stage_version)
    )
    """,
    """
    CREATE TABLE waves (
        wave_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_number INTEGER NOT NULL CHECK (wave_number >= 1),
        rationale TEXT NOT NULL,
        status TEXT NOT NULL,
        budget_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE(run_id, wave_number)
    )
    """,
    """
    CREATE TABLE role_specs (
        role_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_id TEXT NOT NULL REFERENCES waves(wave_id) ON DELETE CASCADE,
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_id TEXT NOT NULL REFERENCES waves(wave_id) ON DELETE CASCADE,
        role_id TEXT REFERENCES role_specs(role_id),
        task_type TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 100,
        status TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        input_json TEXT NOT NULL,
        budget_json TEXT NOT NULL,
        routing_policy_json TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
        cancellation_requested INTEGER NOT NULL DEFAULT 0
            CHECK (cancellation_requested IN (0, 1)),
        lease_owner TEXT,
        lease_expires_at TEXT,
        available_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        output_artifact_id TEXT REFERENCES artifacts(artifact_id),
        error_json TEXT
    )
    """,
    """
    CREATE TABLE task_attempts (
        attempt_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
        worker_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        error_json TEXT,
        usage_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(task_id, attempt_number)
    )
    """,
    """
    CREATE TABLE events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL CHECK (sequence >= 1),
        event_type TEXT NOT NULL,
        stage TEXT,
        wave_id TEXT,
        task_id TEXT,
        data_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(run_id, sequence)
    )
    """,
    """
    CREATE TABLE evidence_refs (
        evidence_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id),
        file_id TEXT NOT NULL REFERENCES files(file_id),
        content_hash TEXT NOT NULL,
        start_line INTEGER NOT NULL CHECK (start_line >= 1),
        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
        start_byte INTEGER,
        end_byte INTEGER,
        excerpt_hash TEXT NOT NULL,
        evidence_kind TEXT NOT NULL,
        provenance_json TEXT NOT NULL,
        collected_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE finding_candidates (
        candidate_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT NOT NULL REFERENCES tasks(task_id),
        role_id TEXT REFERENCES role_specs(role_id),
        fingerprint TEXT NOT NULL,
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE finding_candidate_evidence (
        candidate_id TEXT NOT NULL REFERENCES finding_candidates(candidate_id)
            ON DELETE CASCADE,
        evidence_id TEXT NOT NULL REFERENCES evidence_refs(evidence_id),
        PRIMARY KEY(candidate_id, evidence_id)
    )
    """,
    """
    CREATE TABLE finding_verdicts (
        verdict_id TEXT PRIMARY KEY,
        candidate_id TEXT NOT NULL REFERENCES finding_candidates(candidate_id)
            ON DELETE CASCADE,
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE canonical_findings (
        finding_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        fingerprint TEXT NOT NULL,
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(run_id, fingerprint)
    )
    """,
    """
    CREATE TABLE canonical_finding_members (
        finding_id TEXT NOT NULL REFERENCES canonical_findings(finding_id)
            ON DELETE CASCADE,
        candidate_id TEXT NOT NULL REFERENCES finding_candidates(candidate_id),
        relationship TEXT NOT NULL,
        PRIMARY KEY(finding_id, candidate_id)
    )
    """,
    """
    CREATE TABLE coverage_assessments (
        assessment_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_id TEXT NOT NULL REFERENCES waves(wave_id),
        contract_json TEXT NOT NULL,
        contract_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE model_calls (
        model_call_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_id TEXT REFERENCES waves(wave_id),
        task_id TEXT REFERENCES tasks(task_id),
        attempt_id TEXT REFERENCES task_attempts(attempt_id),
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        response_artifact_id TEXT REFERENCES artifacts(artifact_id),
        status TEXT NOT NULL,
        usage_json TEXT NOT NULL DEFAULT '{}',
        trace_correlation_json TEXT NOT NULL DEFAULT '{}',
        started_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE(task_id, request_hash, provider, model)
    )
    """,
    """
    CREATE TABLE experience_records (
        experience_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        task_id TEXT REFERENCES tasks(task_id),
        record_json TEXT NOT NULL,
        evaluation_json TEXT,
        promotion_status TEXT NOT NULL DEFAULT 'unreviewed',
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_runs_status_created ON runs(status, created_at)",
    "CREATE INDEX idx_tasks_runnable ON tasks(status, available_at, priority)",
    "CREATE INDEX idx_tasks_run_status ON tasks(run_id, status)",
    "CREATE INDEX idx_events_run_sequence ON events(run_id, sequence)",
    "CREATE INDEX idx_files_snapshot_language ON files(snapshot_id, language)",
    "CREATE INDEX idx_symbols_snapshot_name ON symbols(snapshot_id, qualified_name)",
    "CREATE INDEX idx_edges_snapshot_source ON edges(snapshot_id, source_id)",
    "CREATE INDEX idx_candidates_run_fingerprint ON finding_candidates(run_id, fingerprint)",
)

_MIGRATION_2 = (
    """
    CREATE TABLE finding_reviews (
        finding_id TEXT PRIMARY KEY REFERENCES canonical_findings(finding_id)
            ON DELETE CASCADE,
        review_state TEXT NOT NULL CHECK (
            review_state IN ('new', 'validated', 'acknowledged', 'reviewed',
                             'dismissed', 'reopened', 'resolved')
        ),
        note TEXT,
        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE finding_review_events (
        review_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id TEXT NOT NULL REFERENCES canonical_findings(finding_id)
            ON DELETE CASCADE,
        previous_state TEXT,
        review_state TEXT NOT NULL,
        note TEXT,
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_finding_reviews_state ON finding_reviews(review_state, updated_at)",
    "CREATE INDEX idx_finding_review_events_finding "
    "ON finding_review_events(finding_id, review_event_id)",
)

_MIGRATION_3 = (
    """
    CREATE TABLE application_settings (
        settings_key TEXT PRIMARY KEY,
        settings_json TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE review_presets (
        preset_id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        request_json TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_review_presets_name ON review_presets(name)",
)

_MIGRATION_4 = (
    "ALTER TABLE repository_snapshots ADD COLUMN parent_snapshot_id TEXT "
    "REFERENCES repository_snapshots(snapshot_id)",
    "ALTER TABLE repository_snapshots ADD COLUMN git_ref TEXT",
    """
    CREATE TABLE semantic_components (
        component_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        stable_key TEXT NOT NULL,
        name TEXT NOT NULL,
        component_kind TEXT NOT NULL CHECK (
            component_kind IN ('service', 'datastore', 'external_system',
                               'library', 'queue', 'unknown')
        ),
        support_tier TEXT NOT NULL CHECK (
            support_tier IN ('exact', 'inferred', 'partial', 'unsupported')
        ),
        completeness TEXT NOT NULL CHECK (
            completeness IN ('complete', 'partial', 'unknown', 'unsupported')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, stable_key)
    )
    """,
    """
    CREATE TABLE semantic_boundaries (
        boundary_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        stable_key TEXT NOT NULL,
        name TEXT NOT NULL,
        boundary_kind TEXT NOT NULL,
        support_tier TEXT NOT NULL CHECK (
            support_tier IN ('exact', 'inferred', 'partial', 'unsupported')
        ),
        completeness TEXT NOT NULL CHECK (
            completeness IN ('complete', 'partial', 'unknown', 'unsupported')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, stable_key)
    )
    """,
    """
    CREATE TABLE semantic_memberships (
        membership_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_id TEXT NOT NULL REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        file_id TEXT REFERENCES files(file_id) ON DELETE CASCADE,
        symbol_id TEXT REFERENCES symbols(symbol_id) ON DELETE CASCADE,
        membership_kind TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        provenance_json TEXT NOT NULL DEFAULT '{}',
        CHECK (file_id IS NOT NULL OR symbol_id IS NOT NULL)
    )
    """,
    """
    CREATE TABLE semantic_boundary_memberships (
        boundary_id TEXT NOT NULL REFERENCES semantic_boundaries(boundary_id)
            ON DELETE CASCADE,
        component_id TEXT NOT NULL REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        PRIMARY KEY(boundary_id, component_id)
    )
    """,
    """
    CREATE TABLE semantic_resources (
        resource_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_id TEXT REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        stable_key TEXT NOT NULL,
        resource_kind TEXT NOT NULL,
        name TEXT NOT NULL,
        locator TEXT,
        support_tier TEXT NOT NULL CHECK (
            support_tier IN ('exact', 'inferred', 'partial', 'unsupported')
        ),
        completeness TEXT NOT NULL CHECK (
            completeness IN ('complete', 'partial', 'unknown', 'unsupported')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, stable_key)
    )
    """,
    """
    CREATE TABLE semantic_endpoints (
        endpoint_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_id TEXT REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        stable_key TEXT NOT NULL,
        protocol TEXT NOT NULL,
        method TEXT,
        route TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
        support_tier TEXT NOT NULL CHECK (
            support_tier IN ('exact', 'inferred', 'partial', 'unsupported')
        ),
        completeness TEXT NOT NULL CHECK (
            completeness IN ('complete', 'partial', 'unknown', 'unsupported')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, stable_key)
    )
    """,
    """
    CREATE TABLE semantic_relations (
        relation_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        source_component_id TEXT NOT NULL
            REFERENCES semantic_components(component_id) ON DELETE CASCADE,
        target_component_id TEXT NOT NULL
            REFERENCES semantic_components(component_id) ON DELETE CASCADE,
        stable_key TEXT NOT NULL,
        relation_kind TEXT NOT NULL CHECK (
            relation_kind IN ('request', 'event', 'data_access', 'dependency',
                              'call', 'unknown')
        ),
        transport TEXT,
        is_async INTEGER NOT NULL DEFAULT 0 CHECK (is_async IN (0, 1)),
        support_tier TEXT NOT NULL CHECK (
            support_tier IN ('exact', 'inferred', 'partial', 'unsupported')
        ),
        completeness TEXT NOT NULL CHECK (
            completeness IN ('complete', 'partial', 'unknown', 'unsupported')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(snapshot_id, stable_key)
    )
    """,
    """
    CREATE TABLE semantic_provenance (
        provenance_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        entity_kind TEXT NOT NULL CHECK (
            entity_kind IN ('component', 'boundary', 'membership', 'resource',
                            'endpoint', 'relation')
        ),
        entity_id TEXT NOT NULL,
        file_id TEXT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
        start_line INTEGER NOT NULL CHECK (start_line >= 1),
        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
        derivation TEXT NOT NULL,
        extractor_version TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE semantic_finding_links (
        link_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        finding_id TEXT NOT NULL REFERENCES canonical_findings(finding_id)
            ON DELETE CASCADE,
        component_id TEXT REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        relation_id TEXT REFERENCES semantic_relations(relation_id)
            ON DELETE CASCADE,
        resource_id TEXT REFERENCES semantic_resources(resource_id)
            ON DELETE CASCADE,
        link_kind TEXT NOT NULL,
        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        CHECK (
            component_id IS NOT NULL OR relation_id IS NOT NULL
            OR resource_id IS NOT NULL
        ),
        UNIQUE(snapshot_id, finding_id, component_id, relation_id, resource_id, link_kind)
    )
    """,
    "CREATE INDEX idx_semantic_components_snapshot_kind "
    "ON semantic_components(snapshot_id, component_kind, stable_key)",
    "CREATE INDEX idx_semantic_memberships_component "
    "ON semantic_memberships(snapshot_id, component_id)",
    "CREATE INDEX idx_semantic_memberships_file "
    "ON semantic_memberships(snapshot_id, file_id)",
    "CREATE INDEX idx_semantic_boundaries_snapshot "
    "ON semantic_boundaries(snapshot_id, boundary_kind)",
    "CREATE INDEX idx_semantic_resources_component "
    "ON semantic_resources(snapshot_id, component_id, resource_kind)",
    "CREATE INDEX idx_semantic_endpoints_component "
    "ON semantic_endpoints(snapshot_id, component_id, direction)",
    "CREATE INDEX idx_semantic_relations_source "
    "ON semantic_relations(snapshot_id, source_component_id, relation_kind)",
    "CREATE INDEX idx_semantic_relations_target "
    "ON semantic_relations(snapshot_id, target_component_id, relation_kind)",
    "CREATE INDEX idx_semantic_provenance_entity "
    "ON semantic_provenance(snapshot_id, entity_kind, entity_id)",
    "CREATE INDEX idx_snapshots_project_created "
    "ON repository_snapshots(project_id, created_at)",
)

_MIGRATION_5 = (
    """
    CREATE TABLE semantic_projection_state (
        snapshot_id TEXT PRIMARY KEY REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        extractor_version TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('complete', 'partial', 'unknown', 'unsupported', 'failed')
        ),
        counts_json TEXT NOT NULL DEFAULT '{}',
        generated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_semantic_projection_version "
    "ON semantic_projection_state(extractor_version, status)",
)

_MIGRATION_6 = (
    """
    CREATE TABLE specialist_prompt_artifacts (
        prompt_artifact_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        wave_id TEXT NOT NULL REFERENCES waves(wave_id) ON DELETE CASCADE,
        role_id TEXT NOT NULL REFERENCES role_specs(role_id) ON DELETE CASCADE,
        task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
        prompt_template TEXT NOT NULL,
        prompt_version INTEGER NOT NULL CHECK (prompt_version >= 1),
        prompt_text TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        source_content_included INTEGER NOT NULL DEFAULT 0
            CHECK (source_content_included = 0),
        redaction_json TEXT NOT NULL DEFAULT '{}',
        retention_policy TEXT NOT NULL,
        retention_days INTEGER NOT NULL CHECK (retention_days >= 1),
        created_at TEXT NOT NULL,
        UNIQUE(task_id, request_hash, prompt_template, prompt_version)
    )
    """,
    """
    CREATE TRIGGER specialist_prompt_artifacts_immutable
    BEFORE UPDATE ON specialist_prompt_artifacts
    BEGIN
        SELECT RAISE(ABORT, 'specialist prompt artifacts are immutable');
    END
    """,
    "CREATE INDEX idx_specialist_prompts_run_created "
    "ON specialist_prompt_artifacts(run_id, created_at)",
)

_MIGRATION_7 = (
    """
    CREATE TABLE architecture_component_annotations (
        annotation_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_id TEXT NOT NULL REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        note TEXT NOT NULL CHECK (length(note) BETWEEN 1 AND 4000),
        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(snapshot_id, component_id)
    )
    """,
    """
    CREATE TABLE architecture_component_annotation_events (
        annotation_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        annotation_id TEXT NOT NULL
            REFERENCES architecture_component_annotations(annotation_id)
            ON DELETE CASCADE,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_id TEXT NOT NULL REFERENCES semantic_components(component_id)
            ON DELETE CASCADE,
        note TEXT NOT NULL CHECK (length(note) BETWEEN 1 AND 4000),
        version INTEGER NOT NULL CHECK (version >= 1),
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_architecture_annotations_snapshot "
    "ON architecture_component_annotations(snapshot_id, updated_at)",
    "CREATE INDEX idx_architecture_annotation_events_annotation "
    "ON architecture_component_annotation_events(annotation_id, version)",
)

_MIGRATION_8 = (
    "DROP INDEX idx_architecture_annotations_snapshot",
    "DROP INDEX idx_architecture_annotation_events_annotation",
    "ALTER TABLE architecture_component_annotation_events "
    "RENAME TO architecture_component_annotation_events_v7",
    "ALTER TABLE architecture_component_annotations "
    "RENAME TO architecture_component_annotations_v7",
    """
    CREATE TABLE architecture_component_annotations (
        annotation_id TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_stable_key TEXT NOT NULL,
        note TEXT NOT NULL CHECK (length(trim(note)) BETWEEN 1 AND 4000),
        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(snapshot_id, component_stable_key)
    )
    """,
    """
    INSERT INTO architecture_component_annotations(
        annotation_id, snapshot_id, component_stable_key, note, version,
        actor, created_at, updated_at
    )
    SELECT annotations.annotation_id, annotations.snapshot_id,
           components.stable_key, annotations.note, annotations.version,
           annotations.actor, annotations.created_at, annotations.updated_at
    FROM architecture_component_annotations_v7 AS annotations
    JOIN semantic_components AS components
      ON components.snapshot_id = annotations.snapshot_id
     AND components.component_id = annotations.component_id
    """,
    """
    CREATE TABLE architecture_component_annotation_events (
        annotation_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        annotation_id TEXT NOT NULL
            REFERENCES architecture_component_annotations(annotation_id)
            ON DELETE CASCADE,
        snapshot_id TEXT NOT NULL REFERENCES repository_snapshots(snapshot_id)
            ON DELETE CASCADE,
        component_stable_key TEXT NOT NULL,
        note TEXT NOT NULL CHECK (length(trim(note)) BETWEEN 1 AND 4000),
        version INTEGER NOT NULL CHECK (version >= 1),
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    INSERT INTO architecture_component_annotation_events(
        annotation_event_id, annotation_id, snapshot_id,
        component_stable_key, note, version, actor, created_at
    )
    SELECT events.annotation_event_id, events.annotation_id, events.snapshot_id,
           components.stable_key, events.note, events.version,
           events.actor, events.created_at
    FROM architecture_component_annotation_events_v7 AS events
    JOIN semantic_components AS components
      ON components.snapshot_id = events.snapshot_id
     AND components.component_id = events.component_id
    """,
    "DROP TABLE architecture_component_annotation_events_v7",
    "DROP TABLE architecture_component_annotations_v7",
    "CREATE INDEX idx_architecture_annotations_snapshot "
    "ON architecture_component_annotations(snapshot_id, updated_at)",
    "CREATE INDEX idx_architecture_annotation_events_annotation "
    "ON architecture_component_annotation_events(annotation_id, version)",
)

MIGRATIONS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: ("initial durable application ledger", _MIGRATION_1),
    2: ("durable finding review lifecycle", _MIGRATION_2),
    3: ("durable local settings and review presets", _MIGRATION_3),
    4: ("evidence-derived semantic architecture ledger", _MIGRATION_4),
    5: ("semantic extractor projection version state", _MIGRATION_5),
    6: ("immutable redacted specialist prompt artifacts", _MIGRATION_6),
    7: ("durable architecture component annotations", _MIGRATION_7),
    8: ("stable architecture annotation identity", _MIGRATION_8),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _apply_pragmas(connection: sqlite3.Connection, *, busy_timeout_ms: int) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA wal_autocheckpoint = 1000")


def open_database(
    database_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    allow_cross_thread: bool = False,
) -> sqlite3.Connection:
    """Open one execution-context connection to the canonical database file."""

    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=busy_timeout_ms / 1000,
        isolation_level=None,
        check_same_thread=not allow_cross_thread,
    )
    connection.row_factory = sqlite3.Row
    _apply_pragmas(connection, busy_timeout_ms=busy_timeout_ms)
    return connection


@contextmanager
def database_connection(
    database_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> Iterator[sqlite3.Connection]:
    connection = open_database(database_path, busy_timeout_ms=busy_timeout_ms)
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(
    database_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> int:
    """Create or upgrade the one application database in place."""

    with database_connection(database_path, busy_timeout_ms=busy_timeout_ms) as connection:
        # journal_mode is persistent database state. Set it during the serialized
        # initialization path rather than repeating the disk-level negotiation on
        # every short-lived repository connection.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            current_version = int(row["version"])
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current_version} is newer than "
                    f"supported schema {SCHEMA_VERSION}"
                )

            applied_migrations = connection.execute(
                """
                SELECT version, description, checksum
                FROM schema_migrations
                WHERE version <= ?
                ORDER BY version
                """,
                (SCHEMA_VERSION,),
            ).fetchall()
            for applied in applied_migrations:
                version = int(applied["version"])
                expected_description, statements = MIGRATIONS[version]
                expected_checksum = hashlib.sha256(
                    "\n".join(statements).encode("utf-8")
                ).hexdigest()
                if (
                    applied["description"] != expected_description
                    or applied["checksum"] != expected_checksum
                ):
                    raise RuntimeError(
                        f"database migration {version} does not match the canonical schema source"
                    )

            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                description, statements = MIGRATIONS[version]
                for statement in statements:
                    connection.execute(statement)
                checksum = hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()
                connection.execute(
                    """
                    INSERT INTO schema_migrations(
                        version, description, checksum, applied_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (version, description, checksum, _utc_now()),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return SCHEMA_VERSION


def checkpoint_database(
    database_path: str | Path, *, truncate: bool = False
) -> tuple[int, int, int]:
    """Checkpoint the WAL during explicit maintenance or clean shutdown."""

    mode = "TRUNCATE" if truncate else "PASSIVE"
    with database_connection(database_path) as connection:
        row = connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return int(row[0]), int(row[1]), int(row[2])
