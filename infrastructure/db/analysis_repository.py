"""Durable typed analysis records over the canonical application schema."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from domain.contracts import (
    CanonicalFinding,
    CoverageAssessment,
    EvidenceRef,
    FindingCandidate,
    FindingVerdict,
    RoleSpec,
)

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteAnalysisRepository:
    """Persist evidence and findings atomically without defining schema."""

    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def bind_snapshot(self, run_id: str, snapshot_id: str) -> bool:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE runs SET snapshot_id = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                  AND (snapshot_id IS NULL OR snapshot_id = ?)
                """,
                (snapshot_id, timestamp, run_id, snapshot_id),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)

    def acquire_coordinator_lease(
        self,
        run_id: str,
        *,
        owner: str,
        lease_seconds: float = 300,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=lease_seconds)

        def operation(connection: sqlite3.Connection) -> bool:
            run = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND status = 'running'",
                (run_id,),
            ).fetchone()
            if run is None:
                return False
            cursor = connection.execute(
                """
                INSERT INTO run_stages(
                    run_stage_id, run_id, stage, stage_version, status,
                    input_json, attempt_count, lease_owner, lease_expires_at,
                    started_at
                ) VALUES (?, ?, 'native_analysis', 'v1', 'running', '{}',
                          1, ?, ?, ?)
                ON CONFLICT(run_id, stage, stage_version) DO UPDATE SET
                    status = 'running',
                    attempt_count = run_stages.attempt_count + 1,
                    lease_owner = excluded.lease_owner,
                    lease_expires_at = excluded.lease_expires_at,
                    started_at = excluded.started_at,
                    completed_at = NULL,
                    error_json = NULL
                WHERE run_stages.status != 'running'
                   OR run_stages.lease_expires_at <= ?
                """,
                (
                    f"native-analysis:{run_id}",
                    run_id,
                    owner,
                    expires.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)

    def release_coordinator_lease(
        self,
        run_id: str,
        *,
        owner: str,
        status: str,
        error: str | None = None,
    ) -> bool:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> bool:
            cursor = connection.execute(
                """
                UPDATE run_stages
                SET status = ?, completed_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL, error_json = ?
                WHERE run_id = ? AND stage = 'native_analysis'
                  AND stage_version = 'v1' AND lease_owner = ?
                """,
                (
                    status,
                    timestamp,
                    json.dumps({"message": error}) if error else None,
                    run_id,
                    owner,
                ),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)

    def get_role(self, role_id: str) -> RoleSpec | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT contract_json FROM role_specs WHERE role_id = ?",
                (role_id,),
            ).fetchone()
            return RoleSpec.model_validate_json(row["contract_json"]) if row else None

    def persist_specialist_result(
        self,
        *,
        run_id: str,
        task_id: str,
        evidence: Sequence[EvidenceRef],
        candidates: Sequence[FindingCandidate],
        verdicts: Sequence[FindingVerdict],
        fingerprints: dict[str, str],
        analyzed_paths: Sequence[str],
        unresolved_uncertainty: Sequence[str],
    ) -> None:
        timestamp = _now()
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        evidence_ids = {item.evidence_id for item in evidence}
        if set(fingerprints) != set(candidate_by_id):
            raise ValueError("every candidate must have exactly one fingerprint")
        if any(
            evidence_id not in evidence_ids
            for candidate in candidates
            for evidence_id in candidate.evidence_ids
        ):
            raise ValueError("candidate references evidence outside this result")
        if any(
            verdict.candidate_id not in candidate_by_id for verdict in verdicts
        ):
            raise ValueError("verdict references an unknown candidate")

        def operation(connection: sqlite3.Connection) -> None:
            connection.executemany(
                """
                INSERT INTO evidence_refs(
                    evidence_id, snapshot_id, file_id, content_hash,
                    start_line, end_line, start_byte, end_byte, excerpt_hash,
                    evidence_kind, provenance_json, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO NOTHING
                """,
                [
                    (
                        item.evidence_id,
                        item.snapshot_id,
                        item.file_id,
                        item.content_hash,
                        item.start_line,
                        item.end_line,
                        item.start_byte,
                        item.end_byte,
                        item.excerpt_hash,
                        item.evidence_kind,
                        json.dumps(
                            item.provenance,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        item.collected_at.isoformat(),
                    )
                    for item in evidence
                ],
            )
            connection.execute(
                """
                INSERT INTO experience_records(
                    experience_id, run_id, task_id, record_json,
                    promotion_status, created_at
                ) VALUES (?, ?, ?, ?, 'unreviewed', ?)
                ON CONFLICT(experience_id) DO UPDATE SET
                    record_json = excluded.record_json
                """,
                (
                    f"specialist-summary:{task_id}",
                    run_id,
                    task_id,
                    json.dumps(
                        {
                            "record_type": "specialist-summary",
                            "analyzed_paths": sorted(set(analyzed_paths)),
                            "unresolved_uncertainty": sorted(
                                set(unresolved_uncertainty)
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    timestamp,
                ),
            )
            connection.executemany(
                """
                INSERT INTO finding_candidates(
                    candidate_id, run_id, task_id, role_id, fingerprint,
                    contract_json, contract_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO NOTHING
                """,
                [
                    (
                        item.candidate_id,
                        run_id,
                        item.producer_task_id,
                        item.producer_role_id,
                        fingerprints[item.candidate_id],
                        item.model_dump_json(),
                        item.schema_version,
                        timestamp,
                    )
                    for item in candidates
                ],
            )
            connection.executemany(
                """
                INSERT INTO finding_candidate_evidence(candidate_id, evidence_id)
                VALUES (?, ?)
                ON CONFLICT(candidate_id, evidence_id) DO NOTHING
                """,
                [
                    (candidate.candidate_id, evidence_id)
                    for candidate in candidates
                    for evidence_id in candidate.evidence_ids
                ],
            )
            connection.executemany(
                """
                INSERT INTO finding_verdicts(
                    verdict_id, candidate_id, contract_json,
                    contract_version, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(verdict_id) DO NOTHING
                """,
                [
                    (
                        item.verdict_id,
                        item.candidate_id,
                        item.model_dump_json(),
                        item.schema_version,
                        timestamp,
                    )
                    for item in verdicts
                ],
            )

        self.ledger.write_transaction(operation)

    def accepted_candidates(
        self, run_id: str
    ) -> list[tuple[FindingCandidate, FindingVerdict, str]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                WITH latest_verdict AS (
                    SELECT candidate_id, contract_json,
                           ROW_NUMBER() OVER (
                               PARTITION BY candidate_id ORDER BY created_at DESC
                           ) AS rank
                    FROM finding_verdicts
                )
                SELECT candidates.contract_json AS candidate_json,
                       candidates.fingerprint,
                       latest_verdict.contract_json AS verdict_json
                FROM finding_candidates AS candidates
                JOIN latest_verdict
                  ON latest_verdict.candidate_id = candidates.candidate_id
                 AND latest_verdict.rank = 1
                WHERE candidates.run_id = ?
                ORDER BY candidates.fingerprint, candidates.candidate_id
                """,
                (run_id,),
            ).fetchall()
            result = []
            for row in rows:
                verdict = FindingVerdict.model_validate_json(row["verdict_json"])
                if verdict.disposition == "accepted":
                    result.append(
                        (
                            FindingCandidate.model_validate_json(
                                row["candidate_json"]
                            ),
                            verdict,
                            str(row["fingerprint"]),
                        )
                    )
            return result

    def replace_canonical_findings(
        self,
        run_id: str,
        findings: Sequence[CanonicalFinding],
    ) -> None:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                "DELETE FROM canonical_findings WHERE run_id = ?", (run_id,)
            )
            for finding in findings:
                connection.execute(
                    """
                    INSERT INTO canonical_findings(
                        finding_id, run_id, fingerprint, contract_json,
                        contract_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        finding.finding_id,
                        run_id,
                        finding.fingerprint,
                        finding.model_dump_json(),
                        finding.schema_version,
                        timestamp,
                    ),
                )
                supporting = set(finding.candidate_ids)
                conflicting = set(finding.conflicting_candidate_ids)
                connection.executemany(
                    """
                    INSERT INTO canonical_finding_members(
                        finding_id, candidate_id, relationship
                    ) VALUES (?, ?, ?)
                    """,
                    [
                        (
                            finding.finding_id,
                            candidate_id,
                            "conflicting"
                            if candidate_id in conflicting
                            else "supporting",
                        )
                        for candidate_id in sorted(supporting | conflicting)
                    ],
                )

        self.ledger.write_transaction(operation)

    def persist_coverage(self, assessment: CoverageAssessment) -> None:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO coverage_assessments(
                    assessment_id, run_id, wave_id, contract_json,
                    contract_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    contract_json = excluded.contract_json,
                    contract_version = excluded.contract_version
                """,
                (
                    assessment.assessment_id,
                    assessment.run_id,
                    assessment.wave_id,
                    assessment.model_dump_json(),
                    assessment.schema_version,
                    timestamp,
                ),
            )

        self.ledger.write_transaction(operation)

    def complete_wave(self, wave_id: str, *, status: str = "completed") -> None:
        timestamp = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                UPDATE waves SET status = ?, completed_at = ?
                WHERE wave_id = ?
                """,
                (status, timestamp, wave_id),
            )

        self.ledger.write_transaction(operation)

    def wave_count(self, run_id: str) -> int:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM waves WHERE run_id = ?", (run_id,)
            ).fetchone()
            return int(row["count"])

    def task_records(self, run_id: str, wave_id: str) -> list[dict[str, Any]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT task_id, role_id, status, input_json, error_json
                FROM tasks WHERE run_id = ? AND wave_id = ?
                ORDER BY task_id
                """,
                (run_id, wave_id),
            ).fetchall()
            return [
                {
                    "task_id": str(row["task_id"]),
                    "role_id": str(row["role_id"]) if row["role_id"] else None,
                    "status": str(row["status"]),
                    "input": json.loads(row["input_json"]),
                    "error": json.loads(row["error_json"])
                    if row["error_json"]
                    else None,
                }
                for row in rows
            ]

    def wave_evidence_file_ids(self, wave_id: str) -> tuple[str, ...]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT evidence.file_id
                FROM tasks
                JOIN finding_candidates AS candidates
                  ON candidates.task_id = tasks.task_id
                JOIN finding_candidate_evidence AS links
                  ON links.candidate_id = candidates.candidate_id
                JOIN evidence_refs AS evidence
                  ON evidence.evidence_id = links.evidence_id
                WHERE tasks.wave_id = ?
                ORDER BY evidence.file_id
                """,
                (wave_id,),
            ).fetchall()
            return tuple(str(row["file_id"]) for row in rows)

    def wave_analysis_summary(self, wave_id: str) -> dict[str, tuple[str, ...]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT records.record_json
                FROM experience_records AS records
                JOIN tasks ON tasks.task_id = records.task_id
                WHERE tasks.wave_id = ?
                  AND records.promotion_status = 'unreviewed'
                """,
                (wave_id,),
            ).fetchall()
            records = [json.loads(row["record_json"]) for row in rows]
            return {
                "analyzed_paths": tuple(
                    sorted(
                        {
                            str(path)
                            for record in records
                            for path in record.get("analyzed_paths", ())
                        }
                    )
                ),
                "unresolved_uncertainty": tuple(
                    sorted(
                        {
                            str(item)
                            for record in records
                            for item in record.get("unresolved_uncertainty", ())
                        }
                    )
                ),
            }

    def wave_verdict_counts(self, wave_id: str) -> tuple[int, int]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                WITH latest AS (
                    SELECT verdicts.candidate_id, verdicts.contract_json,
                           ROW_NUMBER() OVER (
                               PARTITION BY verdicts.candidate_id
                               ORDER BY verdicts.created_at DESC
                           ) AS rank
                    FROM finding_verdicts AS verdicts
                    JOIN finding_candidates AS candidates
                      ON candidates.candidate_id = verdicts.candidate_id
                    JOIN tasks ON tasks.task_id = candidates.task_id
                    WHERE tasks.wave_id = ?
                )
                SELECT contract_json FROM latest WHERE rank = 1
                """,
                (wave_id,),
            ).fetchall()
            verdicts = [
                FindingVerdict.model_validate_json(row["contract_json"])
                for row in rows
            ]
            return (
                len(verdicts),
                sum(verdict.disposition == "accepted" for verdict in verdicts),
            )

    def list_canonical_findings(self, run_id: str) -> list[CanonicalFinding]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT contract_json FROM canonical_findings
                WHERE run_id = ? ORDER BY fingerprint
                """,
                (run_id,),
            ).fetchall()
            return [
                CanonicalFinding.model_validate_json(row["contract_json"])
                for row in rows
            ]

    def list_coverage(self, run_id: str) -> list[CoverageAssessment]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT contract_json FROM coverage_assessments
                WHERE run_id = ? ORDER BY created_at
                """,
                (run_id,),
            ).fetchall()
            return [
                CoverageAssessment.model_validate_json(row["contract_json"])
                for row in rows
            ]
