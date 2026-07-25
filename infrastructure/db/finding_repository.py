"""Finding query, evidence, and review lifecycle projections."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import database_connection
from .run_ledger import SqliteRunLedger

REVIEW_STATES = frozenset(
    {"new", "validated", "acknowledged", "reviewed", "dismissed", "reopened", "resolved"}
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteFindingRepository:
    """Read canonical findings and mutate only their local review metadata."""

    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def query(
        self,
        *,
        run_id: str | None = None,
        search: str | None = None,
        severity: str | None = None,
        review_state: str | None = None,
        min_confidence: float | None = None,
        affected_path: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        clauses = ["1 = 1"]
        values: list[Any] = []
        if run_id:
            clauses.append("findings.run_id = ?")
            values.append(run_id)
        if search:
            clauses.append(
                "(LOWER(json_extract(findings.contract_json, '$.title')) LIKE ? "
                "OR LOWER(json_extract(findings.contract_json, '$.claim')) LIKE ?)"
            )
            pattern = f"%{search.lower()}%"
            values.extend((pattern, pattern))
        if severity:
            clauses.append("json_extract(findings.contract_json, '$.severity') = ?")
            values.append(severity)
        if review_state:
            clauses.append("COALESCE(reviews.review_state, 'new') = ?")
            values.append(review_state)
        if min_confidence is not None:
            clauses.append(
                "CAST(json_extract(findings.contract_json, '$.confidence') AS REAL) >= ?"
            )
            values.append(min_confidence)
        if affected_path:
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM canonical_finding_members AS members
                    JOIN finding_candidates AS candidates
                      ON candidates.candidate_id = members.candidate_id
                    WHERE members.finding_id = findings.finding_id
                      AND LOWER(json_extract(candidates.contract_json, '$.affected_path'))
                          LIKE ?
                )"""
            )
            values.append(f"%{affected_path.lower()}%")
        if cursor:
            clauses.append("findings.finding_id > ?")
            values.append(cursor)
        where = " AND ".join(clauses)
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT findings.finding_id, findings.run_id, findings.contract_json,
                       COALESCE(reviews.review_state, 'new') AS review_state,
                       reviews.note, reviews.version, reviews.updated_at
                FROM canonical_findings AS findings
                LEFT JOIN finding_reviews AS reviews USING(finding_id)
                WHERE {where}
                ORDER BY findings.finding_id
                LIMIT ?
                """,
                (*values, limit + 1),
            ).fetchall()
            summary_rows = connection.execute(
                f"""
                SELECT json_extract(findings.contract_json, '$.severity') AS severity,
                       COUNT(*) AS count
                FROM canonical_findings AS findings
                LEFT JOIN finding_reviews AS reviews USING(finding_id)
                WHERE {where}
                GROUP BY severity
                """,
                tuple(values),
            ).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        return {
            "items": [self._summary(row) for row in page],
            "next_cursor": str(page[-1]["finding_id"]) if has_more else None,
            "counts_by_severity": {str(row["severity"]): int(row["count"]) for row in summary_rows},
        }

    def detail(self, finding_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT findings.finding_id, findings.run_id, findings.contract_json,
                       COALESCE(reviews.review_state, 'new') AS review_state,
                       reviews.note, reviews.version, reviews.updated_at
                FROM canonical_findings AS findings
                LEFT JOIN finding_reviews AS reviews USING(finding_id)
                WHERE findings.finding_id = ?
                """,
                (finding_id,),
            ).fetchone()
            if row is None:
                return None
            candidates = connection.execute(
                """
                SELECT members.relationship, candidates.contract_json,
                       roles.contract_json AS role_json,
                       verdicts.contract_json AS verdict_json
                FROM canonical_finding_members AS members
                JOIN finding_candidates AS candidates USING(candidate_id)
                LEFT JOIN role_specs AS roles ON roles.role_id = candidates.role_id
                LEFT JOIN finding_verdicts AS verdicts
                  ON verdicts.verdict_id = (
                    SELECT latest.verdict_id FROM finding_verdicts AS latest
                    WHERE latest.candidate_id = candidates.candidate_id
                    ORDER BY latest.created_at DESC, latest.verdict_id DESC LIMIT 1
                  )
                WHERE members.finding_id = ?
                ORDER BY candidates.candidate_id
                """,
                (finding_id,),
            ).fetchall()
            evidence = connection.execute(
                """
                SELECT DISTINCT evidence.*, files.relative_path,
                       artifacts.storage_path
                FROM canonical_finding_members AS members
                JOIN finding_candidate_evidence AS links USING(candidate_id)
                JOIN evidence_refs AS evidence USING(evidence_id)
                JOIN files USING(file_id)
                LEFT JOIN artifacts USING(artifact_id)
                WHERE members.finding_id = ?
                ORDER BY files.relative_path, evidence.start_line
                """,
                (finding_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT review_event_id, previous_state, review_state, note,
                       actor, created_at
                FROM finding_review_events WHERE finding_id = ?
                ORDER BY review_event_id
                """,
                (finding_id,),
            ).fetchall()
        result = self._summary(row)
        result["candidates"] = [
            {
                "relationship": candidate["relationship"],
                "candidate": json.loads(candidate["contract_json"]),
                "role": json.loads(candidate["role_json"]) if candidate["role_json"] else None,
                "verdict": (
                    json.loads(candidate["verdict_json"]) if candidate["verdict_json"] else None
                ),
            }
            for candidate in candidates
        ]
        result["evidence"] = [self._evidence(item) for item in evidence]
        result["review_history"] = [dict(item) for item in events]
        return result

    def set_review_state(
        self,
        finding_id: str,
        *,
        review_state: str,
        note: str | None,
        expected_version: int | None,
        actor: str = "local-user",
    ) -> dict[str, Any] | None:
        if review_state not in REVIEW_STATES:
            raise ValueError("unsupported finding review state")
        timestamp = _now()

        def operation(connection):
            finding = connection.execute(
                "SELECT 1 FROM canonical_findings WHERE finding_id = ?", (finding_id,)
            ).fetchone()
            if finding is None:
                return None
            current = connection.execute(
                "SELECT review_state, version FROM finding_reviews WHERE finding_id = ?",
                (finding_id,),
            ).fetchone()
            current_version = int(current["version"]) if current else 0
            if expected_version is not None and expected_version != current_version:
                raise RuntimeError("finding review state changed; refresh and retry")
            previous = str(current["review_state"]) if current else None
            next_version = current_version + 1
            connection.execute(
                """
                INSERT INTO finding_reviews(
                    finding_id, review_state, note, version, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    review_state = excluded.review_state,
                    note = excluded.note,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (finding_id, review_state, note, next_version, timestamp),
            )
            connection.execute(
                """
                INSERT INTO finding_review_events(
                    finding_id, previous_state, review_state, note, actor, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (finding_id, previous, review_state, note, actor, timestamp),
            )
            return True

        changed = self.ledger.write_transaction(operation)
        return self.detail(finding_id) if changed else None

    @staticmethod
    def _summary(row) -> dict[str, Any]:
        finding = json.loads(row["contract_json"])
        return {
            **finding,
            "run_id": str(row["run_id"]),
            "review_state": str(row["review_state"]),
            "review_note": row["note"],
            "review_version": int(row["version"] or 0),
            "reviewed_at": row["updated_at"],
        }

    @staticmethod
    def _evidence(row) -> dict[str, Any]:
        excerpt: str | None = None
        integrity = "unavailable"
        storage_path = row["storage_path"]
        if storage_path:
            try:
                content = Path(storage_path).read_bytes()
                integrity = (
                    "valid"
                    if hashlib.sha256(content).hexdigest() == row["content_hash"]
                    else "invalid"
                )
                if integrity == "valid":
                    lines = content.decode("utf-8", errors="replace").splitlines()
                    excerpt = "\n".join(lines[int(row["start_line"]) - 1 : int(row["end_line"])])
                    if hashlib.sha256(excerpt.encode()).hexdigest() != row["excerpt_hash"]:
                        excerpt = None
                        integrity = "invalid"
            except OSError:
                integrity = "unavailable"
        excerpt_truncated = excerpt is not None and len(excerpt) > 16_000
        if excerpt_truncated:
            excerpt = f"{excerpt[:16_000]}\n… excerpt truncated …"
        return {
            "evidence_id": row["evidence_id"],
            "relative_path": row["relative_path"],
            "content_hash": row["content_hash"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "start_byte": row["start_byte"],
            "end_byte": row["end_byte"],
            "evidence_kind": row["evidence_kind"],
            "provenance": json.loads(row["provenance_json"]),
            "excerpt": excerpt,
            "integrity": integrity,
            "redacted": excerpt is None,
            "truncated": excerpt_truncated,
        }
