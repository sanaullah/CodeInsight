from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from application.analysis_service import AnalysisService
from infrastructure.db.finding_repository import SqliteFindingRepository
from infrastructure.db.run_ledger import SqliteRunLedger


def _seed(ledger: SqliteRunLedger, tmp_path: Path) -> None:
    source = b"def authorize(user):\n    return True\n"
    source_path = tmp_path / "source"
    source_path.write_bytes(source)
    excerpt = b"    return True"
    now = "2026-07-25T12:00:00+00:00"
    finding = {
        "schema_version": 1,
        "finding_id": "finding-1",
        "fingerprint": "auth",
        "title": "Authorization is bypassed",
        "claim": "The handler permits every user.",
        "severity": "high",
        "confidence": 0.94,
        "candidate_ids": ["candidate-1"],
        "supporting_evidence_ids": ["evidence-1"],
        "conflicting_candidate_ids": [],
        "recommendation": "Enforce policy before returning.",
    }
    candidate = {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "producer_task_id": "task-1",
        "producer_role_id": "role-1",
        "category": "security",
        "concept_id": "authorization",
        "title": finding["title"],
        "claim": finding["claim"],
        "evidence_ids": ["evidence-1"],
        "preconditions": [],
        "affected_path": "src/auth.py",
        "impact": "Unauthorized access",
        "proposed_severity": "high",
        "proposed_confidence": 0.94,
        "uncertainty": None,
        "missing_context": [],
        "recommendation": finding["recommendation"],
        "fingerprint_inputs": ["auth"],
    }
    verdict = {
        "schema_version": 1,
        "verdict_id": "verdict-1",
        "candidate_id": "candidate-1",
        "disposition": "accepted",
        "evidence_integrity": "valid",
        "contradiction_result": "none",
        "reachability_result": "reachable",
        "calibrated_confidence": 0.94,
        "calibrated_severity": "high",
        "verifier": "native",
        "verifier_version": "1",
        "rationale": "The source span directly supports the claim.",
    }

    def operation(connection):
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            ("project-1", str(tmp_path), "fixture", now, now),
        )
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, '{}', ?)",
            (
                "artifact-1",
                hashlib.sha256(source).hexdigest(),
                "source",
                str(source_path),
                len(source),
                "text/plain",
                now,
            ),
        )
        connection.execute(
            """INSERT INTO repository_snapshots VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)""",
            ("snapshot-1", "project-1", "identity", "config", "1", None, None, None, 0, now),
        )
        connection.execute(
            """INSERT INTO files VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')""",
            (
                "file-1",
                "snapshot-1",
                "src/auth.py",
                hashlib.sha256(source).hexdigest(),
                "python",
                "source",
                "parsed",
                len(source),
                2,
                "artifact-1",
            ),
        )
        connection.execute(
            """INSERT INTO runs(run_id, submission_key, snapshot_id, status, mode,
               request_json, budget_json, created_at, updated_at)
               VALUES (?, ?, ?, 'succeeded', 'security', '{}', '{}', ?, ?)""",
            ("run-1", "submission-1", "snapshot-1", now, now),
        )
        connection.execute(
            "INSERT INTO waves VALUES (?, ?, 1, ?, 'succeeded', '{}', ?, ?)",
            ("wave-1", "run-1", "security review", now, now),
        )
        connection.execute(
            "INSERT INTO role_specs VALUES (?, ?, ?, ?, 1, ?)",
            (
                "role-1",
                "run-1",
                "wave-1",
                json.dumps(
                    {
                        "schema_version": 1,
                        "role_id": "role-1",
                        "name": "Security specialist",
                        "mission": "Review authorization",
                        "rationale": "security mode",
                        "coverage_targets": ["src/auth.py"],
                        "required_capabilities": [],
                        "model_policy": "balanced",
                    }
                ),
                now,
            ),
        )
        connection.execute(
            """INSERT INTO tasks(task_id, run_id, wave_id, role_id, task_type,
               status, idempotency_key, input_json, budget_json,
               routing_policy_json, available_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'specialist_analysis', 'succeeded', ?, '{}',
                       '{}', '{}', ?, ?, ?)""",
            ("task-1", "run-1", "wave-1", "role-1", "task-key", now, now, now),
        )
        connection.execute(
            """INSERT INTO evidence_refs VALUES
               (?, ?, ?, ?, 2, 2, NULL, NULL, ?, 'source', ?, ?)""",
            (
                "evidence-1",
                "snapshot-1",
                "file-1",
                hashlib.sha256(source).hexdigest(),
                hashlib.sha256(excerpt).hexdigest(),
                json.dumps({"run_id": "run-1"}),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO finding_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("candidate-1", "run-1", "task-1", "role-1", "auth", json.dumps(candidate), 1, now),
        )
        connection.execute(
            "INSERT INTO finding_candidate_evidence VALUES (?, ?)",
            ("candidate-1", "evidence-1"),
        )
        connection.execute(
            "INSERT INTO finding_verdicts VALUES (?, ?, ?, 1, ?)",
            ("verdict-1", "candidate-1", json.dumps(verdict), now),
        )
        connection.execute(
            "INSERT INTO canonical_findings VALUES (?, ?, ?, ?, 1, ?)",
            ("finding-1", "run-1", "auth", json.dumps(finding), now),
        )
        connection.execute(
            "INSERT INTO canonical_finding_members VALUES (?, ?, 'supporting')",
            ("finding-1", "candidate-1"),
        )

    ledger.write_transaction(operation)


def test_query_detail_evidence_integrity_and_review_audit(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    _seed(ledger, tmp_path)
    repository = SqliteFindingRepository(ledger)

    page = repository.query(search="authorization", severity="high", limit=1)
    assert page["next_cursor"] is None
    assert page["counts_by_severity"] == {"high": 1}
    assert page["items"][0]["review_state"] == "new"

    detail = repository.detail("finding-1")
    assert detail is not None
    assert detail["evidence"][0]["excerpt"] == "    return True"
    assert detail["evidence"][0]["integrity"] == "valid"
    assert detail["candidates"][0]["verdict"]["disposition"] == "accepted"

    (tmp_path / "source").write_bytes(b"tampered")
    tampered = repository.detail("finding-1")
    assert tampered is not None
    assert tampered["evidence"][0]["integrity"] == "invalid"
    assert tampered["evidence"][0]["excerpt"] is None

    updated = repository.set_review_state(
        "finding-1", review_state="reviewed", note="Confirmed locally", expected_version=0
    )
    assert updated is not None
    assert updated["review_state"] == "reviewed"
    assert updated["review_version"] == 1
    assert updated["review_history"][0]["previous_state"] is None
    with pytest.raises(RuntimeError, match="refresh"):
        repository.set_review_state(
            "finding-1", review_state="dismissed", note=None, expected_version=0
        )
    ledger.close()


def test_finding_api_filters_mutates_and_exports(tmp_path: Path) -> None:
    service = AnalysisService(database_path=tmp_path / "codeinsight.db")
    ledger = service._get_ledger()
    _seed(ledger, tmp_path)
    app = create_app(service)

    with TestClient(app) as client:
        page = client.get("/api/v1/findings?severity=high&limit=50")
        assert page.status_code == 200
        assert page.json()["items"][0]["finding_id"] == "finding-1"
        detail = client.get("/api/v1/findings/finding-1")
        assert detail.json()["evidence"][0]["integrity"] == "valid"
        updated = client.put(
            "/api/v1/findings/finding-1/review",
            json={"review_state": "acknowledged", "expected_version": 0},
        )
        assert updated.status_code == 200
        assert updated.json()["review_state"] == "acknowledged"
        conflict = client.put(
            "/api/v1/findings/finding-1/review",
            json={"review_state": "dismissed", "expected_version": 0},
        )
        assert conflict.status_code == 409
        exported = client.get("/api/v1/findings-export?format=csv")
        assert exported.status_code == 200
        assert "attachment" in exported.headers["content-disposition"]
        assert "Authorization is bypassed" in exported.text
        assert client.get("/api/v1/findings/missing").status_code == 404


def test_finding_query_stays_bounded_with_one_thousand_records(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    _seed(ledger, tmp_path)

    def operation(connection):
        records = []
        now = "2026-07-25T12:00:00+00:00"
        for index in range(2, 1002):
            finding = {
                "schema_version": 1,
                "finding_id": f"finding-{index:04}",
                "fingerprint": f"fingerprint-{index:04}",
                "title": f"Fixture finding {index}",
                "claim": "A bounded performance fixture.",
                "severity": "medium",
                "confidence": 0.8,
                "candidate_ids": [],
                "supporting_evidence_ids": [],
                "conflicting_candidate_ids": [],
                "recommendation": "Inspect the fixture.",
            }
            records.append(
                (
                    finding["finding_id"],
                    "run-1",
                    finding["fingerprint"],
                    json.dumps(finding),
                    now,
                )
            )
        connection.executemany(
            """INSERT INTO canonical_findings(
                   finding_id, run_id, fingerprint, contract_json,
                   contract_version, created_at
               ) VALUES (?, ?, ?, ?, 1, ?)""",
            records,
        )

    ledger.write_transaction(operation)
    repository = SqliteFindingRepository(ledger)
    started = time.perf_counter()
    page = repository.query(severity="medium", limit=50)
    elapsed = time.perf_counter() - started

    assert len(page["items"]) == 50
    assert page["next_cursor"] is not None
    assert page["counts_by_severity"] == {"medium": 1000}
    assert elapsed < 1.0
    ledger.close()
