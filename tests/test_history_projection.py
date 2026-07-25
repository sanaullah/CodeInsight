from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from application.analysis_service import AnalysisService
from infrastructure.db.history_repository import SqliteHistoryRepository
from infrastructure.db.run_ledger import SqliteRunLedger


def _finding(finding_id: str, fingerprint: str, severity: str) -> dict:
    return {
        "schema_version": 1,
        "finding_id": finding_id,
        "fingerprint": fingerprint,
        "title": f"Finding {fingerprint}",
        "claim": "Persisted comparison fixture.",
        "severity": severity,
        "confidence": 0.9,
        "candidate_ids": [],
        "supporting_evidence_ids": [],
        "conflicting_candidate_ids": [],
        "recommendation": "Inspect.",
    }


def _seed(ledger: SqliteRunLedger, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    earlier = now.replace(microsecond=0).isoformat()
    later = now.isoformat()

    def operation(connection):
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            ("project-1", str(tmp_path), "history-fixture", earlier, later),
        )
        connection.execute(
            """INSERT INTO repository_snapshots(
                   snapshot_id, project_id, identity_hash, configuration_hash,
                   scanner_version, git_repository, base_commit, head_commit,
                   dirty, metadata_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, '{}', ?)""",
            (
                "snapshot-1",
                "project-1",
                "identity",
                "config",
                "1",
                "git",
                "base",
                "head",
                earlier,
            ),
        )
        connection.executemany(
            """INSERT INTO runs(
                   run_id, submission_key, snapshot_id, status, mode,
                   request_json, budget_json, created_at, started_at,
                   completed_at, updated_at
               ) VALUES (?, ?, 'snapshot-1', 'succeeded', ?, '{}', '{}',
                         ?, ?, ?, ?)""",
            [
                ("run-001", "submission-1", "deep", earlier, earlier, later, later),
                ("run-002", "submission-2", "security", later, later, later, later),
            ],
        )
        findings = [
            ("base-stable", "run-001", "stable", _finding("base-stable", "stable", "medium")),
            ("base-resolved", "run-001", "resolved", _finding("base-resolved", "resolved", "high")),
            ("target-stable", "run-002", "stable", _finding("target-stable", "stable", "high")),
            ("target-new", "run-002", "new", _finding("target-new", "new", "low")),
        ]
        connection.executemany(
            """INSERT INTO canonical_findings(
                   finding_id, run_id, fingerprint, contract_json,
                   contract_version, created_at
               ) VALUES (?, ?, ?, ?, 1, ?)""",
            [
                (finding_id, run_id, fingerprint, json.dumps(contract), later)
                for finding_id, run_id, fingerprint, contract in findings
            ],
        )
        connection.execute(
            """INSERT INTO finding_reviews VALUES
               ('target-new', 'reopened', NULL, 1, ?)""",
            (later,),
        )
        connection.execute(
            """INSERT INTO finding_review_events(
                   finding_id, previous_state, review_state, actor, created_at
               ) VALUES ('target-new', 'resolved', 'reopened', 'local-user', ?)""",
            (later,),
        )

    ledger.write_transaction(operation)


def test_history_query_comparison_and_trends_are_deterministic(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    _seed(ledger, tmp_path)
    repository = SqliteHistoryRepository(ledger)

    page = repository.query(search="history", status="succeeded", limit=1)
    assert len(page["items"]) == 1
    assert page["next_cursor"] is not None
    assert page["items"][0]["display_name"] == "history-fixture"
    comparison = repository.compare("run-001", "run-002")
    assert comparison is not None
    assert [item["fingerprint"] for item in comparison["new"]] == ["new"]
    assert [item["fingerprint"] for item in comparison["resolved"]] == ["resolved"]
    assert comparison["severity_moved"][0] == {
        "fingerprint": "stable",
        "title": "Finding stable",
        "from_severity": "medium",
        "to_severity": "high",
    }
    assert comparison["reopened"][0]["finding_id"] == "target-new"
    assert repository.compare("missing", "run-002") is None
    trends = repository.trends(30)
    assert trends["partial"] is False
    assert sum(bucket["review_count"] for bucket in trends["buckets"]) == 2
    assert sum(bucket["finding_count"] for bucket in trends["buckets"]) == 4
    ledger.close()


def test_history_api_contracts(tmp_path: Path) -> None:
    service = AnalysisService(database_path=tmp_path / "codeinsight.db")
    _seed(service._get_ledger(), tmp_path)
    app = create_app(service)

    with TestClient(app) as client:
        history = client.get("/api/v1/history?mode=security&limit=50")
        assert history.status_code == 200
        assert history.json()["items"][0]["run_id"] == "run-002"
        comparison = client.get(
            "/api/v1/history/compare?baseline_run_id=run-001&target_run_id=run-002"
        )
        assert comparison.status_code == 200
        assert len(comparison.json()["new"]) == 1
        assert client.get("/api/v1/history/trends?days=30").status_code == 200
        assert (
            client.get(
                "/api/v1/history/compare?baseline_run_id=missing&target_run_id=run-002"
            ).status_code
            == 404
        )
        assert client.get("/api/v1/history?limit=0").status_code == 422
