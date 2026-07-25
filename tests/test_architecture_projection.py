from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from application.analysis_service import AnalysisService
from infrastructure.db.architecture_repository import SqliteArchitectureRepository
from infrastructure.db.run_ledger import SqliteRunLedger


def _seed(ledger: SqliteRunLedger, tmp_path: Path) -> None:
    now = "2026-07-25T12:00:00+00:00"

    def operation(connection):
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            ("project-1", str(tmp_path), "architecture-fixture", now, now),
        )
        connection.execute(
            """INSERT INTO repository_snapshots VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "snapshot-1",
                "project-1",
                "identity",
                "config",
                "1",
                "git",
                "base",
                "head",
                1,
                json.dumps({"changed_paths": ["a.py"]}),
                now,
            ),
        )
        connection.executemany(
            """INSERT INTO files(
                   file_id, snapshot_id, relative_path, content_hash, language,
                   classification, support_tier, byte_size, line_count,
                   metadata_json
               ) VALUES (?, 'snapshot-1', ?, ?, 'python', 'source', 'parsed',
                         10, ?, '{}')""",
            [
                ("file-a", "a.py", "hash-a", 10),
                ("file-b", "b.py", "hash-b", 20),
                ("file-c", "c.py", "hash-c", 30),
            ],
        )
        connection.executemany(
            """INSERT INTO edges VALUES
               (?, 'snapshot-1', ?, ?, ?, ?, '{}')""",
            [
                ("edge-ab", "file-a", "file-b", "imports", 1.0),
                ("edge-bc", "file-b", "file-c", "imports", 0.9),
            ],
        )

    ledger.write_transaction(operation)


def test_graph_is_bounded_evidence_derived_and_traceable(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    _seed(ledger, tmp_path)
    repository = SqliteArchitectureRepository(ledger)

    snapshots = repository.list_snapshots()
    assert snapshots[0]["file_count"] == 3
    assert snapshots[0]["edge_count"] == 2
    graph = repository.graph("snapshot-1", focus="a.py", depth=1, limit=10)
    assert graph is not None
    assert [node["label"] for node in graph["nodes"]] == ["a.py", "b.py"]
    assert graph["edges"][0]["derivation"] == "repository-index"
    assert graph["summary"]["changed_paths"] == ["a.py"]
    assert repository.graph("snapshot-1", focus="missing.py", limit=10)["nodes"] == []
    assert repository.graph("missing") is None

    trace = repository.trace("snapshot-1", source_id="file-a", target_id="file-c", max_hops=2)
    assert trace is not None
    assert trace["found"] is True
    assert [node["label"] for node in trace["nodes"]] == ["a.py", "b.py", "c.py"]
    missing = repository.trace("snapshot-1", source_id="file-c", target_id="file-a", max_hops=2)
    assert missing is not None
    assert missing["found"] is False
    ledger.close()


def test_architecture_api_contracts_and_limits(tmp_path: Path) -> None:
    service = AnalysisService(database_path=tmp_path / "codeinsight.db")
    _seed(service._get_ledger(), tmp_path)
    app = create_app(service)

    with TestClient(app) as client:
        snapshots = client.get("/api/v1/snapshots")
        assert snapshots.status_code == 200
        assert snapshots.json()[0]["display_name"] == "architecture-fixture"
        graph = client.get("/api/v1/snapshots/snapshot-1/architecture?focus=a.py&depth=2")
        assert graph.status_code == 200
        assert len(graph.json()["nodes"]) == 3
        trace = client.get(
            "/api/v1/snapshots/snapshot-1/trace?source_id=file-a&target_id=file-c&max_hops=2"
        )
        assert trace.status_code == 200
        assert trace.json()["found"] is True
        assert client.get("/api/v1/snapshots/missing/architecture").status_code == 404
        assert client.get("/api/v1/snapshots?limit=0").status_code == 422
        assert client.get("/api/v1/snapshots/snapshot-1/architecture?depth=5").status_code == 422
