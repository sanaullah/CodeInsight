from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from application.analysis_service import AnalysisService
from indexing.repository_index import RepositoryIndexer
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.database import database_connection
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository


def _semantic_snapshot(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text(
        "from fastapi import FastAPI\n"
        "import sqlite3\n"
        "app = FastAPI()\n"
        "@app.post('/orders')\n"
        "def orders():\n"
        "    sqlite3.connect('orders.db')\n",
        encoding="utf-8",
    )
    database_path = tmp_path / "codeinsight.db"
    ledger = SqliteRunLedger(database_path)
    try:
        result = RepositoryIndexer(
            SqliteSnapshotRepository(ledger),
            FilesystemArtifactStore(tmp_path / "artifacts"),
        ).build(root)
        return database_path, result.snapshot.snapshot_id
    finally:
        ledger.close()


def test_semantic_architecture_api_is_bounded_and_traceable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    database_path, snapshot_id = _semantic_snapshot(tmp_path)
    app = create_app(AnalysisService(database_path=database_path))

    with TestClient(app) as client:
        summary = client.get(f"/api/v1/snapshots/{snapshot_id}/semantic-summary")
        assert summary.status_code == 200
        assert summary.json()["counts"]["services"] == 1
        assert summary.json()["totals"]["files"] == 1
        assert summary.json()["git"]["ref"] is None

        graph = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-architecture",
            params={"limit": 10, "depth": 1},
        )
        assert graph.status_code == 200
        body = graph.json()
        assert body["completeness"]["status"] in {"complete", "partial"}
        assert not body["truncated"]
        kinds = {item["component_kind"] for item in body["components"]}
        assert {"service", "datastore"} <= kinds

        service = next(
            item for item in body["components"] if item["component_kind"] == "service"
        )
        store = next(
            item for item in body["components"] if item["component_kind"] == "datastore"
        )
        detail = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-components/"
            f"{service['component_id']}"
        )
        assert detail.status_code == 200
        assert detail.json()["endpoints"][0]["route"] == "/orders"
        assert detail.json()["provenance"]
        assert isinstance(detail.json()["memberships"][0]["provenance"], dict)
        assert "file_metadata" not in detail.json()["memberships"][0]

        trace = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-trace",
            params={
                "source_id": service["component_id"],
                "target_id": store["component_id"],
            },
        )
        assert trace.status_code == 200
        trace_body = trace.json()
        assert trace_body["status"] == "complete"
        assert trace_body["relations"][0]["relation_kind"] == "data_access"
        assert isinstance(trace_body["relations"][0]["is_async"], bool)
        assert trace_body["endpoints"][0]["route"] == "/orders"
        assert any(item["resource_kind"] == "database" for item in trace_body["resources"])
        assert trace_body["provenance"]
        assert not trace_body["evidence_truncated"]


def test_semantic_trace_bounds_evidence_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    database_path, snapshot_id = _semantic_snapshot(tmp_path)
    with database_connection(database_path) as connection:
        service_id = connection.execute(
            """
            SELECT component_id FROM semantic_components
            WHERE snapshot_id = ? AND component_kind = 'service'
            """,
            (snapshot_id,),
        ).fetchone()[0]
        file_id = connection.execute(
            "SELECT file_id FROM files WHERE snapshot_id = ? LIMIT 1",
            (snapshot_id,),
        ).fetchone()[0]
        connection.executemany(
            """
            INSERT INTO semantic_memberships(
                membership_id, snapshot_id, component_id, file_id,
                symbol_id, membership_kind, confidence, provenance_json
            ) VALUES (?, ?, ?, ?, NULL, 'implementation', 1.0, '{}')
            """,
            [
                (
                    f"bounded-membership-{index:03d}",
                    snapshot_id,
                    service_id,
                    file_id,
                )
                for index in range(251)
            ],
        )
        connection.executemany(
            """
            INSERT INTO semantic_endpoints(
                endpoint_id, snapshot_id, stable_key, component_id, route,
                method, protocol, direction, support_tier, completeness,
                confidence, metadata_json
            ) VALUES (?, ?, ?, ?, ?, 'GET', 'http', 'inbound',
                      'exact', 'complete', 1.0, '{}')
            """,
            [
                (
                    f"bounded-endpoint-{index:03d}",
                    snapshot_id,
                    f"endpoint:bounded-{index:03d}",
                    service_id,
                    f"/bounded/{index}",
                )
                for index in range(501)
            ],
        )
        store_id = connection.execute(
            """
            SELECT component_id FROM semantic_components
            WHERE snapshot_id = ? AND component_kind = 'datastore'
            """,
            (snapshot_id,),
        ).fetchone()[0]
    app = create_app(AnalysisService(database_path=database_path))

    started = perf_counter()
    with TestClient(app) as client:
        detail = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-components/{service_id}"
        )
        response = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-trace",
            params={"source_id": service_id, "target_id": store_id},
        )
    elapsed = perf_counter() - started

    assert detail.status_code == 200
    assert len(detail.json()["memberships"]) == 250
    assert detail.json()["evidence_truncated"]
    assert detail.json()["limits"]["detail_row_limit"] == 250
    assert response.status_code == 200
    assert len(response.json()["endpoints"]) == 500
    assert response.json()["evidence_truncated"]
    assert elapsed < 3


def test_semantic_architecture_api_reports_missing_and_rejects_unbounded_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    database_path, snapshot_id = _semantic_snapshot(tmp_path)
    app = create_app(AnalysisService(database_path=database_path))

    with TestClient(app) as client:
        assert client.get("/api/v1/snapshots/missing/semantic-summary").status_code == 404
        assert (
            client.get(
                f"/api/v1/snapshots/{snapshot_id}/semantic-architecture",
                params={"limit": 1001},
            ).status_code
            == 422
        )
        assert (
            client.get(
                f"/api/v1/snapshots/{snapshot_id}/semantic-components/missing"
            ).status_code
            == 404
        )


def test_semantic_graph_bounds_five_thousand_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    database_path, snapshot_id = _semantic_snapshot(tmp_path)
    with database_connection(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO semantic_components(
                component_id, snapshot_id, stable_key, name, component_kind,
                support_tier, completeness, confidence, metadata_json
            ) VALUES (?, ?, ?, ?, 'library', 'partial', 'partial', 0.5, '{}')
            """,
            [
                (
                    f"large-{index:05d}",
                    snapshot_id,
                    f"library:large-{index:05d}",
                    f"Large {index}",
                )
                for index in range(5_000)
            ],
        )
    app = create_app(AnalysisService(database_path=database_path))
    started = perf_counter()
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/snapshots/{snapshot_id}/semantic-architecture",
            params={"limit": 100},
        )
    elapsed = perf_counter() - started
    assert response.status_code == 200
    assert response.json()["truncated"]
    assert len(response.json()["components"]) == 100
    assert elapsed < 3
