from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings
from application.analysis_service import AnalysisService
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.settings_repository import (
    DEFAULT_SETTINGS,
    SqliteSettingsRepository,
)


def _preset_request() -> dict:
    return {
        "goal": None,
        "model_name": None,
        "max_agents": 4,
        "file_extensions": None,
        "selected_directories": None,
        "mode": "deep",
        "max_waves": 2,
        "max_tasks": 100,
        "max_total_tokens": 1_000_000,
        "max_cost_usd": 25,
        "max_elapsed_seconds": 3_600,
    }


def test_settings_persist_with_optimistic_concurrency(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    repository = SqliteSettingsRepository(SqliteRunLedger(database_path))

    assert repository.get() == {
        "settings": DEFAULT_SETTINGS,
        "version": 0,
        "updated_at": None,
    }
    changed = {**DEFAULT_SETTINGS, "default_mode": "security"}
    saved = repository.update(changed, expected_version=0)
    assert saved["settings"] == changed
    assert saved["version"] == 1
    assert SqliteSettingsRepository(SqliteRunLedger(database_path)).get() == saved

    with pytest.raises(RuntimeError, match="refresh and retry"):
        repository.update(DEFAULT_SETTINGS, expected_version=0)
    assert repository.get() == saved


def test_presets_create_update_delete_and_conflicts(tmp_path: Path) -> None:
    repository = SqliteSettingsRepository(SqliteRunLedger(tmp_path / "codeinsight.db"))
    first = repository.save_preset(name="Security", request=_preset_request())
    assert first["version"] == 1
    assert repository.get_preset(first["preset_id"]) == first

    updated = repository.save_preset(
        preset_id=first["preset_id"],
        name="Security deep",
        request={**_preset_request(), "max_waves": 3},
        expected_version=1,
    )
    assert updated["version"] == 2
    assert updated["created_at"] == first["created_at"]

    with pytest.raises(RuntimeError, match="refresh and retry"):
        repository.save_preset(
            preset_id=first["preset_id"],
            name="stale",
            request=_preset_request(),
            expected_version=1,
        )
    with pytest.raises(KeyError, match="preset not found"):
        repository.save_preset(
            preset_id="missing",
            name="missing",
            request=_preset_request(),
            expected_version=1,
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_preset(name="Security deep", request=_preset_request())
    with pytest.raises(RuntimeError, match="refresh and retry"):
        repository.delete_preset(first["preset_id"], expected_version=1)
    assert repository.delete_preset(first["preset_id"], expected_version=2)
    assert not repository.delete_preset(first["preset_id"], expected_version=2)


def test_settings_api_validates_and_never_persists_paths_or_secrets(
    tmp_path: Path,
) -> None:
    app = create_app(AnalysisService(database_path=tmp_path / "codeinsight.db"))

    with TestClient(app) as client:
        initial = client.get("/api/v1/settings")
        assert initial.status_code == 200
        body = initial.json()
        body["settings"]["default_max_agents"] = 7
        updated = client.put(
            "/api/v1/settings",
            json={"settings": body["settings"], "expected_version": body["version"]},
        )
        assert updated.status_code == 200
        assert updated.json()["version"] == 1
        assert (
            client.put(
                "/api/v1/settings",
                json={"settings": body["settings"], "expected_version": 0},
            ).status_code
            == 409
        )
        invalid = {**body["settings"], "retention_days": 0}
        assert (
            client.put(
                "/api/v1/settings",
                json={"settings": invalid, "expected_version": 1},
            ).status_code
            == 422
        )

        created = client.post(
            "/api/v1/presets",
            json={"name": "Local security", "request": _preset_request()},
        )
        assert created.status_code == 201
        preset = created.json()
        serialized = str(preset).lower()
        assert "project_path" not in serialized
        assert "api_key" not in serialized
        assert client.get("/api/v1/presets").json() == [preset]

        preset["name"] = "Local security edited"
        edited = client.put(
            f"/api/v1/presets/{preset['preset_id']}",
            json={
                "name": preset["name"],
                "request": preset["request"],
                "expected_version": preset["version"],
            },
        )
        assert edited.status_code == 200
        assert edited.json()["version"] == 2
        assert (
            client.put(
                "/api/v1/presets/missing",
                json={
                    "name": "missing",
                    "request": _preset_request(),
                    "expected_version": 1,
                },
            ).status_code
            == 404
        )
        assert (
            client.delete(
                f"/api/v1/presets/{preset['preset_id']}?expected_version=1"
            ).status_code
            == 409
        )
        assert (
            client.delete(
                f"/api/v1/presets/{preset['preset_id']}?expected_version=2"
            ).status_code
            == 204
        )


def test_provider_probe_reports_truth_without_revealing_configuration(
    tmp_path: Path,
) -> None:
    offline = create_app(AnalysisService(database_path=tmp_path / "offline.db"))
    with TestClient(offline) as client:
        response = client.post("/api/v1/settings/provider-test")
        assert response.json() == {
            "configured": False,
            "reachable": False,
            "status": "Provider endpoint is not configured.",
            "latency_ms": None,
        }

    settings = ApiSettings(
        database_path=tmp_path / "configured.db",
        model_base_url="http://127.0.0.1:1/v1",
        model_api_key="never-return-this",
    )
    configured = create_app(
        AnalysisService(database_path=settings.database_path, settings=settings)
    )
    with TestClient(configured) as client:
        body = client.post("/api/v1/settings/provider-test").json()
    assert body["configured"] is True
    assert body["reachable"] is False
    assert "127.0.0.1" not in str(body)
    assert "never-return-this" not in str(body)
