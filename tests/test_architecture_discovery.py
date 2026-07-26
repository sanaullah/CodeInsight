# ruff: noqa: E501
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from analysis.native.architecture_discovery import (
    PROMPT_PATH,
    ArchitectureDiscoveryService,
    architecture_system_prompt,
    build_architecture_input,
)
from application.model_gateway import ModelRequest, ModelResponse
from domain.contracts import RepositorySnapshot, RunBudget
from infrastructure.db.planning_repository import SqlitePlanningRepository
from infrastructure.db.run_ledger import SqliteRunLedger


class ValidGateway:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        assert "expert software architect" in request.system_prompt
        assert "Strict grounding" in request.system_prompt
        assert "Extraction targets" in request.system_prompt
        return ModelResponse(
            content={
                "system_name": "fixture",
                "system_type": "api_service",
                "architecture_patterns": [
                    {
                        "pattern": "layered",
                        "evidence_paths": ["api.py"],
                        "confidence": 0.9,
                    }
                ],
                "modules": [
                    {
                        "name": "api",
                        "purpose": "serve requests",
                        "complexity": "simple",
                        "files": ["api.py"],
                        "entry_points": ["api.py"],
                        "exposed_apis": [],
                        "confidence": 0.9,
                    }
                ],
                "dependencies": [
                    {
                        "source": "api.py",
                        "target": "service.py",
                        "type": "import",
                        "confidence": 0.8,
                        "evidence_paths": ["api.py"],
                    }
                ],
                "data_flows": [],
                "api_endpoints": [],
                "design_patterns": [],
                "tech_stack": {"frameworks": [], "libraries": []},
                "database_schema": [],
                "security_architecture": {
                    "authentication": [],
                    "authorization": [],
                    "concerns": [],
                },
                "performance_characteristics": {"bottlenecks": [], "optimizations": []},
                "anti_patterns": [],
                "architectural_smells": [],
                "unknowns": [],
            },
            provider="fixture",
            model=request.model,
        )


class InvalidGateway:
    async def complete(self, _request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content={"system_type": "missing required fields"}, provider="fixture", model="x"
        )


def _setup(tmp_path: Path) -> tuple[SqlitePlanningRepository, RepositorySnapshot]:
    ledger = SqliteRunLedger(tmp_path / "planning.db")
    ledger.create_run(
        run_id="run-1",
        submission_key="submission-1",
        request={"project_path": "sample"},
        mode="quick",
    )

    def seed(connection) -> None:
        connection.execute(
            "INSERT INTO projects(project_id, canonical_path, display_name, created_at, updated_at) VALUES ('project-1', 'sample', 'sample', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO repository_snapshots(snapshot_id, project_id, identity_hash, configuration_hash, scanner_version, created_at) VALUES ('snapshot-1', 'project-1', 'identity', 'config', 'test', 'now')"
        )

    ledger.write_transaction(seed)
    return SqlitePlanningRepository(ledger), RepositorySnapshot(
        snapshot_id="snapshot-1",
        project_id="project-1",
        canonical_path="sample",
        identity_hash="identity",
        configuration_hash="config",
        scanner_version="test",
        created_at=datetime.now(UTC),
    )


def _files() -> list[dict[str, str]]:
    return [
        {
            "relative_path": "api.py",
            "language": "python",
            "classification": "source",
            "support_tier": "semantic",
            "content": "SECRET = 'never include'",
        },
        {
            "relative_path": "tests/test_api.py",
            "language": "python",
            "classification": "test",
            "support_tier": "semantic",
            "content": "",
        },
    ]


def test_architecture_discovery_prompt_preserves_v6_contract_and_v2_bounds() -> None:
    assert PROMPT_PATH.name == "architecture-discovery.md"
    assert PROMPT_PATH.is_file()
    prompt = architecture_system_prompt()
    assert "System structure" in prompt
    assert "Modules" in prompt
    assert "Quality" in prompt
    assert "Strict grounding" in prompt
    assert "unknowns" in prompt


def test_architecture_input_is_bounded_metadata_only() -> None:
    summary = build_architecture_input(_files())
    assert summary["file_count"] == 2
    assert "content" not in str(summary)
    assert "SECRET" not in str(summary)


def test_valid_model_discovery_is_validated_and_durable(tmp_path: Path) -> None:
    repository, snapshot = _setup(tmp_path)
    result = asyncio.run(
        ArchitectureDiscoveryService(
            repository=repository, gateway=ValidGateway(), model="fixture"
        ).discover(run_id="run-1", snapshot=snapshot, files=_files(), budget=RunBudget())
    )
    assert result.status == "model-validated"
    assert result.result["modules"][0]["files"] == ["api.py"]
    assert result.result["dependencies"][0]["type"] == "import"
    stored = repository.discoveries_for_run("run-1")[0]
    assert "SECRET" not in str(stored)
    assert stored["prompt_path"] == "prompts/core/phases/planning/architecture-discovery.md"
    assert len(str(stored["prompt_content_hash"])) == 64


def test_invalid_or_unavailable_model_falls_back_without_inference(tmp_path: Path) -> None:
    repository, snapshot = _setup(tmp_path)
    result = asyncio.run(
        ArchitectureDiscoveryService(
            repository=repository, gateway=InvalidGateway(), model="fixture"
        ).discover(run_id="run-1", snapshot=snapshot, files=_files(), budget=RunBudget())
    )
    assert result.status == "fallback"
    assert result.result["modules"] == ()
    assert "invalid" in result.result["unknowns"][0]
