"""Bounded V6-inspired architecture discovery backed by a real provider."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from application.model_gateway import ModelGateway, ModelRequest
from domain.contracts import (
    ArchitectureDiscovery,
    ContractModel,
    RepositorySnapshot,
    RunBudget,
)
from infrastructure.artifacts.store import FilesystemArtifactStore, StoredArtifact
from infrastructure.db.planning_repository import SqlitePlanningRepository
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository

PROMPT_TEMPLATE = "v6-architecture-discovery"
PROMPT_VERSION = 1
MAX_FILE_SUMMARY = 80
MAX_SOURCE_FILES = 10
MAX_SOURCE_CHARS_PER_FILE = 3_000
MAX_SOURCE_CHARS_TOTAL = 20_000
PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "core"
    / "phases"
    / "planning"
    / "architecture-discovery.md"
)


class EvidencePattern(ContractModel):
    pattern: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class ArchitectureModule(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=2_000)
    complexity: Literal["simple", "medium", "complex", "very_complex", "unknown"]
    files: tuple[str, ...] = Field(max_length=80)
    entry_points: tuple[str, ...] = Field(max_length=40)
    exposed_apis: tuple[str, ...] = Field(max_length=40)
    confidence: float = Field(ge=0, le=1)


class ArchitectureDependency(ContractModel):
    source: str = Field(min_length=1, max_length=400)
    target: str = Field(min_length=1, max_length=400)
    type: Literal["import", "call", "data", "event"]
    confidence: float = Field(ge=0, le=1)
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class ArchitectureDataFlow(ContractModel):
    source: str = Field(min_length=1, max_length=400)
    target: str = Field(min_length=1, max_length=400)
    data_type: str = Field(min_length=1, max_length=400)
    protocol: str = Field(min_length=1, max_length=200)
    direction: Literal["unidirectional", "bidirectional"]
    confidence: float = Field(ge=0, le=1)
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class ApiParameter(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    type: str = Field(min_length=1, max_length=200)


class ArchitectureEndpoint(ContractModel):
    path: str = Field(min_length=1, max_length=1_000)
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "UNKNOWN"]
    description: str = Field(min_length=1, max_length=2_000)
    parameters: tuple[ApiParameter, ...] = Field(max_length=40)
    response_type: str = Field(min_length=1, max_length=400)
    authentication_required: bool | Literal["unknown"]
    rate_limited: bool | Literal["unknown"]
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class Technology(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)


class DatabaseResource(ContractModel):
    table_or_resource: str = Field(min_length=1, max_length=400)
    relationships: tuple[str, ...] = Field(max_length=40)
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class DesignPattern(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=1_000)
    confidence: float = Field(ge=0, le=1)


class SecurityMechanism(ContractModel):
    mechanism: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0, le=1)
    evidence_paths: tuple[str, ...] = Field(max_length=20)


class SecurityArchitecture(ContractModel):
    authentication: tuple[SecurityMechanism, ...] = Field(max_length=40)
    authorization: tuple[SecurityMechanism, ...] = Field(max_length=40)
    concerns: tuple[str, ...] = Field(max_length=80)


class PerformanceCharacteristics(ContractModel):
    bottlenecks: tuple[str, ...] = Field(max_length=80)
    optimizations: tuple[str, ...] = Field(max_length=80)


class AntiPattern(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=1_000)
    severity: Literal["low", "medium", "high"]


class ArchitectureDiscoveryOutput(ContractModel):
    system_name: str | None = Field(default=None, max_length=500)
    system_type: Literal["web_app", "library", "api_service", "cli_tool", "data_science", "unknown"]
    architecture_patterns: tuple[EvidencePattern, ...] = Field(max_length=40)
    modules: tuple[ArchitectureModule, ...] = Field(max_length=80)
    dependencies: tuple[ArchitectureDependency, ...] = Field(max_length=160)
    data_flows: tuple[ArchitectureDataFlow, ...] = Field(max_length=160)
    api_endpoints: tuple[ArchitectureEndpoint, ...] = Field(max_length=160)
    tech_stack: dict[Literal["frameworks", "libraries"], tuple[Technology, ...]]
    database_schema: tuple[DatabaseResource, ...] = Field(max_length=160)
    design_patterns: tuple[DesignPattern, ...] = Field(max_length=80)
    security_architecture: SecurityArchitecture
    performance_characteristics: PerformanceCharacteristics
    anti_patterns: tuple[AntiPattern, ...] = Field(max_length=80)
    architectural_smells: tuple[str, ...] = Field(max_length=40)
    unknowns: tuple[str, ...] = Field(max_length=40)


class ArchitectureDiscoveryService:
    """Request and durably record one validated architecture model."""

    def __init__(
        self,
        *,
        repository: SqlitePlanningRepository,
        gateway: ModelGateway | None,
        model: str,
        snapshots: SqliteSnapshotRepository | None = None,
        artifacts: FilesystemArtifactStore | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.model = model
        self.snapshots = snapshots
        self.artifacts = artifacts

    async def discover(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        files: list[dict[str, Any]],
        budget: RunBudget,
    ) -> ArchitectureDiscovery:
        summary = build_architecture_input(
            files,
            source_files=self._source_bundle(snapshot, files),
        )
        prompt = architecture_system_prompt()
        input_hash = _hash(summary)
        result: dict[str, Any]
        status: str
        if self.gateway is None:
            result = {"unknowns": ("model provider unavailable",)}
            # "fallback" is a legacy persisted enum; this branch does not plan or dispatch roles.
            status = "fallback"
        else:
            try:
                response = await self.gateway.complete(
                    ModelRequest(
                        run_id=run_id,
                        wave_id="architecture-discovery",
                        task_id=_stable_id(run_id, snapshot.snapshot_id),
                        model=self.model,
                        system_prompt=prompt,
                        user_prompt=json.dumps(summary, sort_keys=True, separators=(",", ":")),
                        response_schema=ArchitectureDiscoveryOutput.model_json_schema(),
                        response_model=ArchitectureDiscoveryOutput,
                        max_output_tokens=min(4_000, budget.max_tokens),
                        timeout_seconds=min(90, budget.max_elapsed_seconds),
                        token_budget=min(12_000, budget.max_tokens),
                        cost_budget_usd=min(1.0, budget.max_cost_usd),
                        correlation={
                            "kind": "architecture-discovery",
                            "snapshot_id": snapshot.snapshot_id,
                        },
                    )
                )
                result = ArchitectureDiscoveryOutput.model_validate(response.content).model_dump(
                    mode="json"
                )
                status = "model-validated"
            except (ValidationError, ValueError, RuntimeError, OSError):
                result = {"unknowns": ("model result unavailable or invalid",)}
                # Legacy persisted enum; this branch does not plan or dispatch roles.
                status = "fallback"
        discovery = ArchitectureDiscovery(
            discovery_id=_stable_id(run_id, input_hash),
            run_id=run_id,
            snapshot_id=snapshot.snapshot_id,
            input_hash=input_hash,
            prompt_template=PROMPT_TEMPLATE,
            prompt_version=PROMPT_VERSION,
            prompt_path=PROMPT_PATH.relative_to(PROMPT_PATH.parents[4]).as_posix(),
            prompt_content_hash=_hash_text(prompt),
            prompt_text=prompt,
            result=result,
            result_hash=_hash(result),
            status=status,
        )
        self.repository.record_discovery(discovery)
        return discovery

    def _source_bundle(
        self,
        snapshot: RepositorySnapshot,
        files: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Read a small verified source sample from immutable artifacts only."""
        if self.snapshots is None or self.artifacts is None:
            return ()
        contexts = self.snapshots.file_contexts(
            snapshot.snapshot_id,
            select_architecture_source_paths(files),
        )
        selected: list[dict[str, Any]] = []
        remaining = MAX_SOURCE_CHARS_TOTAL
        for context in contexts:
            if remaining <= 0:
                break
            try:
                artifact = StoredArtifact(
                    artifact_id=str(context["artifact_id"]),
                    content_hash=str(context["content_hash"]),
                    artifact_kind="source",
                    storage_path=str(context["storage_path"]),
                    byte_size=int(context["byte_size"]),
                    media_type=context.get("media_type"),
                )
                content = self.artifacts.read(artifact).decode("utf-8", errors="replace")
            except (OSError, UnicodeError, ValueError):
                continue
            excerpt = content[: min(MAX_SOURCE_CHARS_PER_FILE, remaining)]
            selected.append(
                {
                    "relative_path": str(context["relative_path"]),
                    "language": str(context["language"]),
                    "content": excerpt,
                    "truncated": len(excerpt) < len(content),
                }
            )
            remaining -= len(excerpt)
        return tuple(selected)


def architecture_system_prompt() -> str:
    """Load the editable V6-derived prompt template from the tracked prompt tree."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip() + "\n"
    except OSError as exc:
        raise RuntimeError(f"architecture discovery prompt unavailable: {PROMPT_PATH}") from exc


def build_architecture_input(
    files: list[dict[str, Any]],
    *,
    source_files: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Bound metadata plus a verified source sample for architecture planning."""
    ordered = sorted(files, key=lambda item: str(item["relative_path"]))[:MAX_FILE_SUMMARY]
    languages = Counter(str(item.get("language", "unknown")) for item in files)
    classifications = Counter(str(item.get("classification", "unknown")) for item in files)
    return {
        "summary_version": 1,
        "file_count": len(files),
        "languages": dict(sorted(languages.items())),
        "classifications": dict(sorted(classifications.items())),
        "files": [
            {
                key: item.get(key)
                for key in ("relative_path", "language", "classification", "support_tier")
            }
            for item in ordered
        ],
        "truncated": len(files) > len(ordered),
        "source_files": list(source_files),
    }


def select_architecture_source_paths(files: list[dict[str, Any]]) -> tuple[str, ...]:
    """Prioritize architecture-bearing files and exclude package/test boilerplate."""
    priority_names = {
        "main.py": 100,
        "app.py": 95,
        "server.py": 90,
        "application.py": 90,
        "manage.py": 85,
        "pyproject.toml": 80,
        "package.json": 80,
        "docker-compose.yml": 80,
        "docker-compose.yaml": 80,
    }
    priority_parts = {
        "api": 50,
        "routes": 50,
        "controllers": 45,
        "services": 45,
        "models": 40,
        "database": 40,
        "config": 35,
        "settings": 35,
    }
    candidates: list[tuple[int, str]] = []
    for item in files:
        relative_path = str(item["relative_path"])
        path = Path(relative_path)
        if path.name == "__init__.py" or item.get("classification") == "test":
            continue
        if item.get("classification") not in {"source", "config"}:
            continue
        score = priority_names.get(path.name, 0)
        score += max((priority_parts.get(part.lower(), 0) for part in path.parts), default=0)
        score += min(int(item.get("line_count", 0)), 2_000) // 100
        candidates.append((score, relative_path))
    return tuple(
        path
        for _score, path in sorted(candidates, key=lambda item: (-item[0], item[1]))[
            :MAX_SOURCE_FILES
        ]
    )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()
