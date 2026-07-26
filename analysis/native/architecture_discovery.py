"""Bounded V6-inspired architecture discovery with a deterministic fallback.

The input is repository metadata only.  Source-bearing excerpts remain owned by
later specialist tasks, so a planning artifact never becomes a second source
store or telemetry leak.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from application.model_gateway import ModelGateway, ModelRequest
from domain.contracts import (
    ArchitectureDiscovery,
    ContractModel,
    RepositorySnapshot,
    RunBudget,
)
from infrastructure.db.planning_repository import SqlitePlanningRepository

PROMPT_TEMPLATE = "v6-architecture-discovery"
PROMPT_VERSION = 1
MAX_FILE_SUMMARY = 80
PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "core"
    / "phases"
    / "planning"
    / "architecture-discovery.md"
)


class ArchitectureComponent(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    responsibility: str = Field(min_length=1, max_length=2_000)
    evidence_paths: tuple[str, ...] = Field(max_length=20)
    confidence: float = Field(ge=0, le=1)


class ArchitectureDiscoveryOutput(ContractModel):
    system_type: str = Field(min_length=1, max_length=500)
    architecture_patterns: tuple[str, ...] = Field(max_length=20)
    components: tuple[ArchitectureComponent, ...] = Field(max_length=40)
    dependencies: tuple[str, ...] = Field(max_length=80)
    data_flows: tuple[str, ...] = Field(max_length=80)
    api_endpoints: tuple[str, ...] = Field(max_length=80)
    design_patterns: tuple[str, ...] = Field(max_length=40)
    technology_stack: dict[str, tuple[str, ...]]
    frameworks: tuple[str, ...] = Field(max_length=40)
    libraries: tuple[str, ...] = Field(max_length=80)
    database_schema: tuple[str, ...] = Field(max_length=80)
    security_architecture: tuple[str, ...] = Field(max_length=40)
    security_considerations: tuple[str, ...] = Field(max_length=40)
    performance_considerations: tuple[str, ...] = Field(max_length=40)
    anti_patterns: tuple[str, ...] = Field(max_length=40)
    architectural_smells: tuple[str, ...] = Field(max_length=40)
    unknowns: tuple[str, ...] = Field(max_length=40)


class ArchitectureDiscoveryService:
    """Request one validated architecture model, otherwise retain facts only."""

    def __init__(
        self, *, repository: SqlitePlanningRepository, gateway: ModelGateway | None, model: str
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.model = model

    async def discover(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        files: list[dict[str, Any]],
        budget: RunBudget,
    ) -> ArchitectureDiscovery:
        summary = build_architecture_input(files)
        prompt = architecture_system_prompt()
        input_hash = _hash(summary)
        result: dict[str, Any]
        status: str
        if self.gateway is None:
            result, status = deterministic_fallback(summary, "model provider unavailable")
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
            except (ValidationError, ValueError, RuntimeError):
                result, status = deterministic_fallback(
                    summary, "model result unavailable or invalid"
                )
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


def architecture_system_prompt() -> str:
    """Load the editable V6-derived prompt template from the tracked prompt tree."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip() + "\n"
    except OSError as exc:
        raise RuntimeError(f"architecture discovery prompt unavailable: {PROMPT_PATH}") from exc


def build_architecture_input(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Bound prompt input by metadata count; never include artifacts or content."""
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
    }


def deterministic_fallback(summary: dict[str, Any], reason: str) -> tuple[dict[str, Any], str]:
    return (
        {
            "system_type": "repository requiring architecture discovery",
            "architecture_patterns": (),
            "components": (),
            "dependencies": (),
            "data_flows": (),
            "api_endpoints": (),
            "design_patterns": (),
            "technology_stack": {"languages": tuple(sorted(summary["languages"]))},
            "frameworks": (),
            "libraries": (),
            "database_schema": (),
            "security_architecture": (),
            "security_considerations": (),
            "performance_considerations": (),
            "anti_patterns": (),
            "architectural_smells": (),
            "unknowns": (reason, "deterministic fallback contains no inferred architecture"),
        },
        "fallback",
    )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()
