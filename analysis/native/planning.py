"""Repository-driven trusted role and wave planning."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from typing import Any

from domain.contracts import (
    AnalysisMode,
    AnalysisTask,
    RepositorySnapshot,
    RoleSpec,
    RunBudget,
    WavePlan,
)

TRUSTED_TOOLS = frozenset(
    {
        "repository.read_span",
        "repository.search_symbols",
        "repository.graph_neighborhood",
    }
)
TRUSTED_CAPABILITIES = frozenset(
    {
        "architecture",
        "change-impact",
        "configuration",
        "language-analysis",
        "security",
        "test-quality",
        "verification",
    }
)


class RepositoryRolePlanner:
    """Derive bounded specialist roles from the persisted repository profile."""

    def plan(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        files: Sequence[dict[str, Any]],
        target_paths: Sequence[str],
        mode: AnalysisMode,
        budget: RunBudget,
        wave_number: int,
        remaining_gaps: Sequence[str] = (),
    ) -> WavePlan:
        if wave_number < 1 or wave_number > budget.max_waves:
            raise ValueError("wave number exceeds run budget")
        by_path = {str(item["relative_path"]): item for item in files}
        targets = [
            by_path[path] for path in target_paths if path in by_path
        ] or list(files)
        proposals = (
            self._follow_up_proposals(targets, remaining_gaps)
            if wave_number > 1
            else self._initial_proposals(targets, mode)
        )
        proposals = proposals[: budget.max_specialists]
        if not proposals:
            raise ValueError("repository profile produced no trusted roles")
        wave_id = _stable_id(run_id, f"wave:{wave_number}")
        token_pool = int(
            budget.max_tokens
            * (
                budget.reserved_follow_up_fraction
                if wave_number > 1
                else 1 - budget.reserved_follow_up_fraction
            )
        )
        cost_pool = budget.max_cost_usd * (
            budget.reserved_follow_up_fraction
            if wave_number > 1
            else 1 - budget.reserved_follow_up_fraction
        )
        roles: list[RoleSpec] = []
        tasks: list[AnalysisTask] = []
        for index, proposal in enumerate(proposals):
            role_key = str(proposal["key"])
            role_files = self._select_files(targets, proposal)
            if not role_files:
                continue
            role_id = _stable_id(wave_id, role_key)
            model_policy = (
                "strong"
                if proposal["capability"] in {"architecture", "security", "verification"}
                else "balanced"
            )
            role = RoleSpec(
                role_id=role_id,
                name=str(proposal["name"]),
                mission=str(proposal["mission"]),
                rationale=str(proposal["rationale"]),
                coverage_targets=tuple(proposal["coverage_targets"]),
                required_capabilities=(str(proposal["capability"]),),
                allowed_tools=tuple(sorted(TRUSTED_TOOLS)),
                input_artifact_ids=tuple(
                    str(item["artifact_id"]) for item in role_files
                ),
                model_policy=model_policy,
                token_budget=max(1, token_pool // len(proposals)),
                time_budget_seconds=max(
                    1,
                    min(
                        900,
                        budget.max_elapsed_seconds // max(1, len(proposals)),
                    ),
                ),
                completion_criteria=(
                    "return only typed evidence-backed finding proposals",
                    "identify unresolved uncertainty and analyzed targets",
                ),
            )
            task_id = _stable_id(role_id, "specialist-analysis")
            tasks.append(
                AnalysisTask(
                    task_id=task_id,
                    run_id=run_id,
                    wave_id=wave_id,
                    role_id=role_id,
                    task_type="specialist_analysis",
                    priority=index * 10,
                    immutable_input_ids=(
                        snapshot.snapshot_id,
                        *(str(item["file_id"]) for item in role_files),
                    ),
                    configuration_hash=snapshot.configuration_hash,
                    idempotency_key=_stable_id(
                        snapshot.identity_hash,
                        f"{run_id}:{wave_number}:{role_key}",
                    ),
                    model_policy=model_policy,
                    token_budget=role.token_budget,
                    cost_budget_usd=max(0, cost_pool / len(proposals)),
                    time_budget_seconds=role.time_budget_seconds,
                    tool_call_budget=min(50, 5 + len(role_files)),
                    max_attempts=3,
                )
            )
            roles.append(role)
        if not roles:
            raise ValueError("no role had files inside the trusted snapshot scope")
        coverage_targets = tuple(
            sorted(
                {
                    target
                    for role in roles
                    for target in role.coverage_targets
                }
            )
        )
        return WavePlan(
            wave_id=wave_id,
            run_id=run_id,
            wave_number=wave_number,
            rationale=(
                "close explicit coverage gaps"
                if wave_number > 1
                else "cover repository-specific languages and risk surfaces"
            ),
            roles=tuple(roles),
            tasks=tuple(tasks),
            coverage_targets=coverage_targets,
            reserved_follow_up=wave_number > 1,
        )

    @staticmethod
    def validate_role(role: RoleSpec) -> None:
        if not set(role.allowed_tools).issubset(TRUSTED_TOOLS):
            raise ValueError(f"role {role.role_id} requested an untrusted tool")
        if not set(role.required_capabilities).issubset(TRUSTED_CAPABILITIES):
            raise ValueError(f"role {role.role_id} requested an untrusted capability")

    def _initial_proposals(
        self, targets: Sequence[dict[str, Any]], mode: AnalysisMode
    ) -> list[dict[str, Any]]:
        languages = Counter(
            str(item["language"])
            for item in targets
            if item["classification"] == "source"
        )
        classifications = Counter(str(item["classification"]) for item in targets)
        proposals: list[dict[str, Any]] = [
            {
                "key": "architecture",
                "name": "Repository Architecture Specialist",
                "mission": "Trace module boundaries, dependencies, and structural risks.",
                "rationale": "Every analysis needs repository-wide structural context.",
                "coverage_targets": ("architecture", "dependencies"),
                "capability": "architecture",
                "selection": "all",
            }
        ]
        for language, _count in languages.most_common():
            proposals.append(
                {
                    "key": f"language:{language}",
                    "name": f"{language.title()} Implementation Specialist",
                    "mission": (
                        f"Inspect {language} implementation behavior and failure paths."
                    ),
                    "rationale": f"{language} is present in the target snapshot.",
                    "coverage_targets": (f"language:{language}",),
                    "capability": "language-analysis",
                    "selection": f"language:{language}",
                }
            )
        if classifications["test"]:
            proposals.append(
                {
                    "key": "tests",
                    "name": "Test and Reliability Specialist",
                    "mission": "Assess behavioral coverage, failure paths, and test quality.",
                    "rationale": "The repository contains an executable test surface.",
                    "coverage_targets": ("classification:test", "reliability"),
                    "capability": "test-quality",
                    "selection": "tests-and-source",
                }
            )
        if mode in {AnalysisMode.DEEP, AnalysisMode.SECURITY}:
            security = {
                "key": "security",
                "name": "Security and Trust-Boundary Specialist",
                "mission": "Trace untrusted input, authorization, and dangerous sinks.",
                "rationale": f"{mode.value} mode requires explicit security coverage.",
                "coverage_targets": ("security", "trust-boundaries"),
                "capability": "security",
                "selection": "source-and-config",
            }
            if mode == AnalysisMode.SECURITY:
                proposals.insert(0, security)
            else:
                proposals.append(security)
        if mode == AnalysisMode.CHANGE_SET:
            proposals.insert(
                0,
                {
                    "key": "change-impact",
                    "name": "Change Impact Specialist",
                    "mission": "Trace changed behavior through dependency and call neighbors.",
                    "rationale": "Change-set mode prioritizes regression and blast radius.",
                    "coverage_targets": ("change-impact",),
                    "capability": "change-impact",
                    "selection": "all",
                },
            )
        if classifications["configuration"]:
            proposals.append(
                {
                    "key": "configuration",
                    "name": "Configuration and Supply-Chain Specialist",
                    "mission": "Inspect runtime, dependency, and deployment configuration.",
                    "rationale": "Configuration files affect runtime and dependency behavior.",
                    "coverage_targets": ("classification:configuration",),
                    "capability": "configuration",
                    "selection": "configuration",
                }
            )
        return proposals

    @staticmethod
    def _follow_up_proposals(
        targets: Sequence[dict[str, Any]], gaps: Sequence[str]
    ) -> list[dict[str, Any]]:
        proposals = []
        for gap in sorted(set(gaps)):
            capability = "verification" if gap.startswith("uncertainty:") else (
                "security" if "security" in gap else "language-analysis"
            )
            proposals.append(
                {
                    "key": f"follow-up:{gap}",
                    "name": f"Follow-up Specialist: {gap}",
                    "mission": f"Close the recorded coverage gap: {gap}.",
                    "rationale": "A prior wave recorded a material unresolved gap.",
                    "coverage_targets": (gap,),
                    "capability": capability,
                    "selection": "all",
                }
            )
        return proposals if targets else []

    @staticmethod
    def _select_files(
        targets: Sequence[dict[str, Any]], proposal: dict[str, Any]
    ) -> list[dict[str, Any]]:
        selection = proposal["selection"]
        if selection == "all":
            return list(targets)
        if str(selection).startswith("language:"):
            language = str(selection).split(":", 1)[1]
            return [item for item in targets if item["language"] == language]
        if selection == "tests-and-source":
            return [
                item
                for item in targets
                if item["classification"] in {"test", "source"}
            ]
        if selection == "source-and-config":
            return [
                item
                for item in targets
                if item["classification"] in {"source", "configuration"}
            ]
        if selection == "configuration":
            return [
                item for item in targets if item["classification"] == "configuration"
            ]
        return []


def _stable_id(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()
