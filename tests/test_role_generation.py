from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from analysis.native.coordinator import NativeAnalysisCoordinator
from analysis.native.planning import RepositoryRolePlanner
from analysis.native.role_generation import (
    PROMPT_PATH,
    AiRoleProposalService,
    ProposedRole,
    role_proposal_system_prompt,
)
from application.model_gateway import ModelRequest, ModelResponse
from domain.contracts import ArchitectureDiscovery, RepositorySnapshot, RoleProposal, RunBudget
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
from infrastructure.db.database import database_connection
from infrastructure.db.planning_repository import SqlitePlanningRepository
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository
from infrastructure.db.task_repository import SqliteTaskRepository
from workflow.task_scheduler import NativeTaskScheduler


class ValidGateway:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        assert "host chooses tools" in request.system_prompt
        assert request.correlation["kind"] == "role-proposals"
        return ModelResponse(
            content={
                "roles": [
                    {
                        "name": "API Boundary Specialist",
                        "mission": "Review the documented API module.",
                        "rationale": "The architecture model identifies an API boundary.",
                        "coverage_targets": ["api", "architecture"],
                        "required_capabilities": ["architecture"],
                        "focus_paths": ["api.py"],
                    }
                ],
                "unknowns": [],
            },
            provider="fixture",
            model=request.model,
        )


class InvalidGateway:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content={"roles": []}, provider="fixture", model=request.model)


def _discovery(status: str = "model-validated") -> ArchitectureDiscovery:
    return ArchitectureDiscovery(
        discovery_id="discovery-1",
        run_id="run-1",
        snapshot_id="snapshot-1",
        input_hash="input",
        prompt_template="architecture-discovery",
        prompt_version=1,
        prompt_path="prompts/core/phases/planning/architecture-discovery.md",
        prompt_content_hash="hash",
        prompt_text="prompt",
        result={"modules": []},
        result_hash="architecture-hash",
        status=status,
    )


def test_ai_role_proposals_are_bounded_and_use_the_editable_template() -> None:
    assert PROMPT_PATH.is_file()
    assert "host chooses tools" in role_proposal_system_prompt()
    output = asyncio.run(
        AiRoleProposalService(gateway=ValidGateway(), model="fixture").propose(
            run_id="run-1", discovery=_discovery(), budget=RunBudget(max_specialists=3)
        )
    )
    assert output is not None
    assert output.roles[0].name == "API Boundary Specialist"
    assert output.roles[0].focus_paths == ("api.py",)
    assert (
        asyncio.run(
            AiRoleProposalService(gateway=ValidGateway(), model="fixture").propose(
                run_id="run-1", discovery=_discovery("fallback"), budget=RunBudget()
            )
        )
        is None
    )
    assert (
        asyncio.run(
            AiRoleProposalService(gateway=InvalidGateway(), model="fixture").propose(
                run_id="run-1", discovery=_discovery(), budget=RunBudget()
            )
        )
        is None
    )


def test_host_converts_only_valid_ai_proposals_to_trusted_roles() -> None:
    snapshot = RepositorySnapshot(
        snapshot_id="snapshot-1",
        project_id="project-1",
        canonical_path="fixture",
        identity_hash="identity",
        configuration_hash="configuration",
        scanner_version="test",
        created_at=datetime.now(UTC),
    )
    proposal = RoleProposal(
        proposal_id="proposal-1",
        run_id="run-1",
        wave_number=1,
        proposal_hash="proposal-hash",
        name="API Boundary Specialist",
        mission="Review the API boundary.",
        rationale="The architecture model identifies one.",
        coverage_targets=("api",),
        required_capabilities=("architecture",),
        focus_paths=("api.py",),
        validation_status="approved",
    )
    files = [
        {
            "file_id": "file-1",
            "artifact_id": "artifact-1",
            "relative_path": "api.py",
            "language": "python",
            "classification": "source",
        }
    ]
    planner = RepositoryRolePlanner()
    assert planner.validate_role_proposal(proposal, allowed_paths={"api.py"}) is None
    plan = planner.plan_from_role_proposals(
        run_id="run-1",
        snapshot=snapshot,
        files=files,
        target_paths=("api.py",),
        budget=RunBudget(max_specialists=2, max_tasks=2),
        wave_number=1,
        proposals=(proposal,),
    )
    assert plan.rationale == "validated architecture-informed AI role proposals"
    assert plan.roles[0].name == proposal.name
    assert plan.tasks[0].immutable_input_ids == ("snapshot-1", "file-1")
    outside = proposal.model_copy(update={"focus_paths": ("outside.py",)})
    assert planner.validate_role_proposal(outside, allowed_paths={"api.py"})
    with pytest.raises(ValueError, match="first wave"):
        planner.plan_from_role_proposals(
            run_id="run-1",
            snapshot=snapshot,
            files=files,
            target_paths=("api.py",),
            budget=RunBudget(),
            wave_number=2,
            proposals=(proposal,),
        )
    with pytest.raises(ValueError, match="no approved"):
        planner.plan_from_role_proposals(
            run_id="run-1",
            snapshot=snapshot,
            files=files,
            target_paths=("api.py",),
            budget=RunBudget(),
            wave_number=1,
            proposals=(),
        )


def test_coordinator_persists_ai_proposal_verdicts_before_dispatch(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "planning.db")
    try:
        ledger.create_run(
            run_id="run-1",
            submission_key="submission-1",
            request={"project_path": "fixture"},
            mode="quick",
        )
        tasks = SqliteTaskRepository(ledger)
        coordinator = NativeAnalysisCoordinator(
            ledger=ledger,
            tasks=tasks,
            snapshots=SqliteSnapshotRepository(ledger),
            analysis=SqliteAnalysisRepository(ledger),
            scheduler=NativeTaskScheduler(tasks, {}),
            planning=SqlitePlanningRepository(ledger),
        )
        snapshot = RepositorySnapshot(
            snapshot_id="snapshot-1",
            project_id="project-1",
            canonical_path="fixture",
            identity_hash="identity",
            configuration_hash="configuration",
            scanner_version="test",
            created_at=datetime.now(UTC),
        )
        files = [
            {
                "file_id": "file-1",
                "artifact_id": "artifact-1",
                "relative_path": "api.py",
                "language": "python",
                "classification": "source",
            }
        ]
        accepted = ProposedRole(
            name="API Specialist",
            mission="Review the API.",
            rationale="The model identified the API.",
            coverage_targets=("api",),
            required_capabilities=("architecture",),
            focus_paths=("api.py",),
        )
        rejected = accepted.model_copy(
            update={"name": "Outside", "focus_paths": ("nope.py",)}
        )
        plan = coordinator._validated_ai_plan(
            run_id="run-1",
            snapshot=snapshot,
            files=files,
            target_paths=("api.py",),
            budget=RunBudget(max_specialists=2, max_tasks=2),
            proposals=(accepted, rejected),
        )
        assert plan is not None
        assert [role.name for role in plan.roles] == ["API Specialist"]
        with database_connection(ledger.database_path) as connection:
            verdicts = [
                row[0]
                for row in connection.execute(
                    "SELECT validation_status FROM role_proposals ORDER BY validation_status"
                ).fetchall()
            ]
        assert verdicts == ["approved", "rejected"]
        assert (
            coordinator._validated_ai_plan(
                run_id="run-1",
                snapshot=snapshot,
                files=files,
                target_paths=("api.py",),
                budget=RunBudget(max_specialists=1, max_tasks=1),
                proposals=(rejected,),
            )
            is None
        )
    finally:
        ledger.close()
