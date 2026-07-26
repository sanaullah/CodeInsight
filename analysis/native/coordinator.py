"""Durable outer workflow for native role waves and evidence analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from analysis.native.architecture_discovery import ArchitectureDiscoveryService
from analysis.native.pipeline import (
    CoverageAssessor,
    CoverageInputs,
    correlate_and_persist,
)
from analysis.native.planning import RepositoryRolePlanner
from analysis.native.role_generation import AiRoleProposalService, ProposedRole
from domain.contracts import (
    AnalysisMode,
    CanonicalFinding,
    CoverageAssessment,
    RepositorySnapshot,
    RoleProposal,
    RunBudget,
    RunStage,
    WavePlan,
)
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
from infrastructure.db.planning_repository import SqlitePlanningRepository
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository
from infrastructure.db.task_repository import SqliteTaskRepository
from workflow.task_scheduler import NativeTaskScheduler


@dataclass(frozen=True, slots=True)
class NativeAnalysisResult:
    findings: tuple[CanonicalFinding, ...]
    coverage: tuple[CoverageAssessment, ...]
    wave_count: int
    failed_task_count: int = 0


class NativeAnalysisCoordinator:
    """Keep durable wave state outside adaptive specialist execution."""

    def __init__(
        self,
        *,
        ledger: SqliteRunLedger,
        tasks: SqliteTaskRepository,
        snapshots: SqliteSnapshotRepository,
        analysis: SqliteAnalysisRepository,
        scheduler: NativeTaskScheduler,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        planner: RepositoryRolePlanner | None = None,
        architecture_discovery: ArchitectureDiscoveryService | None = None,
        role_proposals: AiRoleProposalService | None = None,
        planning: SqlitePlanningRepository | None = None,
        assessor: CoverageAssessor | None = None,
    ) -> None:
        self.ledger = ledger
        self.tasks = tasks
        self.snapshots = snapshots
        self.analysis = analysis
        self.scheduler = scheduler
        self.event_sink = event_sink or (
            lambda event_type, data: self.ledger.append_event(
                str(data.get("run_id", "")), event_type, data
            )
        )
        self.planner = planner or RepositoryRolePlanner()
        self.architecture_discovery = architecture_discovery
        self.role_proposals = role_proposals
        self.planning = planning
        self.assessor = assessor or CoverageAssessor()

    async def execute(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        target_paths: tuple[str, ...],
        mode: AnalysisMode,
        budget: RunBudget,
    ) -> NativeAnalysisResult:
        owner = f"coordinator-{uuid4().hex}"
        if not self.analysis.acquire_coordinator_lease(run_id, owner=owner):
            raise RuntimeError("native analysis run is already coordinated")
        try:
            result = await self._execute_acquired(
                run_id=run_id,
                snapshot=snapshot,
                target_paths=target_paths,
                mode=mode,
                budget=budget,
            )
        except BaseException as exc:
            self.analysis.release_coordinator_lease(
                run_id, owner=owner, status="failed", error=str(exc)
            )
            raise
        self.analysis.release_coordinator_lease(run_id, owner=owner, status="completed")
        return result

    async def _execute_acquired(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        target_paths: tuple[str, ...],
        mode: AnalysisMode,
        budget: RunBudget,
    ) -> NativeAnalysisResult:
        if not self.analysis.bind_snapshot(run_id, snapshot.snapshot_id):
            raise ValueError("run is not active or is bound to another snapshot")
        files = self.snapshots.list_files(snapshot.snapshot_id)
        by_path = {str(item["relative_path"]): item for item in files}
        targets = [by_path[path] for path in target_paths if path in by_path] or files
        existing_coverage = self.analysis.list_coverage(run_id)
        if existing_coverage and existing_coverage[-1].follow_up_decision != "launch":
            return NativeAnalysisResult(
                findings=tuple(self.analysis.list_canonical_findings(run_id)),
                coverage=tuple(existing_coverage),
                wave_count=self.analysis.wave_count(run_id),
                failed_task_count=sum(
                    record["status"] != "succeeded"
                    for assessment in existing_coverage
                    for record in self.analysis.task_records(run_id, assessment.wave_id)
                ),
            )
        total_tasks = sum(
            len(self.analysis.task_records(run_id, assessment.wave_id))
            for assessment in existing_coverage
        )
        coverage: list[CoverageAssessment] = list(existing_coverage)
        next_plan = self.analysis.incomplete_wave_plan(run_id)
        resuming_wave_id = next_plan.wave_id if next_plan else None
        if next_plan is not None:
            total_tasks += len(next_plan.tasks)
        remaining_gaps: tuple[str, ...] = (
            existing_coverage[-1].remaining_gaps if existing_coverage else ()
        )
        wave_number = (
            next_plan.wave_number
            if next_plan is not None
            else max(1, self.analysis.wave_count(run_id) + 1)
        )
        failed_task_count = 0

        while wave_number <= budget.max_waves and (
            resuming_wave_id is not None or total_tasks < budget.max_tasks
        ):
            self._stage(run_id, RunStage.PLAN_WAVE)
            ai_plan: WavePlan | None = None
            if self.architecture_discovery is not None and wave_number == 1 and next_plan is None:
                discovery = await self.architecture_discovery.discover(
                    run_id=run_id,
                    snapshot=snapshot,
                    files=files,
                    budget=budget,
                )
                self.event_sink(
                    "architecture_discovery_completed",
                    {"run_id": run_id, "status": discovery.status},
                )
                if self.role_proposals is not None and self.planning is not None:
                    proposal_output = await self.role_proposals.propose(
                        run_id=run_id, discovery=discovery, budget=budget
                    )
                    if proposal_output is None:
                        self.event_sink(
                            "ai_role_proposals_unavailable",
                            {
                                "run_id": run_id,
                                "reason": "architecture discovery was unavailable or invalid",
                            },
                        )
                    else:
                        ai_plan = self._validated_ai_plan(
                            run_id=run_id,
                            snapshot=snapshot,
                            files=files,
                            target_paths=tuple(str(item["relative_path"]) for item in targets),
                            budget=budget,
                            proposals=proposal_output.roles,
                        )
            if resuming_wave_id is not None:
                wave_budget = budget
            else:
                remaining_task_budget = budget.max_tasks - total_tasks
                wave_budget = budget.model_copy(
                    update={"max_specialists": min(budget.max_specialists, remaining_task_budget)}
                )
            plan = next_plan or ai_plan or self.planner.plan(
                run_id=run_id,
                snapshot=snapshot,
                files=files,
                target_paths=tuple(str(item["relative_path"]) for item in targets),
                mode=mode,
                budget=wave_budget,
                wave_number=wave_number,
                remaining_gaps=remaining_gaps,
            )
            next_plan = None
            if plan.wave_id == resuming_wave_id:
                self.event_sink(
                    "wave_resumed",
                    {
                        "run_id": run_id,
                        "wave_id": plan.wave_id,
                        "wave_number": plan.wave_number,
                        "task_count": len(plan.tasks),
                    },
                )
                resuming_wave_id = None
            else:
                self._persist_plan(plan, wave_budget)
                total_tasks += len(plan.tasks)
            self._stage(run_id, RunStage.DISPATCH_TASKS)
            await self.scheduler.run_until_idle(run_id=run_id)
            self._stage(run_id, RunStage.VERIFY_EVIDENCE)
            self._stage(run_id, RunStage.DEDUPLICATE_AND_CORRELATE)
            correlate_and_persist(self.analysis, run_id)
            self._stage(run_id, RunStage.ASSESS_COVERAGE)
            task_records = self.analysis.task_records(run_id, plan.wave_id)
            failed_task_count += sum(record["status"] != "succeeded" for record in task_records)
            verdict_count, accepted_count = self.analysis.wave_verdict_counts(plan.wave_id)
            wave_summary = self.analysis.wave_analysis_summary(plan.wave_id)
            assessment = self.assessor.assess(
                plan=plan,
                task_records=task_records,
                inputs=CoverageInputs(
                    target_file_ids=tuple(str(item["file_id"]) for item in targets),
                    analyzed_file_ids=tuple(
                        str(by_path[path]["file_id"])
                        for path in wave_summary["analyzed_paths"]
                        if path in by_path
                    ),
                    evidence_file_ids=self.analysis.wave_evidence_file_ids(plan.wave_id),
                    verdict_count=verdict_count,
                    accepted_count=accepted_count,
                    unsupported_areas=tuple(
                        sorted(
                            {
                                f"language:{item['language']}"
                                for item in targets
                                if item["support_tier"] == "discovery"
                            }
                        )
                    ),
                    reported_uncertainty=wave_summary["unresolved_uncertainty"],
                ),
                mode=mode,
                budget=budget,
            )
            remaining_gaps = assessment.remaining_gaps
            if (
                assessment.follow_up_decision == "launch"
                and wave_number < budget.max_waves
                and total_tasks < budget.max_tasks
            ):
                self._stage(run_id, RunStage.PLAN_FOLLOW_UP_WAVE)
                follow_up_budget = budget.model_copy(
                    update={
                        "max_specialists": min(
                            budget.max_specialists,
                            budget.max_tasks - total_tasks,
                        )
                    }
                )
                next_plan = self.planner.plan(
                    run_id=run_id,
                    snapshot=snapshot,
                    files=files,
                    target_paths=tuple(str(item["relative_path"]) for item in targets),
                    mode=mode,
                    budget=follow_up_budget,
                    wave_number=wave_number + 1,
                    remaining_gaps=remaining_gaps,
                )
                assessment = assessment.model_copy(
                    update={
                        "proposed_follow_up_task_ids": tuple(
                            task.task_id for task in next_plan.tasks
                        )
                    }
                )
            self.analysis.persist_coverage(assessment)
            self.analysis.complete_wave(
                plan.wave_id,
                status=(
                    "completed"
                    if all(record["status"] == "succeeded" for record in task_records)
                    else "completed_with_failures"
                ),
            )
            coverage.append(assessment)
            if next_plan is None:
                break
            wave_number += 1

        self._stage(run_id, RunStage.SYNTHESIZE)
        findings = tuple(self.analysis.list_canonical_findings(run_id))
        self._stage(run_id, RunStage.COMPLETE)
        return NativeAnalysisResult(
            findings=findings,
            coverage=tuple(coverage),
            wave_count=self.analysis.wave_count(run_id),
            failed_task_count=failed_task_count,
        )

    def _validated_ai_plan(
        self,
        *,
        run_id: str,
        snapshot: RepositorySnapshot,
        files: Sequence[dict[str, Any]],
        target_paths: tuple[str, ...],
        budget: RunBudget,
        proposals: Sequence[ProposedRole],
    ) -> WavePlan | None:
        """Persist every candidate, then dispatch only host-approved proposals."""
        if self.planning is None:
            return None
        allowed_paths = set(target_paths)
        approved: list[RoleProposal] = []
        rejected: list[RoleProposal] = []
        for candidate in proposals[: budget.max_specialists]:
            payload = candidate.model_dump(mode="json")
            proposal_hash = _hash_payload(payload)
            proposal_id = _stable_id(run_id, f"role-proposal:{proposal_hash}")
            proposed = RoleProposal(
                proposal_id=proposal_id,
                run_id=run_id,
                wave_number=1,
                proposal_hash=proposal_hash,
                name=candidate.name,
                mission=candidate.mission,
                rationale=candidate.rationale,
                coverage_targets=candidate.coverage_targets,
                required_capabilities=candidate.required_capabilities,
                focus_paths=candidate.focus_paths,
                validation_status="proposed",
            )
            reason = self.planner.validate_role_proposal(
                proposed, allowed_paths=allowed_paths
            )
            if reason is None:
                approved.append(
                    proposed.model_copy(
                        update={"validation_status": "approved"}
                    )
                )
            else:
                rejected.append(
                    proposed.model_copy(
                        update={"validation_status": "rejected", "validation_reason": reason}
                    )
                )
        if not approved:
            for proposal in rejected:
                self.planning.record_role_proposal(proposal)
            self.event_sink(
                "ai_role_proposals_rejected",
                {"run_id": run_id, "proposal_count": len(rejected)},
            )
            return None
        try:
            plan = self.planner.plan_from_role_proposals(
                run_id=run_id,
                snapshot=snapshot,
                files=files,
                target_paths=target_paths,
                budget=budget,
                wave_number=1,
                proposals=approved,
            )
        except ValueError as exc:
            for proposal in rejected:
                self.planning.record_role_proposal(proposal)
            self.event_sink(
                "ai_role_proposals_unusable",
                {"run_id": run_id, "reason": str(exc)},
            )
            return None
        for proposal in (*approved, *rejected):
            self.planning.record_role_proposal(proposal)
        self.event_sink(
            "ai_role_plan_selected",
            {
                "run_id": run_id,
                "approved_count": len(approved),
                "rejected_count": len(rejected),
                "role_count": len(plan.roles),
            },
        )
        return plan

    def _persist_plan(self, plan: WavePlan, budget: RunBudget) -> None:
        self.tasks.create_wave(
            run_id=plan.run_id,
            wave_id=plan.wave_id,
            wave_number=plan.wave_number,
            rationale=plan.rationale,
            budget=budget.model_dump(mode="json"),
        )
        for role in plan.roles:
            self.planner.validate_role(role)
            self.tasks.add_role(run_id=plan.run_id, wave_id=plan.wave_id, role=role)
        for task in plan.tasks:
            self.tasks.enqueue(task)
        self.event_sink(
            "wave_planned",
            {
                "run_id": plan.run_id,
                "wave_id": plan.wave_id,
                "wave_number": plan.wave_number,
                "role_count": len(plan.roles),
                "task_count": len(plan.tasks),
                "coverage_targets": list(plan.coverage_targets),
            },
        )
        for role in plan.roles:
            self.event_sink(
                "role_planned",
                {
                    "run_id": plan.run_id,
                    "wave_id": plan.wave_id,
                    "role_id": role.role_id,
                    "role_name": role.name,
                    "model_policy": role.model_policy,
                },
            )
        for task in plan.tasks:
            self.event_sink(
                "task_enqueued",
                {
                    "run_id": plan.run_id,
                    "wave_id": plan.wave_id,
                    "role_id": task.role_id,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                },
            )

    def _stage(self, run_id: str, stage: RunStage) -> None:
        if not self.ledger.set_stage(run_id, stage):
            raise RuntimeError(f"run left active state before stage {stage.value}")
        self.event_sink(
            "stage_changed",
            {"run_id": run_id, "stage": stage.value},
        )


def _stable_id(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def _hash_payload(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
