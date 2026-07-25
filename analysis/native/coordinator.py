"""Durable outer workflow for native role waves and evidence analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from analysis.native.pipeline import (
    CoverageAssessor,
    CoverageInputs,
    correlate_and_persist,
)
from analysis.native.planning import RepositoryRolePlanner
from domain.contracts import (
    AnalysisMode,
    CanonicalFinding,
    CoverageAssessment,
    RepositorySnapshot,
    RunBudget,
    RunStage,
    WavePlan,
)
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
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
        self.analysis.release_coordinator_lease(
            run_id, owner=owner, status="completed"
        )
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
                    for record in self.analysis.task_records(
                        run_id, assessment.wave_id
                    )
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
            if resuming_wave_id is not None:
                wave_budget = budget
            else:
                remaining_task_budget = budget.max_tasks - total_tasks
                wave_budget = budget.model_copy(
                    update={
                        "max_specialists": min(
                            budget.max_specialists, remaining_task_budget
                        )
                    }
                )
            plan = next_plan or self.planner.plan(
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
            failed_task_count += sum(
                record["status"] != "succeeded" for record in task_records
            )
            verdict_count, accepted_count = self.analysis.wave_verdict_counts(
                plan.wave_id
            )
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
                    evidence_file_ids=self.analysis.wave_evidence_file_ids(
                        plan.wave_id
                    ),
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
                    reported_uncertainty=wave_summary[
                        "unresolved_uncertainty"
                    ],
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
                    target_paths=tuple(
                        str(item["relative_path"]) for item in targets
                    ),
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
            self.tasks.add_role(
                run_id=plan.run_id, wave_id=plan.wave_id, role=role
            )
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
