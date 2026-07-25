from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from analysis.native.coordinator import NativeAnalysisCoordinator
from analysis.native.harness import (
    EvidenceProposal,
    FindingProposal,
    SpecialistOutput,
    SpecialistRequest,
    TrustedSpecialistHarness,
    _cached_source,
    _read_verified_source,
)
from analysis.native.planning import TRUSTED_TOOLS, RepositoryRolePlanner
from domain.contracts import AnalysisMode, RoleSpec, RunBudget
from indexing.repository_index import IndexResult, RepositoryIndexer
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
from infrastructure.db.artifact_repository import SqliteArtifactRepository
from infrastructure.db.database import database_connection
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository
from infrastructure.db.task_repository import SqliteTaskRepository
from workflow.task_scheduler import NativeTaskScheduler


class CompleteClient:
    def __init__(self, *, conflicting: bool = False, delay: float = 0) -> None:
        self.conflicting = conflicting
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()

    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        self.calls += 1
        self.active += 1
        self.started.set()
        self.max_active = max(self.max_active, self.active)
        if self.delay:
            await asyncio.sleep(self.delay)
        evidence = []
        findings = []
        architecture = "architecture" in request.role.required_capabilities
        for index, file in enumerate(request.files):
            evidence.append(
                EvidenceProposal(
                    relative_path=file.relative_path,
                    start_line=1,
                    end_line=1,
                )
            )
            claim = (
                "shared behavior is unsafe"
                if not self.conflicting or architecture
                else "shared behavior is safe"
            )
            findings.append(
                FindingProposal(
                    category="quality",
                    concept_id="shared-rule" if self.conflicting else "file-rule",
                    title="Evidence-backed observation",
                    claim=claim,
                    evidence_indexes=(index,),
                    affected_path=(
                        "app.py" if self.conflicting else file.relative_path
                    ),
                    impact="The behavior can affect maintainability.",
                    severity="medium",
                    confidence=0.9,
                    recommendation="Review the demonstrated behavior.",
                    fingerprint_inputs=(
                        ("shared",)
                        if self.conflicting
                        else ("file-rule", file.relative_path)
                    ),
                )
            )
        self.active -= 1
        return SpecialistOutput(
            analyzed_paths=tuple(file.relative_path for file in request.files),
            evidence=tuple(evidence),
            findings=tuple(findings),
            usage={"tokens": 50},
        )


class InvalidScopeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, _request: SpecialistRequest) -> SpecialistOutput:
        self.calls += 1
        return SpecialistOutput(analyzed_paths=("outside.py",))


class InvalidEvidenceClient:
    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        file = request.files[0]
        return SpecialistOutput(
            analyzed_paths=(file.relative_path,),
            evidence=(
                EvidenceProposal(
                    relative_path=file.relative_path,
                    start_line=1,
                    end_line=999,
                ),
            ),
            findings=(
                FindingProposal(
                    category="quality",
                    concept_id="bad-span",
                    title="Unverified observation",
                    claim="This claim lacks a valid span.",
                    evidence_indexes=(0,),
                    affected_path=file.relative_path,
                    impact="Unknown.",
                    severity="low",
                    confidence=0.8,
                    recommendation="Gather valid evidence.",
                ),
            ),
        )


class NoFindingsClient:
    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        return SpecialistOutput(
            analyzed_paths=tuple(file.relative_path for file in request.files),
            usage={"tokens": 50},
        )


class UncertainClient:
    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        return SpecialistOutput(
            analyzed_paths=tuple(file.relative_path for file in request.files),
            unresolved_uncertainty=("uncertainty:runtime-configuration",),
        )


class OutsideFindingClient:
    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        file = request.files[0]
        return SpecialistOutput(
            analyzed_paths=(file.relative_path,),
            evidence=(
                EvidenceProposal(
                    relative_path=file.relative_path, start_line=1, end_line=1
                ),
            ),
            findings=(
                FindingProposal(
                    category="quality",
                    concept_id="outside",
                    title="Outside claim",
                    claim="The finding claims an unassigned path.",
                    evidence_indexes=(0,),
                    affected_path="outside.py",
                    impact="Invalid scope.",
                    severity="low",
                    confidence=0.8,
                    recommendation="Reject it.",
                ),
            ),
        )


@dataclass
class Runtime:
    ledger: SqliteRunLedger
    snapshots: SqliteSnapshotRepository
    analysis: SqliteAnalysisRepository
    tasks: SqliteTaskRepository
    index: IndexResult
    artifacts: FilesystemArtifactStore

    def close(self) -> None:
        self.ledger.close()


def _runtime(tmp_path: Path, *, extra_files: int = 0) -> Runtime:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text(
        "from helpers import calculate\n\ndef run():\n    return calculate(2)\n",
        encoding="utf-8",
    )
    (root / "helpers.py").write_text(
        "def calculate(value):\n    return value + 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_app.py").write_text(
        "from app import run\n\ndef test_run():\n    assert run() == 3\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n", encoding="utf-8"
    )
    for number in range(extra_files):
        (root / f"module_{number}.py").write_text(
            f"VALUE = {number}\n", encoding="utf-8"
        )
    ledger = SqliteRunLedger(tmp_path / "application.sqlite3")
    snapshots = SqliteSnapshotRepository(ledger)
    artifacts = FilesystemArtifactStore(tmp_path / "artifacts")
    index = RepositoryIndexer(snapshots, artifacts).build(root)
    return Runtime(
        ledger=ledger,
        snapshots=snapshots,
        analysis=SqliteAnalysisRepository(ledger),
        tasks=SqliteTaskRepository(ledger),
        index=index,
        artifacts=artifacts,
    )


def _create_run(runtime: Runtime, run_id: str = "run-1") -> None:
    runtime.ledger.create_run(
        run_id=run_id,
        submission_key=f"submission-{run_id}",
        request={"project_path": runtime.index.snapshot.canonical_path},
        mode="deep",
        budget=RunBudget().model_dump(mode="json"),
    )
    assert runtime.ledger.mark_running(run_id)


def _coordinator(
    runtime: Runtime,
    client: Any,
    *,
    max_concurrent: int = 4,
) -> NativeAnalysisCoordinator:
    artifact_records = SqliteArtifactRepository(runtime.ledger)
    harness = TrustedSpecialistHarness(
        client=client,
        snapshots=runtime.snapshots,
        analysis=runtime.analysis,
        artifacts=runtime.artifacts,
        artifact_records=artifact_records,
    )
    scheduler = NativeTaskScheduler(
        runtime.tasks,
        {"specialist_analysis": harness.execute},
        max_concurrent=max_concurrent,
        cancellation_poll_seconds=0.001,
    )
    return NativeAnalysisCoordinator(
        ledger=runtime.ledger,
        tasks=runtime.tasks,
        snapshots=runtime.snapshots,
        analysis=runtime.analysis,
        scheduler=scheduler,
    )


def test_repository_planner_creates_dynamic_bounded_trusted_roles(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    try:
        files = runtime.snapshots.list_files(runtime.index.snapshot.snapshot_id)
        plan = RepositoryRolePlanner().plan(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            files=files,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(max_specialists=5, max_tasks=5),
            wave_number=1,
        )

        assert 1 <= len(plan.roles) <= 5
        assert len(plan.tasks) == len(plan.roles)
        capabilities = {
            capability
            for role in plan.roles
            for capability in role.required_capabilities
        }
        assert {"architecture", "language-analysis", "test-quality"} <= capabilities
        assert all(set(role.allowed_tools) == TRUSTED_TOOLS for role in plan.roles)
        assert all(
            task.immutable_input_ids[0] == runtime.index.snapshot.snapshot_id
            for task in plan.tasks
        )
    finally:
        runtime.close()


def test_planner_rejects_untrusted_generated_role() -> None:
    role = RoleSpec(
        role_id="bad",
        name="Bad role",
        mission="Escape",
        rationale="Untrusted",
        coverage_targets=("x",),
        required_capabilities=("architecture",),
        allowed_tools=("shell.execute",),
        model_policy="strong",
        token_budget=1,
        time_budget_seconds=1,
        completion_criteria=("done",),
    )
    with pytest.raises(ValueError, match="untrusted tool"):
        RepositoryRolePlanner.validate_role(role)


def test_verified_source_cache_reuses_unchanged_body_and_detects_tamper(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    body = b"VALUE = 1\n"
    source.write_bytes(body)
    expected = hashlib.sha256(body).hexdigest()
    _cached_source.cache_clear()
    assert _read_verified_source(str(source), expected) == body
    assert _read_verified_source(str(source), expected) == body
    assert _cached_source.cache_info().hits == 1
    source.write_bytes(b"VALUE = 22\n")
    with pytest.raises(RuntimeError, match="integrity failure"):
        _read_verified_source(str(source), expected)


def test_security_mode_prioritizes_security_with_one_specialist(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    try:
        plan = RepositoryRolePlanner().plan(
            run_id="run-security",
            snapshot=runtime.index.snapshot,
            files=runtime.snapshots.list_files(runtime.index.snapshot.snapshot_id),
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.SECURITY,
            budget=RunBudget(max_specialists=1, max_tasks=1),
            wave_number=1,
        )
        assert plan.roles[0].required_capabilities == ("security",)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_native_analysis_executes_parallel_specialists_and_persists_results(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    client = CompleteClient(delay=0.01)
    coordinator = _coordinator(runtime, client, max_concurrent=3)
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(
                max_specialists=5,
                max_tasks=10,
                max_waves=2,
                max_tokens=100_000,
            ),
        )

        assert result.wave_count == 1
        assert result.coverage[-1].follow_up_decision == "complete"
        assert result.coverage[-1].measured["file_evidence_coverage"] == 1
        assert result.findings
        assert client.max_active >= 2
        with database_connection(runtime.ledger.database_path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) AS count FROM evidence_refs"
                ).fetchone()["count"]
                >= 4
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) AS count FROM artifacts "
                    "WHERE artifact_kind = 'specialist-response'"
                ).fetchone()["count"]
                >= 1
            )
            stage = connection.execute(
                "SELECT current_stage FROM runs WHERE run_id = 'run-1'"
            ).fetchone()["current_stage"]
            assert stage == "complete"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_coordinator_lease_rejects_concurrent_duplicate_driver(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    client = CompleteClient(delay=0.05)
    first = _coordinator(runtime, client)
    second = _coordinator(runtime, client)
    budget = RunBudget(max_specialists=2, max_tasks=2, max_waves=1)
    try:
        running = asyncio.create_task(
            first.execute(
                run_id="run-1",
                snapshot=runtime.index.snapshot,
                target_paths=runtime.index.target_paths,
                mode=AnalysisMode.QUICK,
                budget=budget,
            )
        )
        await asyncio.wait_for(client.started.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="already coordinated"):
            await second.execute(
                run_id="run-1",
                snapshot=runtime.index.snapshot,
                target_paths=runtime.index.target_paths,
                mode=AnalysisMode.QUICK,
                budget=budget,
            )
        assert (await running).wave_count == 1
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_expired_coordinator_lease_is_recoverable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    try:
        assert runtime.analysis.acquire_coordinator_lease(
            "run-1", owner="dead", lease_seconds=0.001
        )
        await asyncio.sleep(0.01)
        assert runtime.analysis.acquire_coordinator_lease(
            "run-1", owner="recovery", lease_seconds=1
        )
        assert not runtime.analysis.release_coordinator_lease(
            "run-1", owner="dead", status="failed"
        )
        assert runtime.analysis.release_coordinator_lease(
            "run-1", owner="recovery", status="completed"
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deterministic_deduplication_records_conflicting_candidates(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    coordinator = _coordinator(runtime, CompleteClient(conflicting=True))
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(max_specialists=4, max_tasks=8, max_waves=1),
        )

        shared = [
            finding
            for finding in result.findings
            if finding.claim in {"shared behavior is unsafe", "shared behavior is safe"}
        ]
        assert len(shared) == 1
        assert shared[0].conflicting_candidate_ids
        first_fingerprint = shared[0].fingerprint
        repeated = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(max_specialists=4, max_tasks=8, max_waves=1),
        )
        assert repeated.findings[0].fingerprint == first_fingerprint
        assert repeated.wave_count == 1
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_invalid_specialist_scope_retries_then_bounds_follow_up(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    client = InvalidScopeClient()
    coordinator = _coordinator(runtime, client)
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(max_specialists=2, max_tasks=4, max_waves=2),
        )

        assert result.wave_count == 2
        assert result.coverage[0].follow_up_decision == "launch"
        assert result.coverage[0].proposed_follow_up_task_ids
        assert result.coverage[-1].follow_up_decision == "defer"
        assert not result.findings
        assert client.calls == 4
        with database_connection(runtime.ledger.database_path) as connection:
            failed = connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE status = 'failed'"
            ).fetchone()["count"]
            assert failed == 4
            assert {
                row["attempt_count"]
                for row in connection.execute(
                    "SELECT attempt_count FROM tasks"
                ).fetchall()
            } == {1}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_invalid_evidence_is_not_promoted_and_triggers_bounded_gap(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    coordinator = _coordinator(runtime, InvalidEvidenceClient())
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.SECURITY,
            budget=RunBudget(max_specialists=1, max_tasks=2, max_waves=2),
        )

        assert not result.findings
        assert result.coverage[0].follow_up_decision == "launch"
        assert "uncertainty:evidence-validity" in result.coverage[0].remaining_gaps
        assert result.coverage[-1].follow_up_decision == "defer"
        with database_connection(runtime.ledger.database_path) as connection:
            verdicts = connection.execute(
                "SELECT contract_json FROM finding_verdicts"
            ).fetchall()
            assert all(
                '"disposition":"needs-more-evidence"' in row["contract_json"]
                for row in verdicts
            )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_clean_specialist_results_complete_without_fabricated_findings(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    coordinator = _coordinator(runtime, NoFindingsClient())
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.QUICK,
            budget=RunBudget(max_specialists=2, max_tasks=4, max_waves=2),
        )

        assert result.wave_count == 1
        assert result.coverage[-1].follow_up_decision == "complete"
        assert result.coverage[-1].measured["file_analysis_coverage"] == 1
        assert result.coverage[-1].measured["file_evidence_coverage"] == 0
        assert not result.findings
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_specialist_uncertainty_is_durable_coverage_input(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    coordinator = _coordinator(runtime, UncertainClient())
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.QUICK,
            budget=RunBudget(max_specialists=1, max_tasks=1, max_waves=1),
        )

        assert (
            "uncertainty:runtime-configuration"
            in result.coverage[-1].unresolved_uncertainty
        )
        assert result.coverage[-1].follow_up_decision == "defer"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_finding_cannot_claim_a_path_outside_assigned_scope(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    coordinator = _coordinator(runtime, OutsideFindingClient())
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.QUICK,
            budget=RunBudget(max_specialists=1, max_tasks=1, max_waves=1),
        )

        task = runtime.analysis.task_records(
            "run-1", result.coverage[-1].wave_id
        )[0]
        assert task["status"] == "failed"
        assert "outside assigned scope" in task["error"]["message"]
        assert not result.findings
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_tampered_source_artifact_fails_before_specialist_call(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _create_run(runtime)
    first_file = runtime.snapshots.file_contexts(
        runtime.index.snapshot.snapshot_id,
        (runtime.index.target_paths[0],),
    )[0]
    Path(first_file["storage_path"]).write_bytes(b"tampered")  # noqa: ASYNC240
    client = CompleteClient()
    coordinator = _coordinator(runtime, client)
    try:
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.QUICK,
            budget=RunBudget(max_specialists=1, max_tasks=1, max_waves=1),
        )

        assert result.coverage[-1].follow_up_decision == "defer"
        assert client.calls == 0
        assert runtime.analysis.task_records(
            "run-1", result.coverage[-1].wave_id
        )[0]["status"] == "failed"
        with database_connection(runtime.ledger.database_path) as connection:
            assert (
                connection.execute(
                    "SELECT attempt_count FROM tasks"
                ).fetchone()["attempt_count"]
                == 1
            )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_native_analysis_performance_smoke(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, extra_files=100)
    _create_run(runtime)
    client = NoFindingsClient()
    coordinator = _coordinator(runtime, client)
    try:
        started = time.perf_counter()
        result = await coordinator.execute(
            run_id="run-1",
            snapshot=runtime.index.snapshot,
            target_paths=runtime.index.target_paths,
            mode=AnalysisMode.DEEP,
            budget=RunBudget(max_specialists=5, max_tasks=5, max_waves=1),
        )
        elapsed = time.perf_counter() - started

        assert result.coverage[-1].follow_up_decision == "complete"
        assert elapsed < 5
        with database_connection(runtime.ledger.database_path) as connection:
            usages = connection.execute(
                "SELECT usage_json FROM task_attempts WHERE status = 'succeeded'"
            ).fetchall()
            assert len(usages) == 5
            assert all('"tokens":50' in row["usage_json"] for row in usages)
    finally:
        runtime.close()
