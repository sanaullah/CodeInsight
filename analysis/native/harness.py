"""Trusted specialist host and untrusted typed proposal boundary."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from analysis.native.planning import RepositoryRolePlanner
from application.model_gateway import ModelBudgetExceeded
from domain.contracts import (
    EvidenceRef,
    FindingCandidate,
    FindingVerdict,
    RoleSpec,
)
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
from infrastructure.db.artifact_repository import SqliteArtifactRepository
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository
from infrastructure.db.task_repository import TaskLease
from workflow.task_scheduler import PermanentTaskError, TaskContext, TaskResult


class ProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceProposal(ProposalModel):
    relative_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    evidence_kind: str = "source-span"


class FindingProposal(ProposalModel):
    category: str
    concept_id: str
    title: str
    claim: str
    evidence_indexes: tuple[int, ...]
    affected_path: str | None = None
    impact: str
    severity: Literal["info", "low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    uncertainty: str | None = None
    missing_context: tuple[str, ...] = ()
    recommendation: str
    fingerprint_inputs: tuple[str, ...] = ()


class SpecialistOutput(ProposalModel):
    analyzed_paths: tuple[str, ...]
    evidence: tuple[EvidenceProposal, ...] = ()
    findings: tuple[FindingProposal, ...] = ()
    unresolved_uncertainty: tuple[str, ...] = ()
    usage: dict[str, int | float] = Field(default_factory=dict)


class SpecialistFile(ProposalModel):
    file_id: str
    relative_path: str
    language: str
    classification: str
    support_tier: str
    content: str


class SpecialistRequest(ProposalModel):
    run_id: str
    wave_id: str
    task_id: str
    role: RoleSpec
    files: tuple[SpecialistFile, ...]
    token_budget: int
    cost_budget_usd: float
    time_budget_seconds: int


class SpecialistClient(Protocol):
    async def analyze(
        self, request: SpecialistRequest
    ) -> SpecialistOutput | dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ProcessedSpecialistResult:
    evidence: tuple[EvidenceRef, ...]
    candidates: tuple[FindingCandidate, ...]
    verdicts: tuple[FindingVerdict, ...]
    fingerprints: dict[str, str]
    unresolved_uncertainty: tuple[str, ...]
    analyzed_paths: tuple[str, ...]


class EvidencePipeline:
    """Derive trusted evidence and verdicts from untrusted proposals."""

    verifier_version = "native-evidence-v1"

    def process(
        self,
        *,
        lease: TaskLease,
        role: RoleSpec,
        snapshot_id: str,
        output: SpecialistOutput,
        contexts: list[dict[str, Any]],
        source_bodies: dict[str, bytes],
    ) -> ProcessedSpecialistResult:
        by_path = {str(item["relative_path"]): item for item in contexts}
        assigned_paths = set(by_path)
        if not set(output.analyzed_paths).issubset(assigned_paths):
            raise ValueError("specialist reported analysis outside assigned scope")
        evidence: list[EvidenceRef] = []
        evidence_by_index: dict[int, EvidenceRef] = {}
        for index, proposal in enumerate(output.evidence):
            context = by_path.get(proposal.relative_path)
            if context is None:
                continue
            lines = source_bodies[proposal.relative_path].decode(
                "utf-8", errors="replace"
            ).splitlines()
            if (
                proposal.end_line < proposal.start_line
                or proposal.end_line > len(lines)
            ):
                continue
            excerpt = "\n".join(
                lines[proposal.start_line - 1 : proposal.end_line]
            ).encode()
            excerpt_hash = hashlib.sha256(excerpt).hexdigest()
            evidence_id = _stable_hash(
                snapshot_id,
                str(context["file_id"]),
                str(proposal.start_line),
                str(proposal.end_line),
                proposal.evidence_kind,
                excerpt_hash,
            )
            item = EvidenceRef(
                evidence_id=evidence_id,
                snapshot_id=snapshot_id,
                file_id=str(context["file_id"]),
                content_hash=str(context["content_hash"]),
                start_line=proposal.start_line,
                end_line=proposal.end_line,
                excerpt_hash=excerpt_hash,
                evidence_kind=proposal.evidence_kind,
                provenance={
                    "run_id": lease.run_id,
                    "wave_id": lease.wave_id,
                    "task_id": lease.task_id,
                    "role_id": role.role_id,
                    "relative_path": proposal.relative_path,
                    "verifier": self.verifier_version,
                },
                collected_at=datetime.now(UTC),
            )
            evidence.append(item)
            evidence_by_index[index] = item

        candidates: list[FindingCandidate] = []
        verdicts: list[FindingVerdict] = []
        fingerprints: dict[str, str] = {}
        for index, proposal in enumerate(output.findings):
            if (
                proposal.affected_path is not None
                and proposal.affected_path not in assigned_paths
            ):
                raise ValueError("finding affected path is outside assigned scope")
            linked = tuple(
                evidence_by_index[evidence_index].evidence_id
                for evidence_index in proposal.evidence_indexes
                if evidence_index in evidence_by_index
            )
            candidate_id = _stable_hash(lease.task_id, f"candidate:{index}")
            fingerprint = _finding_fingerprint(proposal)
            candidate = FindingCandidate(
                candidate_id=candidate_id,
                producer_task_id=lease.task_id,
                producer_role_id=role.role_id,
                category=proposal.category,
                concept_id=proposal.concept_id,
                title=proposal.title,
                claim=proposal.claim,
                evidence_ids=linked,
                affected_path=proposal.affected_path,
                impact=proposal.impact,
                proposed_severity=proposal.severity,
                proposed_confidence=proposal.confidence,
                uncertainty=proposal.uncertainty,
                missing_context=proposal.missing_context,
                recommendation=proposal.recommendation,
                fingerprint_inputs=proposal.fingerprint_inputs,
            )
            integrity = (
                "valid"
                if linked and len(linked) == len(set(proposal.evidence_indexes))
                else "incomplete"
            )
            accepted = integrity == "valid" and proposal.confidence >= 0.4
            disposition = "accepted" if accepted else "needs-more-evidence"
            verdict = FindingVerdict(
                verdict_id=_stable_hash(candidate_id, self.verifier_version),
                candidate_id=candidate_id,
                disposition=disposition,
                evidence_integrity=integrity,
                contradiction_result="not-observed",
                reachability_result=(
                    "evidence-bound" if linked else "not-demonstrated"
                ),
                calibrated_confidence=proposal.confidence if linked else 0,
                calibrated_severity=proposal.severity,
                verifier="native-evidence-verifier",
                verifier_version=self.verifier_version,
                rationale=(
                    "all referenced spans were verified against immutable source"
                    if accepted
                    else "one or more evidence spans were missing or invalid"
                ),
            )
            candidates.append(candidate)
            verdicts.append(verdict)
            fingerprints[candidate_id] = fingerprint
        return ProcessedSpecialistResult(
            evidence=tuple(evidence),
            candidates=tuple(candidates),
            verdicts=tuple(verdicts),
            fingerprints=fingerprints,
            unresolved_uncertainty=output.unresolved_uncertainty,
            analyzed_paths=output.analyzed_paths,
        )


class TrustedSpecialistHarness:
    """Execute one generated role through a fixed, validated host boundary."""

    def __init__(
        self,
        *,
        client: SpecialistClient,
        snapshots: SqliteSnapshotRepository,
        analysis: SqliteAnalysisRepository,
        artifacts: FilesystemArtifactStore,
        artifact_records: SqliteArtifactRepository,
        max_context_bytes: int = 4_000_000,
        max_findings: int = 100,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.snapshots = snapshots
        self.analysis = analysis
        self.artifacts = artifacts
        self.artifact_records = artifact_records
        self.max_context_bytes = max_context_bytes
        self.max_findings = max_findings
        self.pipeline = EvidencePipeline()
        self.event_sink = event_sink or (lambda _event, _data: None)

    def handler(
        self,
    ) -> Callable[[TaskLease, TaskContext], Awaitable[TaskResult]]:
        return self.execute

    async def execute(self, lease: TaskLease, context: TaskContext) -> TaskResult:
        if lease.role_id is None:
            raise PermanentTaskError("specialist task has no durable role")
        role = self.analysis.get_role(lease.role_id)
        if role is None:
            raise PermanentTaskError(f"unknown durable role: {lease.role_id}")
        try:
            RepositoryRolePlanner.validate_role(role)
        except ValueError as exc:
            raise PermanentTaskError(str(exc)) from exc
        immutable_ids = tuple(lease.input.get("immutable_input_ids", ()))
        if len(immutable_ids) < 2:
            raise PermanentTaskError("specialist task has no snapshot files")
        snapshot_id = str(immutable_ids[0])
        file_ids = tuple(str(value) for value in immutable_ids[1:])
        contexts = self.snapshots.file_contexts_by_ids(snapshot_id, file_ids)
        if len(contexts) != len(set(file_ids)):
            raise PermanentTaskError(
                "specialist task referenced files outside its snapshot"
            )
        source_bodies: dict[str, bytes] = {}
        specialist_files: list[SpecialistFile] = []
        total_bytes = 0
        for item in contexts:
            body = await asyncio.to_thread(
                _read_verified_source,
                str(item["storage_path"]),
                str(item["content_hash"]),
            )
            total_bytes += len(body)
            if total_bytes > self.max_context_bytes:
                raise PermanentTaskError(
                    "specialist context exceeds trusted byte budget"
                )
            path = str(item["relative_path"])
            source_bodies[path] = body
            specialist_files.append(
                SpecialistFile(
                    file_id=str(item["file_id"]),
                    relative_path=path,
                    language=str(item["language"]),
                    classification=str(item["classification"]),
                    support_tier=str(item["support_tier"]),
                    content=body.decode("utf-8", errors="replace"),
                )
            )
        context.raise_if_cancelled()
        try:
            raw_output = await self.client.analyze(
                SpecialistRequest(
                    run_id=lease.run_id,
                    wave_id=lease.wave_id,
                    task_id=lease.task_id,
                    role=role,
                    files=tuple(specialist_files),
                    token_budget=int(
                        lease.budget.get("token_budget", role.token_budget)
                    ),
                    cost_budget_usd=float(lease.budget.get("cost_budget_usd", 0)),
                    time_budget_seconds=int(
                        lease.budget.get(
                            "time_budget_seconds", role.time_budget_seconds
                        )
                    ),
                )
            )
        except ModelBudgetExceeded as exc:
            raise PermanentTaskError(str(exc)) from exc
        context.raise_if_cancelled()
        try:
            output = (
                raw_output
                if isinstance(raw_output, SpecialistOutput)
                else SpecialistOutput.model_validate(raw_output)
            )
        except ValueError as exc:
            raise PermanentTaskError(f"invalid specialist output: {exc}") from exc
        if len(output.findings) > self.max_findings:
            raise PermanentTaskError("specialist exceeded finding count budget")
        try:
            processed = self.pipeline.process(
                lease=lease,
                role=role,
                snapshot_id=snapshot_id,
                output=output,
                contexts=contexts,
                source_bodies=source_bodies,
            )
        except ValueError as exc:
            raise PermanentTaskError(str(exc)) from exc
        self.analysis.persist_specialist_result(
            run_id=lease.run_id,
            task_id=lease.task_id,
            evidence=processed.evidence,
            candidates=processed.candidates,
            verdicts=processed.verdicts,
            fingerprints=processed.fingerprints,
            analyzed_paths=processed.analyzed_paths,
            unresolved_uncertainty=processed.unresolved_uncertainty,
        )
        response_artifact = self.artifacts.put(
            output.model_dump_json().encode(),
            artifact_kind="specialist-response",
            media_type="application/json",
        )
        self.artifact_records.register(response_artifact)
        self.event_sink(
            "specialist_result_verified",
            {
                "run_id": lease.run_id,
                "wave_id": lease.wave_id,
                "task_id": lease.task_id,
                "candidate_count": len(processed.candidates),
                "accepted_count": sum(
                    verdict.disposition == "accepted"
                    for verdict in processed.verdicts
                ),
            },
        )
        return TaskResult(
            output_artifact_id=response_artifact.artifact_id,
            usage=dict(output.usage),
        )


def _finding_fingerprint(proposal: FindingProposal) -> str:
    normalized = [
        proposal.category,
        proposal.concept_id,
        proposal.affected_path or "",
        *(proposal.fingerprint_inputs or (proposal.claim,)),
    ]
    return _stable_hash(
        *(" ".join(value.lower().split()) for value in normalized)
    )


def _read_verified_source(storage_path: str, expected_hash: str) -> bytes:
    path = Path(storage_path)
    stat = path.stat()
    return _cached_source(
        storage_path,
        expected_hash,
        stat.st_mtime_ns,
        stat.st_size,
    )


@lru_cache(maxsize=512)
def _cached_source(
    storage_path: str,
    expected_hash: str,
    _modified_ns: int,
    _byte_size: int,
) -> bytes:
    body = Path(storage_path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise PermanentTaskError(
            f"source artifact integrity failure: {expected_hash}"
        )
    return body


def _stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()
