"""Deterministic correlation and explicit coverage decisions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from domain.contracts import (
    AnalysisMode,
    CanonicalFinding,
    CoverageAssessment,
    FindingCandidate,
    FindingVerdict,
    RunBudget,
    WavePlan,
)
from infrastructure.db.analysis_repository import SqliteAnalysisRepository

_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class FindingCorrelator:
    """Collapse accepted candidates by stable semantic fingerprint."""

    def correlate(
        self,
        run_id: str,
        accepted: Sequence[tuple[FindingCandidate, FindingVerdict, str]],
    ) -> tuple[CanonicalFinding, ...]:
        grouped: dict[
            str, list[tuple[FindingCandidate, FindingVerdict]]
        ] = defaultdict(list)
        for candidate, verdict, fingerprint in accepted:
            grouped[fingerprint].append((candidate, verdict))
        findings = []
        for fingerprint in sorted(grouped):
            members = sorted(
                grouped[fingerprint],
                key=lambda item: (
                    -item[1].calibrated_confidence,
                    -_SEVERITY_ORDER[item[1].calibrated_severity],
                    item[0].candidate_id,
                ),
            )
            representative, _representative_verdict = members[0]
            representative_claim = _normalize(representative.claim)
            supporting = [
                candidate
                for candidate, _verdict in members
                if _normalize(candidate.claim) == representative_claim
            ]
            conflicting = [
                candidate
                for candidate, _verdict in members
                if _normalize(candidate.claim) != representative_claim
            ]
            severity = max(
                (verdict.calibrated_severity for _candidate, verdict in members),
                key=_SEVERITY_ORDER.__getitem__,
            )
            confidence = round(
                sum(verdict.calibrated_confidence for _candidate, verdict in members)
                / len(members),
                4,
            )
            findings.append(
                CanonicalFinding(
                    finding_id=_stable_hash(run_id, fingerprint),
                    fingerprint=fingerprint,
                    title=representative.title,
                    claim=representative.claim,
                    severity=severity,
                    confidence=confidence,
                    candidate_ids=tuple(
                        candidate.candidate_id for candidate in supporting
                    ),
                    supporting_evidence_ids=tuple(
                        sorted(
                            {
                                evidence_id
                                for candidate in supporting
                                for evidence_id in candidate.evidence_ids
                            }
                        )
                    ),
                    conflicting_candidate_ids=tuple(
                        candidate.candidate_id for candidate in conflicting
                    ),
                    recommendation=representative.recommendation,
                )
            )
        return tuple(findings)


@dataclass(frozen=True, slots=True)
class CoverageInputs:
    target_file_ids: tuple[str, ...]
    analyzed_file_ids: tuple[str, ...]
    evidence_file_ids: tuple[str, ...]
    verdict_count: int
    accepted_count: int
    unsupported_areas: tuple[str, ...] = ()
    reported_uncertainty: tuple[str, ...] = ()


class CoverageAssessor:
    """Measure completed work and make one bounded follow-up decision."""

    _thresholds = {
        AnalysisMode.QUICK: 0.5,
        AnalysisMode.DEEP: 0.8,
        AnalysisMode.SECURITY: 0.85,
        AnalysisMode.CHANGE_SET: 0.9,
    }

    def assess(
        self,
        *,
        plan: WavePlan,
        task_records: Sequence[dict[str, object]],
        inputs: CoverageInputs,
        mode: AnalysisMode,
        budget: RunBudget,
    ) -> CoverageAssessment:
        total_tasks = len(task_records)
        succeeded = sum(record["status"] == "succeeded" for record in task_records)
        task_completion = succeeded / total_tasks if total_tasks else 0
        target_files = set(inputs.target_file_ids)
        file_analysis = (
            len(target_files & set(inputs.analyzed_file_ids)) / len(target_files)
            if target_files
            else 1
        )
        file_evidence = (
            len(target_files & set(inputs.evidence_file_ids)) / len(target_files)
            if target_files
            else 1
        )
        verified_rate = (
            inputs.accepted_count / inputs.verdict_count
            if inputs.verdict_count
            else 1
        )
        successful_role_ids = {
            str(record["role_id"])
            for record in task_records
            if record["status"] == "succeeded" and record["role_id"]
        }
        covered_targets = {
            target
            for role in plan.roles
            if role.role_id in successful_role_ids
            for target in role.coverage_targets
        }
        required_targets = set(plan.coverage_targets)
        target_completion = (
            len(covered_targets & required_targets) / len(required_targets)
            if required_targets
            else 1
        )
        remaining = sorted(required_targets - covered_targets)
        threshold = self._thresholds[mode]
        unresolved = list(inputs.reported_uncertainty)
        if verified_rate < threshold:
            unresolved.append("uncertainty:evidence-validity")
        if file_analysis < threshold:
            unresolved.append("uncertainty:file-analysis-coverage")
        remaining.extend(unresolved)
        remaining = sorted(set(remaining))
        can_follow_up = (
            bool(remaining)
            and plan.wave_number < budget.max_waves
            and budget.reserved_follow_up_fraction > 0
        )
        decision = "launch" if can_follow_up else (
            "defer" if remaining else "complete"
        )
        return CoverageAssessment(
            assessment_id=_stable_hash(plan.run_id, plan.wave_id, "coverage"),
            run_id=plan.run_id,
            wave_id=plan.wave_id,
            measured={
                "task_completion": round(task_completion, 4),
                "target_completion": round(target_completion, 4),
                "file_analysis_coverage": round(file_analysis, 4),
                "file_evidence_coverage": round(file_evidence, 4),
                "verified_finding_rate": round(verified_rate, 4),
            },
            unresolved_uncertainty=tuple(unresolved),
            unsupported_areas=inputs.unsupported_areas,
            remaining_gaps=tuple(remaining),
            proposed_follow_up_task_ids=(),
            confidence=round(
                (task_completion + target_completion + verified_rate) / 3,
                4,
            ),
            follow_up_decision=decision,
            decision_rationale=(
                "material gaps remain and reserved wave budget is available"
                if decision == "launch"
                else (
                    "material gaps remain but the bounded wave budget is exhausted"
                    if decision == "defer"
                    else "declared coverage targets completed within policy"
                )
            ),
        )


def correlate_and_persist(
    repository: SqliteAnalysisRepository,
    run_id: str,
) -> tuple[CanonicalFinding, ...]:
    findings = FindingCorrelator().correlate(
        run_id, repository.accepted_candidates(run_id)
    )
    repository.replace_canonical_findings(run_id, findings)
    return findings


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()
