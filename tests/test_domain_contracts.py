from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from domain.contracts import (
    AnalysisTask,
    EvidenceRef,
    FindingCandidate,
    RunBudget,
)


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunBudget(untrusted_override=True)


def test_contracts_are_immutable_and_versioned() -> None:
    budget = RunBudget()

    assert budget.schema_version == 1
    with pytest.raises(ValidationError, match="frozen"):
        budget.max_waves = 99


def test_analysis_task_rejects_invalid_attempt_budget() -> None:
    with pytest.raises(ValidationError):
        AnalysisTask(
            task_id="task-1",
            run_id="run-1",
            wave_id="wave-1",
            role_id="role-1",
            task_type="specialist",
            immutable_input_ids=("snapshot-1",),
            configuration_hash="config-hash",
            idempotency_key="stable-key",
            model_policy="default",
            token_budget=1_000,
            cost_budget_usd=1,
            time_budget_seconds=60,
            tool_call_budget=5,
            max_attempts=0,
        )


def test_finding_candidate_requires_calibrated_confidence_range() -> None:
    with pytest.raises(ValidationError):
        FindingCandidate(
            candidate_id="candidate-1",
            producer_task_id="task-1",
            producer_role_id="role-1",
            category="correctness",
            concept_id="state-race",
            title="Concurrent state race",
            claim="Two writers mutate the same state.",
            evidence_ids=("evidence-1",),
            impact="A result can be lost.",
            proposed_severity="high",
            proposed_confidence=1.5,
            recommendation="Serialize the transition.",
            fingerprint_inputs=("state-race", "file.py"),
        )


def test_evidence_round_trips_with_immutable_snapshot_identity() -> None:
    evidence = EvidenceRef(
        evidence_id="evidence-1",
        snapshot_id="snapshot-1",
        file_id="file-1",
        content_hash="a" * 64,
        start_line=10,
        end_line=12,
        excerpt_hash="b" * 64,
        evidence_kind="source",
        provenance={"analyzer": "python-ast", "version": "1"},
        collected_at=datetime.now(UTC),
    )

    restored = EvidenceRef.model_validate_json(evidence.model_dump_json())
    assert restored == evidence
    assert restored.snapshot_id == "snapshot-1"


def test_evidence_rejects_reversed_line_and_byte_spans() -> None:
    common = {
        "evidence_id": "evidence-1",
        "snapshot_id": "snapshot-1",
        "file_id": "file-1",
        "content_hash": "a" * 64,
        "excerpt_hash": "b" * 64,
        "evidence_kind": "source",
        "provenance": {},
        "collected_at": datetime.now(UTC),
    }
    with pytest.raises(ValidationError, match="end_line"):
        EvidenceRef(start_line=5, end_line=4, **common)
    with pytest.raises(ValidationError, match="end_byte"):
        EvidenceRef(
            start_line=1,
            end_line=1,
            start_byte=10,
            end_byte=9,
            **common,
        )
