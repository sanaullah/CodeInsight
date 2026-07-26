from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from domain.contracts import (
    ArchitectureDiscovery,
    GeneratedRolePrompt,
    RoleProposal,
)
from infrastructure.db.database import database_connection
from infrastructure.db.planning_repository import SqlitePlanningRepository
from infrastructure.db.run_ledger import SqliteRunLedger


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _repository(tmp_path: Path) -> tuple[SqliteRunLedger, SqlitePlanningRepository]:
    ledger = SqliteRunLedger(tmp_path / "planning.db")
    ledger.create_run(
        run_id="run-1",
        submission_key="submission-1",
        request={"project_path": "sample"},
        mode="quick",
    )

    def seed(connection) -> None:
        connection.execute(
            """INSERT INTO projects(
                   project_id, canonical_path, display_name, created_at, updated_at
               )
               VALUES ('project-1', 'sample', 'sample', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO repository_snapshots(
                   snapshot_id, project_id, identity_hash, configuration_hash,
                   scanner_version, created_at
               ) VALUES ('snapshot-1', 'project-1', 'identity', 'config', 'test', 'now')"""
        )

    ledger.write_transaction(seed)
    return ledger, SqlitePlanningRepository(ledger)


def test_planning_repository_persists_immutable_redacted_records(tmp_path: Path) -> None:
    ledger, repository = _repository(tmp_path)
    try:
        discovery = ArchitectureDiscovery(
            discovery_id="discovery-1",
            run_id="run-1",
            snapshot_id="snapshot-1",
            input_hash=_hash("input"),
            prompt_template="architecture-discovery",
            prompt_version=1,
            prompt_text="Return only a typed architecture summary.",
            result={"modules": ["api"], "source_content_included": False},
            result_hash=_hash("result"),
            status="deterministic",
        )
        repository.record_discovery(discovery)
        repository.record_discovery(discovery)
        assert repository.discoveries_for_run("run-1")[0]["result"] == {
            "modules": ["api"],
            "source_content_included": False,
        }

        repository.record_role_proposal(
            RoleProposal(
                proposal_id="proposal-1",
                run_id="run-1",
                wave_number=1,
                proposal_hash=_hash("proposal"),
                name="API Contract Specialist",
                mission="Review API boundaries.",
                rationale="The snapshot exposes routes.",
                coverage_targets=("api",),
                required_capabilities=("architecture",),
                validation_status="proposed",
            )
        )
        repository.record_generated_prompt(
            GeneratedRolePrompt(
                prompt_id="role-prompt-1",
                run_id="run-1",
                wave_number=1,
                prompt_template="generated-role-instruction",
                prompt_version=1,
                instruction_text="Inspect API contracts and return typed evidence only.",
                architecture_hash=_hash("architecture"),
                goal_hash=_hash("goal"),
                content_hash=_hash("instruction"),
                validation_status="approved",
                validation={"source_content_included": False},
            )
        )
        with database_connection(ledger.database_path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM role_proposals"
            ).fetchone()[0] == 1
            assert connection.execute(
                "SELECT COUNT(*) FROM generated_role_prompts"
            ).fetchone()[0] == 1
            with pytest.raises(Exception, match="immutable"):
                connection.execute(
                    "UPDATE generated_role_prompts SET instruction_text = 'tampered'"
                )
    finally:
        ledger.close()


def test_planning_repository_rejects_conflicting_immutable_discovery(tmp_path: Path) -> None:
    ledger, repository = _repository(tmp_path)
    try:
        base = dict(
            discovery_id="discovery-1",
            run_id="run-1",
            snapshot_id="snapshot-1",
            input_hash=_hash("input"),
            prompt_template="architecture-discovery",
            prompt_version=1,
            prompt_text="Return only architecture.",
            result={"modules": []},
            result_hash=_hash("result"),
            status="deterministic",
        )
        repository.record_discovery(ArchitectureDiscovery(**base))
        with pytest.raises(RuntimeError, match="immutable architecture discovery conflict"):
            repository.record_discovery(
                ArchitectureDiscovery(**(base | {"prompt_text": "different"}))
            )
    finally:
        ledger.close()
