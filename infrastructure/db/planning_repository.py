"""Durable, immutable records for V6-style planning and generated prompts."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from domain.contracts import ArchitectureDiscovery, GeneratedRolePrompt, RoleProposal

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class SqlitePlanningRepository:
    """Keep planning facts immutable and separate from source-bearing requests."""

    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger

    def record_discovery(self, discovery: ArchitectureDiscovery) -> None:
        values = (
            discovery.discovery_id,
            discovery.run_id,
            discovery.snapshot_id,
            discovery.input_hash,
            discovery.prompt_template,
            discovery.prompt_version,
            discovery.prompt_text,
            _json(discovery.result),
            discovery.result_hash,
            discovery.status,
            _now(),
        )

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR IGNORE INTO architecture_discoveries(
                    discovery_id, run_id, snapshot_id, input_hash,
                    prompt_template, prompt_version, prompt_text, result_json,
                    result_hash, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """SELECT run_id, snapshot_id, input_hash, prompt_template,
                          prompt_version, prompt_text, result_json, result_hash, status
                   FROM architecture_discoveries WHERE discovery_id = ?""",
                (discovery.discovery_id,),
            ).fetchone()
            if row is None or tuple(row) != values[1:-1]:
                raise RuntimeError("immutable architecture discovery conflict")

        self.ledger.write_transaction(operation)

    def record_role_proposal(self, proposal: RoleProposal) -> None:
        payload = {
            "name": proposal.name,
            "mission": proposal.mission,
            "rationale": proposal.rationale,
            "coverage_targets": proposal.coverage_targets,
            "required_capabilities": proposal.required_capabilities,
        }
        values = (
            proposal.proposal_id,
            proposal.run_id,
            proposal.wave_number,
            proposal.proposal_hash,
            _json(payload),
            proposal.validation_status,
            proposal.validation_reason,
            proposal.approved_role_id,
            _now(),
        )

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR IGNORE INTO role_proposals(
                    proposal_id, run_id, wave_number, proposal_hash, proposal_json,
                    validation_status, validation_reason, approved_role_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """SELECT run_id, wave_number, proposal_hash, proposal_json,
                          validation_status, validation_reason, approved_role_id
                   FROM role_proposals WHERE proposal_id = ?""",
                (proposal.proposal_id,),
            ).fetchone()
            if row is None or tuple(row) != values[1:-1]:
                raise RuntimeError("immutable role proposal conflict")

        self.ledger.write_transaction(operation)

    def record_generated_prompt(self, prompt: GeneratedRolePrompt) -> None:
        values = (
            prompt.prompt_id,
            prompt.run_id,
            prompt.wave_number,
            prompt.role_id,
            prompt.prompt_template,
            prompt.prompt_version,
            prompt.instruction_text,
            prompt.architecture_hash,
            prompt.goal_hash,
            prompt.content_hash,
            prompt.validation_status,
            _json(prompt.validation),
            _now(),
        )

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR IGNORE INTO generated_role_prompts(
                    prompt_id, run_id, wave_number, role_id, prompt_template,
                    prompt_version, instruction_text, architecture_hash, goal_hash,
                    content_hash, validation_status, validation_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """SELECT run_id, wave_number, role_id, prompt_template,
                          prompt_version, instruction_text, architecture_hash, goal_hash,
                          content_hash, validation_status, validation_json
                   FROM generated_role_prompts WHERE prompt_id = ?""",
                (prompt.prompt_id,),
            ).fetchone()
            if row is None or tuple(row) != values[1:-1]:
                raise RuntimeError("immutable generated role prompt conflict")

        self.ledger.write_transaction(operation)

    def discoveries_for_run(self, run_id: str) -> list[dict[str, object]]:
        with database_connection(self.ledger.database_path) as connection:
            rows = connection.execute(
                """SELECT discovery_id, snapshot_id, input_hash, prompt_template,
                          prompt_version, prompt_text, result_json, result_hash, status,
                          created_at
                   FROM architecture_discoveries WHERE run_id = ? ORDER BY created_at""",
                (run_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "result": json.loads(str(row["result_json"])),
            }
            for row in rows
        ]
