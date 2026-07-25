"""Durable non-secret settings and review presets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import database_connection
from .run_ledger import SqliteRunLedger

DEFAULT_SETTINGS = {
    "default_mode": "deep",
    "default_max_agents": 4,
    "default_max_waves": 2,
    "default_max_tasks": 100,
    "default_max_total_tokens": 1_000_000,
    "default_max_cost_usd": 25.0,
    "default_max_elapsed_seconds": 3_600,
    "evidence_excerpt_enabled": True,
    "retention_days": 90,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _preset_record(row) -> dict[str, Any]:
    return {
        **{key: row[key] for key in row.keys() if key != "request_json"},
        "request": json.loads(row["request_json"]),
    }


class SqliteSettingsRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.ledger = ledger
        self.database_path = Path(ledger.database_path)

    def get(self) -> dict[str, Any]:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT settings_json, version, updated_at
                FROM application_settings WHERE settings_key = 'local-defaults'
                """
            ).fetchone()
        if row is None:
            return {"settings": dict(DEFAULT_SETTINGS), "version": 0, "updated_at": None}
        return {
            "settings": json.loads(row["settings_json"]),
            "version": int(row["version"]),
            "updated_at": row["updated_at"],
        }

    def update(self, settings: dict[str, Any], expected_version: int) -> dict[str, Any]:
        timestamp = _now()

        def operation(connection):
            current = connection.execute(
                """
                SELECT version FROM application_settings
                WHERE settings_key = 'local-defaults'
                """
            ).fetchone()
            current_version = int(current["version"]) if current else 0
            if current_version != expected_version:
                raise RuntimeError("settings changed; refresh and retry")
            next_version = current_version + 1
            connection.execute(
                """
                INSERT INTO application_settings(
                    settings_key, settings_json, version, updated_at
                ) VALUES ('local-defaults', ?, ?, ?)
                ON CONFLICT(settings_key) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    json.dumps(settings, sort_keys=True, separators=(",", ":")),
                    next_version,
                    timestamp,
                ),
            )

        self.ledger.write_transaction(operation)
        return self.get()

    def list_presets(self) -> list[dict[str, Any]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT preset_id, name, request_json, version, created_at, updated_at
                FROM review_presets ORDER BY LOWER(name), preset_id
                """
            ).fetchall()
        return [_preset_record(row) for row in rows]

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT preset_id, name, request_json, version, created_at, updated_at
                FROM review_presets WHERE preset_id = ?
                """,
                (preset_id,),
            ).fetchone()
        return _preset_record(row) if row else None

    def save_preset(
        self,
        *,
        name: str,
        request: dict[str, Any],
        preset_id: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        timestamp = _now()
        identifier = preset_id or uuid4().hex

        def operation(connection):
            current = connection.execute(
                "SELECT version, created_at FROM review_presets WHERE preset_id = ?",
                (identifier,),
            ).fetchone()
            if preset_id is not None and current is None:
                raise KeyError("preset not found")
            current_version = int(current["version"]) if current else 0
            if expected_version is not None and expected_version != current_version:
                raise RuntimeError("preset changed; refresh and retry")
            connection.execute(
                """
                INSERT INTO review_presets(
                    preset_id, name, request_json, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(preset_id) DO UPDATE SET
                    name = excluded.name,
                    request_json = excluded.request_json,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    identifier,
                    name,
                    json.dumps(request, sort_keys=True, separators=(",", ":")),
                    current_version + 1,
                    current["created_at"] if current else timestamp,
                    timestamp,
                ),
            )

        self.ledger.write_transaction(operation)
        saved = self.get_preset(identifier)
        assert saved is not None
        return saved

    def delete_preset(self, preset_id: str, expected_version: int) -> bool:
        def operation(connection):
            current = connection.execute(
                "SELECT version FROM review_presets WHERE preset_id = ?",
                (preset_id,),
            ).fetchone()
            if current is None:
                return False
            if int(current["version"]) != expected_version:
                raise RuntimeError("preset changed; refresh and retry")
            cursor = connection.execute(
                "DELETE FROM review_presets WHERE preset_id = ? AND version = ?",
                (preset_id, expected_version),
            )
            return cursor.rowcount == 1

        return self.ledger.write_transaction(operation)
