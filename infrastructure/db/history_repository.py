"""Cross-run history, comparison, and trend projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import database_connection
from .run_ledger import SqliteRunLedger


class SqliteHistoryRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.database_path = Path(ledger.database_path)

    def query(
        self,
        *,
        search: str | None = None,
        status: str | None = None,
        mode: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        clauses = ["1 = 1"]
        values: list[Any] = []
        if search:
            clauses.append(
                "(LOWER(projects.display_name) LIKE ? OR LOWER(projects.canonical_path) LIKE ?)"
            )
            pattern = f"%{search.lower()}%"
            values.extend((pattern, pattern))
        if status:
            clauses.append("runs.status = ?")
            values.append(status)
        if mode:
            clauses.append("runs.mode = ?")
            values.append(mode)
        if cursor:
            clauses.append("runs.run_id < ?")
            values.append(cursor)
        where = " AND ".join(clauses)
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                WITH task_stats AS (
                    SELECT run_id, COUNT(*) AS task_count FROM tasks GROUP BY run_id
                ), role_stats AS (
                    SELECT run_id, COUNT(*) AS specialist_count
                    FROM role_specs GROUP BY run_id
                ), finding_stats AS (
                    SELECT run_id, COUNT(*) AS finding_count
                    FROM canonical_findings GROUP BY run_id
                ), call_stats AS (
                    SELECT run_id,
                           COALESCE(SUM(CAST(json_extract(usage_json,
                               '$.input_tokens') AS INTEGER)), 0) AS input_tokens,
                           COALESCE(SUM(CAST(json_extract(usage_json,
                               '$.output_tokens') AS INTEGER)), 0) AS output_tokens,
                           COALESCE(SUM(CAST(json_extract(usage_json,
                               '$.cost_usd') AS REAL)), 0) AS cost_usd
                    FROM model_calls GROUP BY run_id
                )
                SELECT runs.run_id, runs.status, runs.mode, runs.current_stage,
                       runs.created_at, runs.started_at, runs.completed_at,
                       runs.snapshot_id, projects.display_name,
                       snapshots.base_commit, snapshots.head_commit,
                       snapshots.dirty,
                       COALESCE(task_stats.task_count, 0) AS task_count,
                       COALESCE(role_stats.specialist_count, 0) AS specialist_count,
                       COALESCE(finding_stats.finding_count, 0) AS finding_count,
                       COALESCE(call_stats.input_tokens, 0) AS input_tokens,
                       COALESCE(call_stats.output_tokens, 0) AS output_tokens,
                       COALESCE(call_stats.cost_usd, 0) AS cost_usd
                FROM runs
                LEFT JOIN repository_snapshots AS snapshots USING(snapshot_id)
                LEFT JOIN projects USING(project_id)
                LEFT JOIN task_stats USING(run_id)
                LEFT JOIN role_stats USING(run_id)
                LEFT JOIN finding_stats USING(run_id)
                LEFT JOIN call_stats USING(run_id)
                WHERE {where}
                ORDER BY runs.run_id DESC
                LIMIT ?
                """,
                (*values, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        return {
            "items": [self._run(row) for row in page],
            "next_cursor": str(page[-1]["run_id"]) if has_more else None,
        }

    def compare(self, baseline_run_id: str, target_run_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            runs = connection.execute(
                "SELECT run_id FROM runs WHERE run_id IN (?, ?)",
                (baseline_run_id, target_run_id),
            ).fetchall()
            if len(runs) != 2:
                return None
            rows = connection.execute(
                """
                SELECT run_id, fingerprint, contract_json
                FROM canonical_findings
                WHERE run_id IN (?, ?)
                ORDER BY fingerprint
                """,
                (baseline_run_id, target_run_id),
            ).fetchall()
            reopened = {
                str(row["finding_id"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT reviews.finding_id
                    FROM finding_review_events AS reviews
                    JOIN canonical_findings AS findings USING(finding_id)
                    WHERE findings.run_id = ? AND reviews.review_state = 'reopened'
                    """,
                    (target_run_id,),
                ).fetchall()
            }
        baseline = {
            str(row["fingerprint"]): json.loads(row["contract_json"])
            for row in rows
            if row["run_id"] == baseline_run_id
        }
        target = {
            str(row["fingerprint"]): json.loads(row["contract_json"])
            for row in rows
            if row["run_id"] == target_run_id
        }
        new = [target[key] for key in sorted(target.keys() - baseline.keys())]
        resolved = [baseline[key] for key in sorted(baseline.keys() - target.keys())]
        unchanged = []
        severity_moved = []
        for key in sorted(baseline.keys() & target.keys()):
            before = baseline[key]
            after = target[key]
            if before["severity"] == after["severity"]:
                unchanged.append(after)
            else:
                severity_moved.append(
                    {
                        "fingerprint": key,
                        "title": after["title"],
                        "from_severity": before["severity"],
                        "to_severity": after["severity"],
                    }
                )
        return {
            "baseline_run_id": baseline_run_id,
            "target_run_id": target_run_id,
            "new": new,
            "resolved": resolved,
            "unchanged": unchanged,
            "reopened": [item for item in target.values() if item["finding_id"] in reopened],
            "severity_moved": severity_moved,
        }

    def trends(self, days: int = 30) -> dict[str, Any]:
        cutoff = datetime.now(UTC).timestamp() - days * 86_400
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                WITH finding_stats AS (
                    SELECT run_id, COUNT(*) AS finding_count
                    FROM canonical_findings GROUP BY run_id
                ), call_stats AS (
                    SELECT run_id,
                           COALESCE(SUM(CAST(json_extract(usage_json,
                               '$.cost_usd') AS REAL)), 0) AS cost_usd
                    FROM model_calls GROUP BY run_id
                )
                SELECT runs.run_id, runs.created_at, runs.started_at,
                       runs.completed_at, runs.status,
                       COALESCE(finding_stats.finding_count, 0) AS finding_count,
                       COALESCE(call_stats.cost_usd, 0) AS cost_usd
                FROM runs
                LEFT JOIN finding_stats USING(run_id)
                LEFT JOIN call_stats USING(run_id)
                ORDER BY runs.created_at
                """
            ).fetchall()
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            created = datetime.fromisoformat(str(row["created_at"]))
            if created.timestamp() < cutoff:
                continue
            bucket = created.date().isoformat()
            record = buckets.setdefault(
                bucket,
                {
                    "date": bucket,
                    "review_count": 0,
                    "finding_count": 0,
                    "cost_usd": 0.0,
                    "duration_seconds": 0.0,
                    "completed_count": 0,
                },
            )
            record["review_count"] += 1
            record["finding_count"] += int(row["finding_count"])
            record["cost_usd"] += float(row["cost_usd"])
            if row["started_at"] and row["completed_at"]:
                started = datetime.fromisoformat(str(row["started_at"]))
                completed = datetime.fromisoformat(str(row["completed_at"]))
                record["duration_seconds"] += max(0.0, (completed - started).total_seconds())
                record["completed_count"] += 1
        for record in buckets.values():
            completed = record.pop("completed_count")
            record["average_duration_seconds"] = (
                record.pop("duration_seconds") / completed if completed else None
            )
        return {"days": days, "buckets": list(buckets.values()), "partial": not bool(buckets)}

    @staticmethod
    def _run(row) -> dict[str, Any]:
        duration = None
        if row["started_at"] and row["completed_at"]:
            duration = max(
                0.0,
                (
                    datetime.fromisoformat(str(row["completed_at"]))
                    - datetime.fromisoformat(str(row["started_at"]))
                ).total_seconds(),
            )
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        return {
            **dict(row),
            "dirty": bool(row["dirty"]) if row["dirty"] is not None else None,
            "duration_seconds": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": float(row["cost_usd"]),
        }
