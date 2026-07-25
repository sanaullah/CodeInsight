"""Bounded read projections for semantic repository architecture."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .database import database_connection
from .run_ledger import SqliteRunLedger


def _record(row) -> dict[str, Any]:
    return {
        **{key: row[key] for key in row.keys() if key != "metadata_json"},
        "metadata": json.loads(row["metadata_json"]) if "metadata_json" in row.keys() else {},
    }


class SqliteSemanticArchitectureQuery:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.database_path = Path(ledger.database_path)

    def summary(self, snapshot_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            snapshot = connection.execute(
                """
                SELECT snapshots.snapshot_id, snapshots.git_ref,
                       snapshots.head_commit, snapshots.parent_snapshot_id,
                       snapshots.created_at, snapshots.dirty,
                       projects.display_name,
                       state.extractor_version, state.status
                FROM repository_snapshots AS snapshots
                JOIN projects USING(project_id)
                LEFT JOIN semantic_projection_state AS state USING(snapshot_id)
                WHERE snapshots.snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            if snapshot is None:
                return None
            component_counts = {
                str(row["component_kind"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT component_kind, COUNT(*) AS count
                    FROM semantic_components WHERE snapshot_id = ?
                    GROUP BY component_kind
                    """,
                    (snapshot_id,),
                )
            }
            boundary_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM semantic_boundaries WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()[0]
            )
            totals = connection.execute(
                """
                SELECT COUNT(*) AS files, COALESCE(SUM(line_count), 0) AS lines
                FROM files WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            languages = {
                str(row["language"] or "unknown"): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT language, COUNT(*) AS count FROM files
                    WHERE snapshot_id = ? GROUP BY language
                    """,
                    (snapshot_id,),
                )
            }
        return {
            "snapshot_id": snapshot_id,
            "display_name": str(snapshot["display_name"]),
            "git": {
                "ref": snapshot["git_ref"],
                "head_commit": snapshot["head_commit"],
                "dirty": bool(snapshot["dirty"]),
                "parent_snapshot_id": snapshot["parent_snapshot_id"],
            },
            "created_at": str(snapshot["created_at"]),
            "counts": {
                "services": component_counts.get("service", 0),
                "datastores": component_counts.get("datastore", 0),
                "external_systems": component_counts.get("external_system", 0),
                "queues": component_counts.get("queue", 0),
                "libraries": component_counts.get("library", 0),
                "unknown": component_counts.get("unknown", 0),
                "boundaries": boundary_count,
            },
            "totals": {
                "files": int(totals["files"]),
                "lines": int(totals["lines"]),
                "languages": languages,
            },
            "completeness": {
                "status": str(snapshot["status"] or "unknown"),
                "extractor_version": snapshot["extractor_version"],
            },
        }

    def graph(
        self,
        snapshot_id: str,
        *,
        component_kinds: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
        focus: str | None = None,
        depth: int = 1,
        limit: int = 250,
    ) -> dict[str, Any] | None:
        if depth < 0 or limit < 1:
            raise ValueError("depth and limit must be bounded")
        summary = self.summary(snapshot_id)
        if summary is None:
            return None
        with database_connection(self.database_path) as connection:
            selected = self._selected_component_ids(
                connection, snapshot_id, focus=focus, depth=depth, limit=limit + 1
            )
            truncated = len(selected) > limit
            selected = selected[:limit]
            if not selected:
                return {
                    **summary,
                    "components": [],
                    "boundaries": [],
                    "relations": [],
                    "findings": [],
                    "truncated": False,
                    "limits": {"depth": depth, "node_limit": limit},
                }
            placeholders = ",".join("?" for _ in selected)
            kind_clause = ""
            values: list[Any] = [snapshot_id, *selected]
            if component_kinds:
                kind_clause = (
                    f" AND component_kind IN ({','.join('?' for _ in component_kinds)})"
                )
                values.extend(component_kinds)
            components = [
                _record(row)
                for row in connection.execute(
                    f"""
                    SELECT * FROM semantic_components
                    WHERE snapshot_id = ? AND component_id IN ({placeholders})
                    {kind_clause}
                    ORDER BY stable_key
                    """,
                    tuple(values),
                )
            ]
            visible = [str(item["component_id"]) for item in components]
            if not visible:
                relations: list[dict[str, Any]] = []
                boundaries: list[dict[str, Any]] = []
            else:
                visible_placeholders = ",".join("?" for _ in visible)
                relation_clause = ""
                relation_values: list[Any] = [snapshot_id, *visible, *visible]
                if relation_kinds:
                    relation_clause = (
                        f" AND relation_kind IN "
                        f"({','.join('?' for _ in relation_kinds)})"
                    )
                    relation_values.extend(relation_kinds)
                relations = [
                    _record(row)
                    for row in connection.execute(
                        f"""
                        SELECT * FROM semantic_relations
                        WHERE snapshot_id = ?
                          AND source_component_id IN ({visible_placeholders})
                          AND target_component_id IN ({visible_placeholders})
                          {relation_clause}
                        ORDER BY stable_key
                        """,
                        tuple(relation_values),
                    )
                ]
                boundaries = self._boundaries(connection, snapshot_id, visible)
            findings = self._findings(connection, snapshot_id, visible)
        return {
            **summary,
            "components": components,
            "boundaries": boundaries,
            "relations": relations,
            "findings": findings,
            "truncated": truncated,
            "limits": {"depth": depth, "node_limit": limit},
        }

    def component(self, snapshot_id: str, component_id: str) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM semantic_components
                WHERE snapshot_id = ? AND component_id = ?
                """,
                (snapshot_id, component_id),
            ).fetchone()
            if row is None:
                return None
            memberships = [
                {
                    **_record(item),
                    "owners": json.loads(item["file_metadata"]).get("owners", []),
                }
                for item in connection.execute(
                    """
                    SELECT memberships.*, files.relative_path, files.language,
                           files.line_count, files.metadata_json AS file_metadata
                    FROM semantic_memberships AS memberships
                    LEFT JOIN files USING(file_id)
                    WHERE memberships.snapshot_id = ?
                      AND memberships.component_id = ?
                    ORDER BY files.relative_path
                    """,
                    (snapshot_id, component_id),
                )
            ]
            resources = self._entity_rows(
                connection, "semantic_resources", snapshot_id, component_id
            )
            endpoints = self._entity_rows(
                connection, "semantic_endpoints", snapshot_id, component_id
            )
            provenance = [
                {
                    **_record(item),
                    "relative_path": item["relative_path"],
                }
                for item in connection.execute(
                    """
                    SELECT provenance.*, files.relative_path
                    FROM semantic_provenance AS provenance
                    JOIN files USING(file_id)
                    WHERE provenance.snapshot_id = ?
                      AND provenance.entity_id = ?
                    ORDER BY files.relative_path, start_line
                    """,
                    (snapshot_id, component_id),
                )
            ]
            findings = self._findings(connection, snapshot_id, [component_id])
        return {
            **_record(row),
            "memberships": memberships,
            "resources": resources,
            "endpoints": endpoints,
            "provenance": provenance,
            "findings": findings,
        }

    def trace(
        self,
        snapshot_id: str,
        *,
        source_id: str,
        target_id: str,
        max_hops: int = 8,
        max_edges: int = 10_000,
    ) -> dict[str, Any] | None:
        summary = self.summary(snapshot_id)
        if summary is None:
            return None
        if summary["completeness"]["status"] in {"unknown", "unsupported"}:
            return {
                "snapshot_id": snapshot_id,
                "status": "unsupported",
                "components": [],
                "relations": [],
                "provenance": [],
                "max_hops": max_hops,
            }
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM semantic_relations WHERE snapshot_id = ?
                ORDER BY stable_key LIMIT ?
                """,
                (snapshot_id, max_edges + 1),
            ).fetchall()
            capped = len(rows) > max_edges
            relations = rows[:max_edges]
            adjacency: dict[str, list[Any]] = {}
            for row in relations:
                adjacency.setdefault(str(row["source_component_id"]), []).append(row)
            queue = deque([(source_id, [], {source_id})])
            path: list[Any] | None = None
            while queue:
                current, current_path, visited = queue.popleft()
                if current == target_id:
                    path = current_path
                    break
                if len(current_path) >= max_hops:
                    continue
                for relation in adjacency.get(current, []):
                    target = str(relation["target_component_id"])
                    if target not in visited:
                        queue.append(
                            (target, [*current_path, relation], {*visited, target})
                        )
            if path is None:
                return {
                    "snapshot_id": snapshot_id,
                    "status": "truncated" if capped else "no_path",
                    "components": [],
                    "relations": [],
                    "provenance": [],
                    "max_hops": max_hops,
                }
            component_ids = [
                source_id, *[str(item["target_component_id"]) for item in path]
            ]
            placeholders = ",".join("?" for _ in component_ids)
            components = [
                _record(row)
                for row in connection.execute(
                    f"""
                    SELECT * FROM semantic_components
                    WHERE snapshot_id = ? AND component_id IN ({placeholders})
                    """,
                    (snapshot_id, *component_ids),
                )
            ]
            relation_ids = [str(item["relation_id"]) for item in path]
            provenance = []
            if relation_ids:
                relation_placeholders = ",".join("?" for _ in relation_ids)
                provenance = [
                    {
                        **_record(row),
                        "relative_path": row["relative_path"],
                    }
                    for row in connection.execute(
                        f"""
                        SELECT provenance.*, files.relative_path
                        FROM semantic_provenance AS provenance
                        JOIN files USING(file_id)
                        WHERE provenance.snapshot_id = ?
                          AND provenance.entity_kind = 'relation'
                          AND provenance.entity_id IN ({relation_placeholders})
                        ORDER BY files.relative_path, start_line
                        """,
                        (snapshot_id, *relation_ids),
                    )
                ]
        component_by_id = {str(item["component_id"]): item for item in components}
        return {
            "snapshot_id": snapshot_id,
            "status": "complete",
            "components": [component_by_id[item] for item in component_ids],
            "relations": [_record(item) for item in path],
            "provenance": provenance,
            "max_hops": max_hops,
        }

    @staticmethod
    def _selected_component_ids(
        connection,
        snapshot_id: str,
        *,
        focus: str | None,
        depth: int,
        limit: int,
    ) -> list[str]:
        if not focus:
            return [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT component_id FROM semantic_components
                    WHERE snapshot_id = ? ORDER BY stable_key LIMIT ?
                    """,
                    (snapshot_id, limit),
                )
            ]
        seed = connection.execute(
            """
            SELECT component_id FROM semantic_components
            WHERE snapshot_id = ?
              AND (component_id = ? OR stable_key = ? OR name = ?)
            ORDER BY stable_key LIMIT 1
            """,
            (snapshot_id, focus, focus, focus),
        ).fetchone()
        if seed is None:
            return []
        selected = [str(seed[0])]
        frontier = selected[:]
        for _ in range(depth):
            if not frontier or len(selected) >= limit:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = connection.execute(
                f"""
                SELECT source_component_id, target_component_id
                FROM semantic_relations
                WHERE snapshot_id = ? AND (
                    source_component_id IN ({placeholders})
                    OR target_component_id IN ({placeholders})
                )
                ORDER BY stable_key LIMIT ?
                """,
                (snapshot_id, *frontier, *frontier, limit * 2),
            )
            next_ids = sorted(
                {
                    str(value)
                    for row in rows
                    for value in row
                    if str(value) not in selected
                }
            )
            frontier = next_ids[: max(0, limit - len(selected))]
            selected.extend(frontier)
        return selected

    @staticmethod
    def _boundaries(connection, snapshot_id: str, component_ids: list[str]):
        placeholders = ",".join("?" for _ in component_ids)
        rows = connection.execute(
            f"""
            SELECT boundaries.*, memberships.component_id
            FROM semantic_boundaries AS boundaries
            JOIN semantic_boundary_memberships AS memberships USING(boundary_id)
            WHERE boundaries.snapshot_id = ?
              AND memberships.component_id IN ({placeholders})
            ORDER BY boundaries.stable_key, memberships.component_id
            """,
            (snapshot_id, *component_ids),
        )
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = grouped.setdefault(
                str(row["boundary_id"]), {**_record(row), "component_ids": []}
            )
            item["component_ids"].append(str(row["component_id"]))
        return list(grouped.values())

    @staticmethod
    def _findings(connection, snapshot_id: str, component_ids: list[str]):
        if not component_ids:
            return []
        placeholders = ",".join("?" for _ in component_ids)
        rows = connection.execute(
            f"""
            SELECT DISTINCT findings.finding_id,
                   json_extract(findings.contract_json, '$.title') AS title,
                   json_extract(findings.contract_json, '$.severity') AS severity,
                   json_extract(findings.contract_json, '$.confidence') AS confidence,
                   memberships.component_id
            FROM canonical_findings AS findings
            JOIN runs USING(run_id)
            JOIN canonical_finding_members AS finding_members USING(finding_id)
            JOIN finding_candidates AS candidates USING(candidate_id)
            JOIN semantic_memberships AS memberships
              ON memberships.snapshot_id = runs.snapshot_id
            JOIN files ON files.file_id = memberships.file_id
            WHERE runs.snapshot_id = ?
              AND memberships.component_id IN ({placeholders})
              AND json_extract(candidates.contract_json, '$.affected_path')
                    = files.relative_path
            ORDER BY findings.finding_id
            """,
            (snapshot_id, *component_ids),
        )
        return [dict(row) for row in rows]

    @staticmethod
    def _entity_rows(connection, table: str, snapshot_id: str, component_id: str):
        return [
            _record(row)
            for row in connection.execute(
                f"""
                SELECT * FROM {table}
                WHERE snapshot_id = ? AND component_id = ?
                ORDER BY stable_key
                """,
                (snapshot_id, component_id),
            )
        ]
