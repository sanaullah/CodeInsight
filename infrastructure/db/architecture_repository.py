"""Bounded architecture projections over the immutable repository index."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .database import database_connection
from .run_ledger import SqliteRunLedger


class SqliteArchitectureRepository:
    def __init__(self, ledger: SqliteRunLedger) -> None:
        self.database_path = Path(ledger.database_path)

    def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT snapshots.snapshot_id, snapshots.project_id,
                       projects.display_name, snapshots.git_repository,
                       snapshots.base_commit, snapshots.head_commit,
                       snapshots.dirty, snapshots.metadata_json,
                       snapshots.created_at,
                       COUNT(DISTINCT files.file_id) AS file_count,
                       COUNT(DISTINCT symbols.symbol_id) AS symbol_count,
                       COUNT(DISTINCT edges.edge_id) AS edge_count
                FROM repository_snapshots AS snapshots
                JOIN projects USING(project_id)
                LEFT JOIN files USING(snapshot_id)
                LEFT JOIN symbols USING(snapshot_id)
                LEFT JOIN edges USING(snapshot_id)
                GROUP BY snapshots.snapshot_id
                ORDER BY snapshots.created_at DESC, snapshots.snapshot_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys() if key != "metadata_json"},
                "dirty": bool(row["dirty"]),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def graph(
        self,
        snapshot_id: str,
        *,
        focus: str | None = None,
        depth: int = 1,
        limit: int = 250,
        edge_kind: str | None = None,
    ) -> dict[str, Any] | None:
        with database_connection(self.database_path) as connection:
            snapshot = connection.execute(
                """
                SELECT snapshots.snapshot_id, projects.display_name,
                       snapshots.metadata_json
                FROM repository_snapshots AS snapshots
                JOIN projects USING(project_id)
                WHERE snapshots.snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
            if snapshot is None:
                return None
            files = connection.execute(
                """
                SELECT file_id, relative_path, language, classification,
                       support_tier, line_count, metadata_json
                FROM files WHERE snapshot_id = ? ORDER BY relative_path
                """,
                (snapshot_id,),
            ).fetchall()
            symbols = connection.execute(
                """
                SELECT symbol_id, file_id FROM symbols WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()
            raw_edges = connection.execute(
                """
                SELECT edge_id, source_id, target_id, edge_kind, confidence,
                       metadata_json
                FROM edges WHERE snapshot_id = ?
                ORDER BY edge_kind, source_id, target_id
                """,
                (snapshot_id,),
            ).fetchall()
            correlations = connection.execute(
                """
                SELECT DISTINCT findings.finding_id,
                       json_extract(candidates.contract_json, '$.affected_path')
                           AS affected_path,
                       json_extract(findings.contract_json, '$.severity') AS severity
                FROM runs
                JOIN canonical_findings AS findings USING(run_id)
                JOIN canonical_finding_members AS members USING(finding_id)
                JOIN finding_candidates AS candidates USING(candidate_id)
                WHERE runs.snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()
        file_by_id = {str(row["file_id"]): row for row in files}
        symbol_files = {str(row["symbol_id"]): str(row["file_id"]) for row in symbols}
        collapsed: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in raw_edges:
            source = symbol_files.get(str(row["source_id"]), str(row["source_id"]))
            target = symbol_files.get(str(row["target_id"]), str(row["target_id"]))
            kind = str(row["edge_kind"])
            if source == target or source not in file_by_id or target not in file_by_id:
                continue
            if edge_kind and kind != edge_kind:
                continue
            key = (source, target, kind)
            current = collapsed.get(key)
            if current is None or float(row["confidence"]) > current["confidence"]:
                collapsed[key] = {
                    "edge_id": str(row["edge_id"]),
                    "source_id": source,
                    "target_id": target,
                    "edge_kind": kind,
                    "confidence": float(row["confidence"]),
                    "derivation": "repository-index",
                }
        selected = set(file_by_id)
        if focus:
            seed = (
                focus
                if focus in file_by_id
                else next(
                    (
                        file_id
                        for file_id, row in file_by_id.items()
                        if row["relative_path"] == focus
                    ),
                    None,
                )
            )
            if seed is None:
                selected = set()
            else:
                selected = {seed}
                frontier = {seed}
                for _ in range(depth):
                    neighbors = {
                        endpoint
                        for edge in collapsed.values()
                        if edge["source_id"] in frontier or edge["target_id"] in frontier
                        for endpoint in (edge["source_id"], edge["target_id"])
                    }
                    frontier = neighbors - selected
                    selected.update(frontier)
        selected = set(sorted(selected, key=lambda item: file_by_id[item]["relative_path"])[:limit])
        finding_by_path: dict[str, list[dict[str, str]]] = {}
        for row in correlations:
            if row["affected_path"]:
                finding_by_path.setdefault(str(row["affected_path"]), []).append(
                    {
                        "finding_id": str(row["finding_id"]),
                        "severity": str(row["severity"]),
                    }
                )
        nodes = [
            {
                "node_id": file_id,
                "label": str(file_by_id[file_id]["relative_path"]),
                "node_kind": "file",
                "language": file_by_id[file_id]["language"],
                "classification": file_by_id[file_id]["classification"],
                "support_tier": file_by_id[file_id]["support_tier"],
                "line_count": int(file_by_id[file_id]["line_count"]),
                "confidence": 1.0,
                "derivation": "repository-index",
                "findings": finding_by_path.get(str(file_by_id[file_id]["relative_path"]), []),
            }
            for file_id in sorted(selected, key=lambda item: file_by_id[item]["relative_path"])
        ]
        edges = [
            edge
            for edge in collapsed.values()
            if edge["source_id"] in selected and edge["target_id"] in selected
        ]
        metadata = json.loads(snapshot["metadata_json"])
        return {
            "snapshot_id": snapshot_id,
            "display_name": snapshot["display_name"],
            "summary": {
                "file_count": len(files),
                "language_counts": self._counts(files, "language"),
                "classification_counts": self._counts(files, "classification"),
                "changed_paths": metadata.get("changed_paths", []),
            },
            "nodes": nodes,
            "edges": edges,
            "truncated": len(selected) < len(files),
            "limits": {"depth": depth, "node_limit": limit},
        }

    def trace(
        self,
        snapshot_id: str,
        *,
        source_id: str,
        target_id: str,
        max_hops: int = 8,
    ) -> dict[str, Any] | None:
        graph = self.graph(snapshot_id, limit=2_000)
        if graph is None:
            return None
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for edge in graph["edges"]:
            adjacency.setdefault(edge["source_id"], []).append(edge)
        queue = deque([(source_id, [], {source_id})])
        while queue:
            current, path, visited = queue.popleft()
            if current == target_id:
                node_map = {node["node_id"]: node for node in graph["nodes"]}
                node_ids = [source_id, *[edge["target_id"] for edge in path]]
                return {
                    "snapshot_id": snapshot_id,
                    "found": True,
                    "nodes": [node_map[item] for item in node_ids],
                    "edges": path,
                    "max_hops": max_hops,
                }
            if len(path) >= max_hops:
                continue
            for edge in adjacency.get(current, []):
                target = edge["target_id"]
                if target not in visited:
                    queue.append((target, [*path, edge], {*visited, target}))
        return {
            "snapshot_id": snapshot_id,
            "found": False,
            "nodes": [],
            "edges": [],
            "max_hops": max_hops,
        }

    @staticmethod
    def _counts(rows, field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row[field] or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))
