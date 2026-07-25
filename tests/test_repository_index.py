from __future__ import annotations

import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

import pytest

from indexing.repository_index import RepositoryIndexer
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.database import database_connection
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        encoding="utf-8",
        shell=False,
    )
    return result.stdout.strip()


def _project(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".github").mkdir()
    (root / "a.py").write_text(
        "from pkg.b import helper\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "b.py").write_text(
        "def helper():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_a.py").write_text(
        "from a import main\n\ndef test_main():\n    assert main() == 1\n",
        encoding="utf-8",
    )
    (root / "frontend.ts").write_text(
        "import { helper } from './helper';\nhelper();\n", encoding="utf-8"
    )
    (root / "helper.ts").write_text(
        "export function helper() { return 1; }\n", encoding="utf-8"
    )
    (root / ".github" / "CODEOWNERS").write_text(
        "*.py @python-team\nfrontend.ts @frontend-team\n", encoding="utf-8"
    )
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "CodeInsight Tests")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    commit = _git(root, "rev-parse", "HEAD")
    return root, commit


def _indexer(
    tmp_path: Path,
) -> tuple[SqliteRunLedger, SqliteSnapshotRepository, RepositoryIndexer]:
    ledger = SqliteRunLedger(tmp_path / "application.sqlite3")
    repository = SqliteSnapshotRepository(ledger)
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    return ledger, repository, RepositoryIndexer(repository, store)


def test_index_persists_shared_files_symbols_edges_ownership_and_artifacts(
    tmp_path: Path,
) -> None:
    root, commit = _project(tmp_path)
    (root / "a.py").write_text(
        "from pkg.b import helper\n\ndef main():\n    return helper() + 1\n",
        encoding="utf-8",
    )
    ledger, repository, indexer = _indexer(tmp_path)
    try:
        result = indexer.build(root, base_commit=commit, neighborhood_depth=1)

        assert not result.cached
        assert result.file_count == 7
        assert result.symbol_count >= 3
        assert result.edge_count >= 4
        assert result.changed_paths == ("a.py",)
        assert "a.py" in result.target_paths
        assert "pkg/b.py" in result.target_paths
        files = repository.list_files(result.snapshot.snapshot_id)
        by_path = {item["relative_path"]: item for item in files}
        assert by_path["a.py"]["metadata"]["owners"] == ["@python-team"]
        assert by_path["frontend.ts"]["metadata"]["owners"] == ["@frontend-team"]
        assert by_path["tests/test_a.py"]["classification"] == "test"
        assert by_path["a.py"]["support_tier"] == "parsed"
        with database_connection(ledger.database_path) as connection:
            file_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(files)")
            }
            assert "content" not in file_columns
            artifact = connection.execute(
                "SELECT storage_path FROM artifacts WHERE artifact_id = ?",
                (by_path["a.py"]["artifact_id"],),
            ).fetchone()
            assert Path(artifact["storage_path"]).read_text(encoding="utf-8").endswith(
                "return helper() + 1\n"
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) AS count FROM edges WHERE edge_kind = 'calls'"
                ).fetchone()["count"]
                >= 1
            )
    finally:
        ledger.close()


def test_unchanged_snapshot_is_reused_without_duplicate_metadata(tmp_path: Path) -> None:
    root, _commit = _project(tmp_path)
    ledger, repository, indexer = _indexer(tmp_path)
    try:
        first = indexer.build(root)
        second = indexer.build(root)

        assert not first.cached
        assert second.cached
        assert second.snapshot.snapshot_id == first.snapshot.snapshot_id
        assert second.snapshot.included_paths == first.snapshot.included_paths
        with database_connection(ledger.database_path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) AS count FROM repository_snapshots"
                ).fetchone()["count"]
                == 1
            )
    finally:
        ledger.close()


def test_repeated_symbol_calls_persist_as_one_graph_edge(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "module.py").write_text(
        "def helper():\n"
        "    return 1\n\n"
        "def main():\n"
        "    return helper() + helper()\n",
        encoding="utf-8",
    )
    ledger, _repository, indexer = _indexer(tmp_path)
    try:
        result = indexer.build(root)
        with database_connection(ledger.database_path) as connection:
            call_edges = connection.execute(
                """
                SELECT source_id, target_id, COUNT(*) AS count
                FROM edges
                WHERE snapshot_id = ? AND edge_kind = 'calls'
                GROUP BY source_id, target_id
                """,
                (result.snapshot.snapshot_id,),
            ).fetchall()
        assert len(call_edges) == 1
        assert call_edges[0]["count"] == 1
        assert result.edge_count == 1
    finally:
        ledger.close()


def test_non_git_repository_gets_deterministic_content_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    ledger, _repository, indexer = _indexer(tmp_path)
    try:
        first = indexer.build(root)
        second = indexer.build(root)

        assert first.snapshot.git_repository is None
        assert not first.snapshot.dirty
        assert second.cached
        assert second.target_paths == ("module.py",)
    finally:
        ledger.close()


def test_selected_directory_cannot_escape_repository(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    ledger, _repository, indexer = _indexer(tmp_path)
    try:
        with pytest.raises(ValueError, match="escapes repository"):
            indexer.build(root, selected_directories=("..",))
    finally:
        ledger.close()


def test_oversized_files_are_excluded_before_reading(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "small.py").write_text("x = 1\n", encoding="utf-8")
    (root / "large.py").write_bytes(b"x" * 100)
    ledger = SqliteRunLedger(tmp_path / "application.sqlite3")
    repository = SqliteSnapshotRepository(ledger)
    indexer = RepositoryIndexer(
        repository,
        FilesystemArtifactStore(tmp_path / "artifacts"),
        max_file_bytes=20,
    )
    try:
        result = indexer.build(root)
        assert result.snapshot.included_paths == ("small.py",)
    finally:
        ledger.close()


def test_artifact_integrity_failure_is_detected(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    artifact = store.put(b"trusted", artifact_kind="source")
    Path(artifact.storage_path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        store.read(artifact)


def test_snapshot_neighborhood_limits_and_validation(tmp_path: Path) -> None:
    root, commit = _project(tmp_path)
    (root / "a.py").write_text(
        "from pkg.b import helper\n\ndef main():\n    return helper() + 1\n",
        encoding="utf-8",
    )
    ledger, repository, indexer = _indexer(tmp_path)
    try:
        result = indexer.build(root, base_commit=commit)
        files = repository.list_files(result.snapshot.snapshot_id)
        seed = next(item["file_id"] for item in files if item["relative_path"] == "a.py")
        assert repository.neighborhood(
            result.snapshot.snapshot_id, [seed], depth=0
        ) == [seed]
        with pytest.raises(ValueError, match="depth"):
            repository.neighborhood(result.snapshot.snapshot_id, [seed], depth=-1)
        with pytest.raises(ValueError, match="max_files"):
            repository.neighborhood(
                result.snapshot.snapshot_id, [seed], max_files=0
            )
    finally:
        ledger.close()


def test_indexing_performance_and_no_open_database_handles(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    for number in range(200):
        (root / f"module_{number}.py").write_text(
            f"def function_{number}():\n    return {number}\n", encoding="utf-8"
        )
    ledger, _repository, indexer = _indexer(tmp_path)
    try:
        started = time.perf_counter()
        result = indexer.build(root)
        elapsed = time.perf_counter() - started
        assert result.file_count == 200
        assert elapsed < 5
    finally:
        ledger.close()
    with closing(sqlite3.connect(tmp_path / "application.sqlite3")) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
