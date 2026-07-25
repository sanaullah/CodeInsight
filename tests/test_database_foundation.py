from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from infrastructure.db.database import (
    SCHEMA_VERSION,
    checkpoint_database,
    database_connection,
    initialize_database,
    open_database,
)
from infrastructure.db.run_ledger import SqliteRunLedger


def _request(project_path: str = ".") -> dict[str, object]:
    return {"project_path": project_path, "max_agents": 4}


def test_database_initialization_is_idempotent_and_enables_wal(tmp_path: Path) -> None:
    database_path = tmp_path / "state" / "codeinsight.db"

    assert initialize_database(database_path) == SCHEMA_VERSION
    assert initialize_database(database_path) == SCHEMA_VERSION

    with database_connection(database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
        migrations = connection.execute(
            "SELECT version, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row["version"] for row in migrations] == [SCHEMA_VERSION]
        assert len(migrations[0]["checksum"]) == 64

    assert list(tmp_path.rglob("*.db")) == [database_path]


def test_database_rejects_schema_newer_than_application(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO schema_migrations(version, description, checksum, applied_at)
            VALUES (?, 'future', 'future', 'future')
            """,
            (SCHEMA_VERSION + 1,),
        )

    try:
        initialize_database(database_path)
    except RuntimeError as exc:
        assert "newer than supported" in str(exc)
    else:
        raise AssertionError("newer schemas must not be opened")


@pytest.mark.parametrize("column", ["description", "checksum"])
def test_database_rejects_tampered_applied_migration(
    tmp_path: Path, column: str
) -> None:
    database_path = tmp_path / "codeinsight.db"
    initialize_database(database_path)
    with database_connection(database_path) as connection:
        connection.execute(
            f"UPDATE schema_migrations SET {column} = 'tampered' WHERE version = ?",
            (SCHEMA_VERSION,),
        )

    with pytest.raises(RuntimeError, match="canonical schema source"):
        initialize_database(database_path)


def test_database_failed_write_rolls_back_and_ledger_close_is_idempotent(
    tmp_path: Path,
) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")

    def invalid_write(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO projects(project_id, canonical_path, display_name, created_at, updated_at)
            VALUES ('project-1', '/first', 'first', 'now', 'now')
            """
        )
        raise RuntimeError("abort transaction")

    with pytest.raises(RuntimeError, match="abort transaction"):
        ledger.write_transaction(invalid_write)
    with database_connection(ledger.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    ledger.close()
    ledger.close()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.create_run(
            run_id="closed",
            submission_key="closed",
            request=_request(),
        )


def test_checkpoint_supports_passive_and_truncate_modes(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    ledger = SqliteRunLedger(database_path)
    ledger.create_run(run_id="run-1", submission_key="request-1", request=_request())
    ledger.close()

    passive = checkpoint_database(database_path)
    truncated = checkpoint_database(database_path, truncate=True)

    assert passive[0] == 0
    assert truncated == (0, 0, 0)


def test_schema_and_pragmas_have_one_canonical_python_source() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    canonical = repository_root / "infrastructure" / "db" / "database.py"
    candidates = [
        *repository_root.joinpath("infrastructure", "db").glob("*.py"),
        *repository_root.joinpath("infrastructure", "scripts").glob("*.py"),
    ]

    ddl_owners = [
        path
        for path in candidates
        if "CREATE TABLE" in path.read_text(encoding="utf-8")
    ]
    pragma_owners = [
        path
        for path in candidates
        if "PRAGMA journal_mode" in path.read_text(encoding="utf-8")
    ]

    assert ddl_owners == [canonical]
    assert pragma_owners == [canonical]


def test_run_submission_is_idempotent_by_submission_key(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")

    first = ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )
    second = ledger.create_run(
        run_id="run-2", submission_key="request-1", request=_request("ignored")
    )

    assert first["run_id"] == "run-1"
    assert second["run_id"] == "run-1"
    assert len(ledger.list_runs()) == 1
    assert [event["event_type"] for event in second["events"]] == [
        "analysis_queued"
    ]
    assert ledger.get_run("missing") is None


def test_list_runs_batches_event_projection_and_handles_empty_database(
    tmp_path: Path,
) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    assert ledger.list_runs() == []
    for index in range(3):
        ledger.create_run(
            run_id=f"run-{index}",
            submission_key=f"request-{index}",
            request=_request(str(index)),
        )

    runs = ledger.list_runs()
    assert len(runs) == 3
    assert all(run["events"][0]["event_type"] == "analysis_queued" for run in runs)


def test_run_lifecycle_and_cancellation_are_atomic(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )

    assert ledger.mark_running("run-1")
    ledger.append_event("run-1", "scan_completed", {"files": 3})
    assert ledger.cancel("run-1")
    assert not ledger.succeed("run-1", {"report": "must not persist"})
    ledger.append_event("run-1", "late_event", {})

    run = ledger.get_run("run-1")
    assert run is not None
    assert run["status"] == "cancelled"
    assert run["result"] is None
    assert [event["event_type"] for event in run["events"]] == [
        "analysis_queued",
        "analysis_started",
        "scan_completed",
        "analysis_cancelled",
    ]


def test_interrupted_runs_recover_without_duplicate_completion(
    tmp_path: Path,
) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )
    assert ledger.mark_running("run-1")

    assert ledger.recover_interrupted() == 1
    assert ledger.recover_interrupted() == 0
    assert ledger.mark_running("run-1")
    assert ledger.succeed("run-1", {"report": "done"})
    assert not ledger.succeed("run-1", {"report": "duplicate"})

    run = ledger.get_run("run-1")
    assert run is not None
    assert run["status"] == "succeeded"
    assert run["result"] == {"report": "done"}
    assert [event["event_type"] for event in run["events"]] == [
        "analysis_queued",
        "analysis_started",
        "analysis_recovered",
        "analysis_started",
        "analysis_completed",
    ]


def test_failure_and_shutdown_requeue_preserve_terminal_invariants(
    tmp_path: Path,
) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db")
    ledger.create_run(
        run_id="failed", submission_key="failed-request", request=_request()
    )
    assert ledger.fail("failed", "provider unavailable")
    assert not ledger.mark_running("failed")
    failed = ledger.get_run("failed")
    assert failed is not None
    assert failed["error"] == "provider unavailable"
    assert failed["events"][-1]["event_type"] == "analysis_failed"

    ledger.create_run(
        run_id="interrupted",
        submission_key="interrupted-request",
        request=_request(),
    )
    assert ledger.mark_running("interrupted")
    assert ledger.requeue_interrupted("interrupted")
    assert not ledger.requeue_interrupted("interrupted")
    interrupted = ledger.get_run("interrupted")
    assert interrupted is not None
    assert interrupted["status"] == "queued"
    assert interrupted["events"][-1]["event_type"] == "analysis_interrupted"


def test_concurrent_event_writers_preserve_unique_sequence(
    tmp_path: Path,
) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db", event_history_limit=200)
    ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )
    assert ledger.mark_running("run-1")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: ledger.append_event(
                    "run-1", "worker_progress", {"index": index}
                ),
                range(64),
            )
        )

    run = ledger.get_run("run-1")
    assert run is not None
    sequences = [event["sequence"] for event in run["events"]]
    assert len(sequences) == 66
    assert sequences == list(range(1, 67))


def test_writer_waits_for_short_lock_and_completes(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    ledger = SqliteRunLedger(database_path)
    ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )
    assert ledger.mark_running("run-1")

    blocker = sqlite3.connect(database_path, isolation_level=None)
    blocker.execute("PRAGMA journal_mode = WAL")
    blocker.execute("BEGIN IMMEDIATE")
    started = threading.Event()

    def append() -> None:
        started.set()
        ledger.append_event("run-1", "after_lock", {})

    thread = threading.Thread(target=append)
    thread.start()
    assert started.wait(timeout=1)
    time.sleep(0.05)
    blocker.commit()
    blocker.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    run = ledger.get_run("run-1")
    assert run is not None
    assert run["events"][-1]["event_type"] == "after_lock"


def test_bounded_busy_timeout_exhausts_under_persistent_writer_lock(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "codeinsight.db"
    initialize_database(database_path)
    blocker = open_database(database_path)
    contender = open_database(database_path, busy_timeout_ms=25)
    blocker.execute("BEGIN IMMEDIATE")
    started = time.perf_counter()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            contender.execute("BEGIN IMMEDIATE")
    finally:
        blocker.rollback()
        blocker.close()
        contender.close()

    elapsed = time.perf_counter() - started
    assert 0.015 <= elapsed < 1


def test_sqlite_backup_includes_committed_wal_state(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    backup_path = tmp_path / "backup.db"
    ledger = SqliteRunLedger(database_path)
    ledger.create_run(
        run_id="run-1", submission_key="request-1", request=_request()
    )

    with database_connection(database_path) as source:
        with closing(sqlite3.connect(backup_path)) as destination:
            source.backup(destination)

    with closing(sqlite3.connect(backup_path)) as backup:
        request_json = backup.execute(
            "SELECT request_json FROM runs WHERE run_id = 'run-1'"
        ).fetchone()[0]
    assert json.loads(request_json) == _request()
