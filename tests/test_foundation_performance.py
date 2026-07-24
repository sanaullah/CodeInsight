from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

from infrastructure.db.database import initialize_database
from infrastructure.db.run_ledger import SqliteRunLedger


def test_cold_database_startup_smoke(tmp_path: Path) -> None:
    started = time.perf_counter()
    initialize_database(tmp_path / "codeinsight.db")
    elapsed = time.perf_counter() - started

    # A deliberately generous regression ceiling for local and CI disks.
    assert elapsed < 2.0


def test_run_event_throughput_and_memory_smoke(tmp_path: Path) -> None:
    ledger = SqliteRunLedger(tmp_path / "codeinsight.db", event_history_limit=500)
    ledger.create_run(
        run_id="run-1",
        submission_key="request-1",
        request={"project_path": "."},
    )
    assert ledger.mark_running("run-1")

    tracemalloc.start()
    started = time.perf_counter()
    for index in range(250):
        ledger.append_event("run-1", "progress", {"index": index})
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert elapsed < 5.0
    assert peak < 16 * 1024 * 1024
    run = ledger.get_run("run-1")
    assert run is not None
    assert len(run["events"]) == 252
