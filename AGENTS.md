# CodeInsight Contribution Guide

## Scope and invariants

- Keep repository analysis read-only. Product features must not write to analyzed repositories.
- Keep SQLite/WAL as the durable system of record. `infrastructure/db/database.py` is the only owner of schema initialization, migrations, and database pragmas.
- Preserve immutable repository snapshots and distinguish exact evidence from inferred, partial, truncated, unsupported, and unavailable results.
- Keep optional integrations non-blocking and redact secrets and repository content from telemetry by default.

## Change discipline

- Work in the active checkout only; do not create or move worktrees unless the user explicitly requests it.
- Preserve unrelated user changes and ignored local plans. Never add files under `docs/` solely to publish a local roadmap.
- Split product work into reviewable conventional commits and rebuild `web/` whenever `frontend/` production source changes.
- Update API contracts, frontend types, tests, and documentation together when a public behavior changes.
- Prefer bounded queries and deterministic evidence links over plausible-looking derived claims.

## Validation

- Frontend changes: run `npm.cmd run check`, `npm.cmd test`, and `npm.cmd run build` from `frontend/`.
- Python changes: run the focused tests, full branch-aware coverage gate, Ruff, and compile checks defined in `pyproject.toml`.
- Before committing, run `git diff --check`, inspect the staged diff, and record any validation limitation rather than looping on browser setup.
- Timebox live browser validation to two minutes; deterministic tests and the production build remain required.
