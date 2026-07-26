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
- The user authorizes committing and pushing completed, validated milestones to
  the existing configured Git remote without a separate conversational pause.
  Do not open a pull request, create a release, or push unrelated/local-only
  files without explicit authorization.
- Update API contracts, frontend types, tests, and documentation together when a public behavior changes.
- Prefer bounded queries and deterministic evidence links over plausible-looking derived claims.

## Validation

- Use fast checks while iterating. Before a normal milestone commit/push, run
  targeted tests plus the applicable lint, type, contract, and production-build
  gates for every affected path.
- Reserve the full repository release gate for explicitly authorized releases
  and unusually broad or high-risk changes. This does not reduce required
  tests or branch-aware coverage for changed runtime paths.
- Frontend changes: run `npm.cmd run check`, `npm.cmd test`, and `npm.cmd run build` from `frontend/`.
- Python changes: run focused tests, branch-aware coverage for changed runtime
  paths, Ruff, and compile checks defined in `pyproject.toml`.
- Before committing, run `git diff --check`, inspect the staged diff, and record any validation limitation rather than looping on browser setup.
- Timebox live browser validation to two minutes; deterministic tests and the production build remain required.

## Versioning and releases

- Use Semantic Versioning for product releases; do not bump a version for every commit.
- `api/config.py` `VERSION_STRING` is the current canonical release-display source used by FastAPI and the frontend footer. Keep Python package metadata (`pyproject.toml`/`uv.lock`) and frontend package metadata synchronized representations only when an authorized release changes that source.
- Keep the release version separate from `CODEINSIGHT_BUILD_COMMIT` and `CODEINSIGHT_BUILD_TIME`; the footer and diagnostics must expose build identity independently.
- Before 1.0, use `0.x.y-alpha.N`, `0.x.y-beta.N`, or `0.x.y-rc.N` as appropriate. Patch releases contain compatible fixes; minor releases represent coherent user-facing milestones.
- Do not declare `1.0.0` until the documented stable-core acceptance criteria and release gates pass.
- Create tags or hosted releases only with explicit release authorization. Detailed Conventional Commit messages remain independent of release numbering.
