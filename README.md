# CodeInsight

CodeInsight is a local-first repository analysis service. It builds one shared
repository snapshot, plans repository-specific specialist roles, executes them
through a bounded native scheduler, verifies evidence, correlates duplicate
findings, measures coverage, and may run bounded follow-up waves.

FastAPI serves both the durable analysis API and the browser workspace. The
application uses one SQLite database as its complete control-plane ledger and
content-addressed files for large immutable source artifacts.

## Current capabilities

- One repository snapshot per content/configuration version, with files,
  symbols, imports, calls, tests, ownership, Git changes, and change-set graph
  neighborhoods.
- Native repository-driven role and wave planning with strict Pydantic
  contracts, trusted specialist execution, bounded concurrency, retries,
  cancellation, and interrupted-run recovery.
- Typed evidence and finding proposals, immutable source-span verification,
  deterministic deduplication/correlation, explicit coverage gaps, and bounded
  follow-up decisions.
- Quick, deep, security, and change-set modes with run, task, token, cost, and
  elapsed-time budgets.
- An OpenAI-compatible provider boundary suitable for local servers. With no
  provider URL, runs operate honestly in index-only mode and do not present
  model-generated findings.
- Optional best-effort Langfuse trace export behind the internal event
  interface. It is asynchronous, redacted by default, and never required for
  local analysis or run success.
- A responsive React workspace for submitting and recovering reviews,
  inspecting live roles/waves/tasks, reviewing evidence-backed findings,
  tracing bounded repository architecture, comparing history/trends, and
  managing durable non-secret defaults and path-free presets.

LangGraph, LangChain, LiteLLM, Redis, PostgreSQL, ClickHouse, MinIO, and Docker
are not application runtime requirements.

## Local setup

Requirements:

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 22 or newer for frontend development/builds only

Create the locked environment:

```bash
uv sync --dev --locked
```

Install and build the browser workspace when frontend sources change:

```bash
cd frontend
npm ci
npm run build
cd ..
```

FastAPI serves the resulting production bundle from `web/`; Node is not a
production server. For frontend-only development, `npm run dev` proxies `/api`
requests to the FastAPI process on port 8000.

Copy `.env.example` to `.env` if local overrides are needed. To enable a local
OpenAI-compatible model server, set:

```text
CODEINSIGHT_MODEL_BASE_URL=http://127.0.0.1:1234/v1
CODEINSIGHT_DEFAULT_MODEL=local-model
```

By default, local application state is stored in the ignored
`.codeinsight/codeinsight.db` file under this checkout. Set
`CODEINSIGHT_DATA_DIR` or `CODEINSIGHT_DATABASE_PATH` in `.env` to use a
different durable location.

### Opt-in live provider compatibility smoke test

The deterministic suite never calls a model provider. To explicitly authorize
one bounded end-to-end API analysis against the configured OpenAI-compatible
endpoint, first configure the endpoint, model, and credentials in `.env`, then
run:

```powershell
$env:CODEINSIGHT_RUN_LIVE_PROVIDER_TEST = "1"
uv run pytest -m live_provider tests/test_live_provider_smoke.py
Remove-Item Env:CODEINSIGHT_RUN_LIVE_PROVIDER_TEST
```

The test uses one agent, one task, one wave, an 8,000-token run budget, a
2,500-token output cap, a $0.10 cost ceiling, a 45-second provider-call
timeout, and a 90-second overall deadline. It disables Langfuse and does not
print credentials, endpoint URLs, fixture source, prompts, or raw provider
responses. Without explicit opt-in, an endpoint, a non-default model, and
credentials, it skips without network access. The live smoke deliberately uses
the schema-in-prompt `json_object` compatibility mode; deterministic contract
tests separately cover strict `json_schema` negotiation and automatic fallback.

Initialize or upgrade the single application database explicitly when desired:

```bash
uv run python -m infrastructure.scripts.init_database
```

`infrastructure/db/database.py` is the only source that owns database
initialization, pragmas, DDL, schema checksums, and in-place version upgrades.
The normal application startup also initializes or upgrades that same database.

Start the application:

```bash
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

On Windows PowerShell, if `uv` is not installed globally but this checkout's
locked environment already exists, use the worktree-local executable:

```powershell
.\.venv\Scripts\uv.exe run uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`. Interactive API documentation is available at
`/api/docs`.

## Optional Langfuse export

Install the optional adapter and configure credentials:

```bash
uv sync --dev --locked --extra observability
```

```text
CODEINSIGHT_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://your-langfuse-host
CODEINSIGHT_LANGFUSE_CAPTURE_PROMPTS=false
CODEINSIGHT_LANGFUSE_CAPTURE_COMPLETIONS=false
```

No Langfuse package, credentials, or server are required for normal runs.
SQLite remains authoritative and import, initialization, queue, send, and
flush failures cannot fail an analysis. Export is metadata-only by default and
correlates stable run, stage, wave, role, task, attempt, model-call, and prompt
artifact IDs plus bounded token/cost metadata.

Generated source-free specialist prompts and model completions have independent
opt-ins. Raw source/file bodies, source-bearing user prompts, tool output,
absolute paths, credentials, authorization data, cookies, and provider secrets
remain omitted or redacted even when either content switch is enabled.

To smoke-test an operator-managed Docker/self-hosted Langfuse instance, install
the optional extra, configure its project keys/host, then explicitly authorize:

```powershell
$env:CODEINSIGHT_RUN_LANGFUSE_DOCKER_SMOKE = "1"
.\.venv\Scripts\python.exe -m pytest `
  tests/test_live_langfuse_smoke.py -s
```

The smoke emits only synthetic IDs and token/cost metadata, queries the
observation back, and is skipped by default. Starting, upgrading, or deleting a
Docker deployment remains an operator action; the test never manages volumes.

## Structured-output provider profiles

The direct OpenAI-compatible gateway remains the default. Select an Instructor
adapter only after testing the exact endpoint/model pair:

```text
CODEINSIGHT_PROVIDER_CAPABILITY_PROFILE=direct
# alternatives: instructor-json, instructor-json-schema, instructor-tools
```

Profiles never silently fall back after a runtime provider, timeout, or
validation error. The default-skipped compatibility gates require explicit
authorization and print only aggregate status, latency, token, and cost data:

```powershell
$env:CODEINSIGHT_RUN_LIVE_PROVIDER_TEST = "1"
.\.venv\Scripts\python.exe -m pytest `
  tests/test_live_provider_smoke.py::test_live_structured_output_ab_compatibility -s

# Optional bounded replay of request metadata/latest files from a local run.
$env:CODEINSIGHT_LIVE_REPLAY_RUN_ID = "<run-id>"
$env:CODEINSIGHT_LIVE_REPLAY_PROFILE = "instructor-json"
.\.venv\Scripts\python.exe -m pytest `
  tests/test_live_provider_smoke.py::test_live_instructor_profile_replays_persisted_request -s
```

The replay reads the selected request from the canonical ledger, analyzes the
repository read-only, and writes results to an isolated temporary ledger. A
profile must not become the default unless the representative replay succeeds.

## HTTP lifecycle

- `POST /api/v1/analyses` submits a durable run.
- `GET /api/v1/analyses/{run_id}` returns lifecycle state and events.
- `GET /api/v1/analyses/{run_id}/intelligence` returns waves, roles, tasks,
  evidence verdicts, findings, coverage, model calls, and usage.
- `DELETE /api/v1/analyses/{run_id}` requests durable cancellation.
- `POST /api/v1/recovery` recovers expired work and schedules queued runs.
- `GET /api/v1/capabilities` reports language support, modes, provider state,
  and optional tracing state.
- `GET /api/v1/findings`, `/api/v1/snapshots`, and `/api/v1/history` expose
  bounded, cursor-ready review intelligence projections.
- `GET/PUT /api/v1/settings` and `/api/v1/presets` manage versioned,
  non-secret local defaults; provider credentials remain environment-only.

The browser workspace uses these same public endpoints; it has no hidden
in-process analysis path.

## Browser accessibility and performance

The browser workspace targets WCAG 2.2 AA. It provides keyboard-operable
navigation and dialogs, semantic tables and text alternatives for charts and
graphs, visible focus, reduced-motion and forced-colors support, responsive
layouts from 360px upward, and text-backed status/severity cues.

Production assets are served directly by FastAPI. The release suite enforces
gzip ceilings of 100 KiB for JavaScript and 20 KiB for CSS, cursor-paginates
large finding/history projections, and bounds architecture graph responses.

## Quality gates

```bash
uv run pytest
uv run pytest --cov=. --cov-report=term-missing
uv run ruff check .
uv lock --check
cd frontend && npm ci && npm run check && npm run test:coverage && npm run build
```

Python coverage is branch-aware and must remain at or above the configured 92
percent gate. The frontend enforces its own behavioral coverage thresholds.
Tests cover database migration/integrity/contention, run and task
recovery, cancellation/idempotency/concurrency, indexing, native analysis,
provider budgets, optional tracing, public API behavior, and browser contract
projection.

## Package layout

| Package | Responsibility |
| --- | --- |
| `api/` | FastAPI routes, settings, and public request/response contracts |
| `application/` | Durable run coordination, model budgets, and tracing ports |
| `domain/` | Framework-neutral analysis and workflow contracts |
| `indexing/` | Shared repository snapshot and language capability metadata |
| `analysis/native/` | Role planning, specialist harness, verification, correlation, coverage, and waves |
| `workflow/` | Native durable bounded task scheduler |
| `infrastructure/db/` | SQLite ledger repositories and canonical schema source |
| `infrastructure/artifacts/` | Content-addressed immutable artifact storage |
| `infrastructure/llm/` | Provider-neutral model transport adapter |
| `infrastructure/observability/` | Optional non-blocking Langfuse exporter |
| `frontend/` | React/TypeScript/Vite source, typed API client, and UI tests |
| `web/` | FastAPI-served production browser bundle |
| `tests/` | Behavioral, contract, integration, and performance gates |

## License

CodeInsight is available under the [MIT License](LICENSE).
