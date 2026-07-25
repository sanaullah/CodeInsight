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
```

No Langfuse credentials or server are required for normal runs. Prompt/source
content is omitted unless `CODEINSIGHT_LANGFUSE_CAPTURE_CONTENT=true` is
explicitly set; key, token, secret, and password fields remain redacted.

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
