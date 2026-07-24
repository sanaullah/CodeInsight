# CodeInsight Technical Audit and Modernization Plan

**Review date:** July 24, 2026  
**Reviewed revision:** `e28c0f3` on `main`  
**Status:** Authoritative modernization record

This is the single source of truth for the audit, target architecture, roadmap,
security and agent guardrails, dependency policy, and validation plan.

## Executive assessment

CodeInsight is a substantial alpha prototype: 184 Python files and about 41,800
lines implementing Streamlit, a 17-node LangGraph, LiteLLM/LangChain, Langfuse,
and PostgreSQL/Redis storage. Its core idea is sound: combine repository
architecture context with focused specialist reviews and one synthesis.

It is not yet a dependable polyglot reviewer. It is primarily a
whole-repository LLM analyzer whose effective default coverage is Python,
JavaScript, and TypeScript. At audit time it had no tests, CI, lockfile, static
quality gates, or measured accuracy corpus. Several state fan-out, tool-loop,
setup, configuration, and security defects affect correctness. Modernization
must proceed in this order:

1. establish a reproducible, tested baseline;
2. fix execution and security invariants;
3. add Git-diff-aware semantic indexing and deterministic analyzers;
4. simplify orchestration into a bounded task DAG;
5. expand languages and learning only when evaluations justify it.

## Evidence and uncertainty

- Git contains 21 commits beginning January 25, 2026. Architecture docs are
  dated December 9, 2025, so earlier work may have been imported.
- `main` was at `e28c0f3` (June 14, 2026). One later `development` commit did
  not address the critical findings below.
- `pytest --collect-only` found no tests. Python syntax parsed, but the legacy
  workflow was not run end to end.
- Docker, databases, and live LLMs were not invoked. Runtime compatibility,
  latency, and review precision/recall remain unknown.
- Package versions below were verified against PyPI JSON on July 24, 2026.
  “Current” does not mean “drop-in compatible.”

## Current-state inventory

- Streamlit is the original UI and mixes presentation with threading, event
  queues, orchestration, persistence, and formatting.
- `SwarmAnalysisOrchestrator` invokes a 17-node LangGraph: scan, LLM
  architecture inference, previous report, LLM role selection, parallel prompt
  generation/validation, metadata “spawn,” parallel role execution, and
  synthesis.
- “Agents” are dictionaries, not isolated agent objects. Role branches can run
  concurrently; chunks inside one role run sequentially.
- LiteLLM/LangChain abstract providers; Langfuse records traces.
- PostgreSQL stores settings, scans, prompts, architecture, experiences, and
  skills. Redis caches. The bundled Langfuse stack adds PostgreSQL, Redis,
  ClickHouse, and MinIO.
- There was no `pyproject.toml`, lock, tests, CI, linter/formatter/type checker,
  coverage policy, security policy, CODEOWNERS, SBOM, or automated security
  scanning. This slice begins the `uv`, API, frontend, and test baseline.

## Audit findings

### Critical — language claims exceed implementation

- Thirty language enums/extensions are advertised, but only twelve have rich
  metadata.
- The scanner factory always supplies Python/JS/TS extensions. The scanner only
  auto-detects when no extensions are supplied, so default auto-detection is
  bypassed.
- The imported `parsers.multi_language_parser` package does not exist and
  silently falls back to Python-only AST metadata.
- Dependency resolution uses Python AST, JS/TS regex, and generic regex for
  every other language.
- No compiler, LSP, Tree-sitter runtime, native analyzer, Git diff, PR,
  base/head, or changed-line review exists.
- Architecture inference selects at most twenty files and sends only the first
  ten files' first fifty lines, fenced as Python regardless of language.

Publish support tiers honestly: **parsed** (Python), **dependency-aware**
(Python/JS/TS), and **discovery/raw text** (remaining advertised languages).

### Critical — repository processing is inefficient

- Scanning performs a recursive traversal per extension and reads every file.
- Detection traverses again.
- Full content lives in graph state and is copied into every role's `Send`.
- Dynamic selection was off by default.
- Chunks pack files and split oversized files by line, not syntax/symbol.
- There is no content-addressed index, symbol graph, incremental invalidation,
  or analyzer cache.

Memory/token use trends toward agents × repository size, while unrelated code
dilutes attention and repeated reviews redo unchanged work.

### Critical — tool-loop correctness

- A response with no tool calls does not immediately return and can repeat
  until the iteration cap.
- A first-call timeout/error can break before `llm_result` is assigned.
- Each iteration appends the entire original source request, causing context
  growth.
- Full tool outputs/history remain in shared state.

### Critical — fan-out drops guardrails

LangGraph `Send` replaces state, but the agent payload omits authorized roles,
architecture hash, tool enablement, tool/iteration limits, timeout, and cache.
Authorization only rejects when a non-empty whitelist exists, making the path
fail open. Architecture constraints and budgets silently disappear.
`file_cache` and `_file_cache` are also inconsistent.

### High — setup, dependency, and configuration drift

- `setup.py` calls undefined `get_activation_command`; a return is misplaced.
- README referenced nonexistent `config.yaml.example`.
- Docs prescribe `custom_openai/`; the factory emits `openai/`.
- YAML prices use `*_per_1m`, the Pydantic model declares `*_per_1k`, and cost
  tracking reads ignored fields, producing zero.
- YAML sections absent from Pydantic models are ignored.
- No tracked `default_model` exists, leaving a hard-coded legacy fallback.
- Lower-bound-only requirements silently cross major versions.
- Architecture docs reference removed MCP support and nonexistent schema files.

### High — security is development-only

File tools correctly resolve paths beneath the project root. Remaining risks:

- arbitrary server directories without authentication/tenant boundaries;
- untrusted repository text, reports, tools, and skills enter prompts without a
  complete injection/data-exfiltration policy;
- no systematic secret/PII redaction before models, reports, logs, or traces;
- role-name regexes are treated as security although labels do not constrain
  capability;
- Compose floats `latest`, uses development passwords/zero encryption key,
  enables telemetry/experimental features, and exposes web/MinIO broadly;
- no SAST, dependency, secret, container, or SBOM automation.

### High — autonomous learning reinforces unverified output

Synthesis creates one experience and storage creates another, so extracted
skills can reference the wrong provenance ID. LLM-authored skills are persisted
and reinjected without review, project/tenant isolation, expiry, or robust
deduplication. “Quality” rewards report length, Markdown, words like
`file`/`line`/`path`, and completion—not correctness. This permits learning-store
poisoning and self-reinforcing verbose false positives.

### Medium — maintainability and operational sprawl

Twenty-five files exceed roughly 400 lines; the main execution node is about
1,400. Duplicate storage utilities/services, legacy wrappers, deprecated
callbacks/chunks, removed MCP branches, and placeholder HITL paths remain. The
six-service-looking local footprint is excessive for a desktop alpha.

## Verified dependency findings and policy

| Cohort | Verified current releases (2026-07-24) |
| --- | --- |
| Orchestration | LangGraph 1.2.9; LangChain 1.3.14; langchain-core 1.5.1; langchain-litellm 0.7.0 |
| Models/observability | LiteLLM 1.91.4; Langfuse 4.14.1; Pydantic 2.13.4; OpenTelemetry 1.44.0 |
| API/UI/test | FastAPI 0.139.2; Uvicorn 0.51.0; Streamlit 1.60.0; pytest 9.1.1; pytest-asyncio 1.4.0; HTTPX 0.28.1 |
| Infrastructure | PyYAML 6.0.3; Redis 8.0.1; psycopg2-binary 2.9.12; clickhouse-driver 0.2.11; boto3 1.43.56; tiktoken 0.13.0; python-dotenv 1.2.2 |

`pyproject.toml` constrains compatible majors; `uv.lock` records the exact
graph. Framework upgrades are one tested cohort, not independent bumps.
Automated updates must modify the lock in a branch and run contract,
integration, and review-quality evaluations. Consider psycopg 3 separately
after storage tests exist.

LiteLLM is intentionally constrained below 1.92. PyPI metadata shows that
1.92+ no longer publishes a universal Python wheel; on Windows it falls back
to a Rust build that requires the Visual C++ linker. Version 1.91.4 is the
newest release in the universal-wheel line verified during this migration.
Re-test and remove the cap when upstream publishes supported Windows wheels.

## Target polyglot architecture

```text
Git change set + manifests
        ↓
one-pass content-addressed repository index
        ↓
language adapters + native analyzers
        ↓
symbol / import / call / type / test graph
        ↓
changed-symbol impact-cone planner
        ↓
bounded specialist workers
        ↓
evidence verifier + finding deduplicator
        ↓
human report + SARIF
```

### Repository index

- Prefer tracked Git blobs and base/head diff; mark untracked, generated,
  vendor, binary, sensitive, and oversized files.
- Traverse once; cache by content, parser, analyzer, and config version.
- Keep source in a controlled artifact store. State carries immutable IDs and
  small metadata, not full repository copies.
- Capture symbols, spans, imports, calls, types, ownership, manifests, tests,
  and build configuration.

### Language-adapter contract

Each adapter implements:

```text
detect → parse symbols → dependency edges → native tools/tests
       → normalized findings/diagnostics
```

Start with Python and TypeScript/JavaScript. Add languages only from measured
demand and corpus results. Publish discovery, parsed, dependency-aware,
analyzer-backed, and tested tiers.

### Review behavior

- Review a changed symbol plus bounded dependency/test impact cone, not an
  arbitrary token block.
- Deterministic analyzers produce candidates; LLMs correlate, reason across
  files, explain, prioritize, and propose fixes.
- Every finding carries category/rule, severity/confidence, exact file/span,
  evidence, fix, stable fingerprint, and analyzer/model/prompt provenance.
- Reject findings without resolvable evidence.
- Import/export SARIF for code-host interoperability.

## Target orchestration

Use a bounded, inspectable DAG:

1. controller creates immutable `ReviewRun` with diff, policy, versions, budgets;
2. deterministic planner emits typed index/language/security/test/architecture tasks;
3. queue workers fan out by language/artifact with schema inputs/outputs;
4. verifier checks evidence, paths/spans, reachability, contradiction, severity;
5. aggregator fingerprints findings; synthesizer formats verified results.

Workers do not recursively spawn workers. Dynamic tasks require planner schema
and remaining budget. Add per-provider concurrency/rate limits, token/time/cost
budgets, idempotency, retry/backoff, circuit breakers, cancellation,
checkpoints, and partial-result semantics.

## Guardrails

- Read-only by default; shell/network/write/execute require explicit policy.
- Resolve paths and symlinks beneath an approved root.
- Enforce file/byte/tool/iteration/token/time/cost/concurrency/trace limits at
  worker boundaries.
- Treat code as untrusted data, separate it from instructions, and block
  injected attempts to reveal secrets or change policy.
- Redact credentials, PII, and sensitive paths before model calls and traces.
- Require auth, authorization, tenant/project scopes, quotas, retention, and
  audit before network deployment.
- Learned artifacts carry run, commit, model, prompt, evaluator, project,
  version, expiry, and approval; promotion is offline, reversible, canaried.
- Pin images by digest and fail fast on dev credentials outside dev profiles.

## FastAPI/frontend migration

The first slice adds a stable `/api/v1` boundary, bounded typed requests,
queued/running/succeeded/failed/cancelled lifecycle, concurrent-run limits,
lazy adaptation to the existing orchestrator, health/capability/run/cancel
endpoints, and a responsive same-origin frontend with real submission, events,
reports, and run history. Streamlit remains the migration fallback.

The in-memory run registry proves the boundary without coupling new code to
legacy storage. Next, define `ReviewRunRepository`, persist runs/events, and
move execution to durable workers. Authentication is mandatory before binding
beyond trusted loopback.

## Phased roadmap

### Phase 0 — this slice

- Canonical `uv` metadata and lock.
- FastAPI API and integrated frontend; Streamlit preserved.
- Tests for validation, paths, lifecycle, health, frontend, and errors.
- This consolidated audit/plan.

### Phase 1 — stabilize (days to two weeks)

- Fix tool loop, `Send` payloads, fail-open authorization, cache keys,
  setup/config/model/cost defects.
- Split execution into typed planner, worker, tool loop, verifier, reducer.
- Add Ruff, Pyright/mypy, coverage gate, CI matrices, dependency/secret/config/
  container scans, SBOM.
- Disable tool calling and ACE writes by default.
- Pin Compose images; explicit dev profiles; loopback/auth enforcement.
- Publish honest language tiers in all UI/docs.

### Phase 2 — semantic review core (two to eight weeks)

- Git base/head and working-tree diff ingestion.
- One-pass content-addressed index and artifact references.
- Python and TS/JS adapters, syntax chunks, impact analysis, native tools.
- Normalized findings, evidence checks, fingerprints, SARIF.
- Queue-backed bounded DAG and durable run/event storage.
- Default UI to changed-code review; full scans become incremental baselines.

### Phase 3 — scale and measured expansion

- Add adapters based on corpus/customer evidence.
- Separate API/UI, analyzer workers, and optional observability/storage.
- Tenant isolation, RBAC, quotas, retention/residency, provider routing, signed
  audit.
- Re-enable learned policies only through offline evaluation, review,
  versioning, canary, decay, rollback.

## Validation plan

### Code and resilience

- Unit/property tests for adapters, chunks, path/symlink escape, config,
  fingerprints, reducers, budgets, and redaction.
- Tool-loop tests for no-tools, malformed/repeated calls, timeout, cancellation,
  provider failure, and output limits.
- API contracts for lifecycle, concurrency, auth, pagination, persistence,
  cancellation, and partial failure.
- Integration with deterministic fake LLM/analyzers and injected DB/provider
  failures.

### Review-quality evaluation

Build a versioned corpus of small real polyglot repos and seeded patches with
correctness, security, API misuse, cross-file, build, and test-impact defects.
Expected findings include exact spans and severity.

Primary metrics: span-level precision/recall/F1, false positives per thousand
changed lines, citation validity, cross-file/test-impact recall, severity
calibration, duplicate rate, reviewer acceptance, escaped defects.

Operational metrics: p50/p95 latency, peak memory/concurrency, tokens/cost per
changed line, cache hit rate, cancellation/retry/circuit behavior.

Model, prompt, analyzer, and dependency changes run offline, then shadow, then
canary. Length/Markdown proxy scores must not gate promotion.

## Modernization definition of done

- `uv sync --locked` is reproducible and CI is green.
- Supported languages have adapter and corpus evidence.
- Diff review is primary and full scans are incremental.
- Every finding has validated evidence and provenance.
- Accuracy and operational thresholds are enforced.
- Capabilities and learned artifacts fail closed.
- API lifecycle is durable, authenticated, isolated, and audited.
- Streamlit can retire without loss of supported functionality.
