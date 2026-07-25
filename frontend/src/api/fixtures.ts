import type {
  AnalysisIntelligence,
  AnalysisRun,
  CapabilitiesResponse,
  HealthResponse,
} from "./contracts";

export const healthFixture: HealthResponse = {
  status: "ok",
  version: "v0.1.0-alpha",
  active_analyses: 1,
  max_concurrent_analyses: 2,
  runtime: {
    api_contract_version: 1,
    application_server: "FastAPI",
    environment_manager: "uv",
    database_engine: "SQLite",
    database_journal_mode: "WAL",
    database_schema_version: 1,
    artifact_store: "filesystem",
    api_docs_url: "/api/docs",
    read_only_analysis: true,
    build_commit: "fixture-build",
    build_time: "2026-07-25T12:00:00Z",
  },
};

export const capabilitiesFixture: CapabilitiesResponse = {
  languages: [
    {
      language: "python",
      display_name: "Python",
      support_level: "parsed",
      extensions: [".py"],
    },
    {
      language: "typescript",
      display_name: "TypeScript",
      support_level: "dependency-aware",
      extensions: [".ts", ".tsx"],
    },
  ],
  default_max_agents: 4,
  max_agents: 12,
  analysis_modes: ["quick", "deep", "security", "change-set"],
  native_durable_workflow: true,
  model_provider_configured: false,
  langfuse_enabled: false,
};

export const runningRunFixture: AnalysisRun = {
  run_id: "run-fixture-1",
  status: "running",
  request: {
    project_path: "H:\\projects\\payments-platform",
    goal: "Review authorization boundaries and reliability risks.",
    model_name: null,
    max_agents: 4,
    file_extensions: null,
    selected_directories: null,
    mode: "deep",
    max_waves: 2,
    max_tasks: 100,
    max_total_tokens: 1_000_000,
    max_cost_usd: 25,
    max_elapsed_seconds: 3_600,
  },
  created_at: "2026-07-25T12:00:00Z",
  started_at: "2026-07-25T12:00:01Z",
  completed_at: null,
  events: [
    {
      sequence: 1,
      event_type: "analysis_queued",
      timestamp: "2026-07-25T12:00:00Z",
      data: {},
    },
    {
      sequence: 2,
      event_type: "repository_index_ready",
      timestamp: "2026-07-25T12:00:03Z",
      data: { file_count: 124, symbol_count: 430, edge_count: 688 },
    },
  ],
  result: null,
  error: null,
  current_stage: "dispatch_tasks",
  snapshot_id: "snapshot-fixture-1",
  mode: "deep",
};

export const runningIntelligenceFixture: AnalysisIntelligence = {
  run_id: "run-fixture-1",
  status: "running",
  current_stage: "dispatch_tasks",
  snapshot_id: "snapshot-fixture-1",
  waves: [
    {
      wave_id: "wave-1",
      wave_number: 1,
      rationale: "Cover repository-specific languages and risk surfaces.",
      status: "running",
      created_at: "2026-07-25T12:00:03Z",
      completed_at: null,
    },
  ],
  roles: [
    {
      role_id: "role-architecture",
      wave_id: "wave-1",
      name: "Repository Architecture Specialist",
      mission: "Trace module boundaries, dependencies, and structural risks.",
      rationale: "Every analysis needs repository-wide structural context.",
      coverage_targets: ["architecture", "dependencies"],
      required_capabilities: ["architecture"],
      model_policy: "strong",
    },
    {
      role_id: "role-python",
      wave_id: "wave-1",
      name: "Python Implementation Specialist",
      mission: "Inspect Python implementation behavior and failure paths.",
      rationale: "Python is present in the target snapshot.",
      coverage_targets: ["language:python"],
      required_capabilities: ["language-analysis"],
      model_policy: "balanced",
    },
  ],
  tasks: [
    {
      task_id: "task-architecture",
      wave_id: "wave-1",
      role_id: "role-architecture",
      task_type: "specialist_analysis",
      status: "succeeded",
      attempt_count: 1,
      max_attempts: 3,
      attempt_usage: {},
      error: null,
    },
    {
      task_id: "task-python",
      wave_id: "wave-1",
      role_id: "role-python",
      task_type: "specialist_analysis",
      status: "leased",
      attempt_count: 1,
      max_attempts: 3,
      attempt_usage: {},
      error: null,
    },
  ],
  candidates: [],
  findings: [],
  coverage: [],
  model_calls: [],
  usage: {
    input_tokens: 420,
    output_tokens: 180,
    total_tokens: 600,
    cost_usd: 0,
  },
};

export const runStateFixtures: Record<string, AnalysisRun> = {
  queued: {
    ...runningRunFixture,
    run_id: "run-queued",
    status: "queued",
    started_at: null,
    current_stage: null,
    snapshot_id: null,
    events: [runningRunFixture.events[0]],
  },
  running: runningRunFixture,
  succeededIndexOnly: {
    ...runningRunFixture,
    run_id: "run-index-only",
    status: "succeeded",
    completed_at: "2026-07-25T12:00:08Z",
    current_stage: "complete",
    result: {
      provider_mode: "index-only",
      snapshot: { file_count: 124 },
      synthesized_report:
        "No evidence-backed findings were accepted; the model provider was not configured.",
    },
  },
  needsAttention: {
    ...runningRunFixture,
    run_id: "run-needs-attention",
    status: "needs_attention",
    completed_at: "2026-07-25T12:01:00Z",
    error: "Specialist retry budget was exhausted.",
  },
  failed: {
    ...runningRunFixture,
    run_id: "run-failed",
    status: "failed",
    completed_at: "2026-07-25T12:00:04Z",
    error: "Repository snapshot could not be completed.",
  },
  cancelled: {
    ...runningRunFixture,
    run_id: "run-cancelled",
    status: "cancelled",
    completed_at: "2026-07-25T12:00:04Z",
  },
};
