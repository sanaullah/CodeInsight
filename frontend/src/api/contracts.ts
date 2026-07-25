export type AnalysisStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "needs_attention";

export type AnalysisMode = "quick" | "deep" | "security" | "change-set";

export interface RuntimeIdentity {
  application_server: "FastAPI";
  environment_manager: "uv";
  database_engine: "SQLite";
  database_journal_mode: "WAL";
  database_schema_version: number;
  artifact_store: "filesystem";
  api_docs_url: string;
  read_only_analysis: true;
  build_commit: string | null;
  build_time: string | null;
}

export interface HealthResponse {
  status: "ok";
  version: string;
  active_analyses: number;
  max_concurrent_analyses: number;
  runtime: RuntimeIdentity;
}

export type LanguageSupport = "parsed" | "dependency-aware" | "discovery";

export interface LanguageCapability {
  language: string;
  display_name: string;
  support_level: LanguageSupport;
  extensions: string[];
}

export interface CapabilitiesResponse {
  languages: LanguageCapability[];
  default_max_agents: number;
  max_agents: number;
  analysis_modes: AnalysisMode[];
  native_durable_workflow: boolean;
  model_provider_configured: boolean;
  langfuse_enabled: boolean;
}

export interface AnalysisRequest {
  project_path: string;
  goal?: string | null;
  model_name?: string | null;
  max_agents: number;
  file_extensions?: string[] | null;
  selected_directories?: string[] | null;
  mode: AnalysisMode;
  max_waves: number;
  max_tasks: number;
  max_total_tokens: number;
  max_cost_usd: number;
  max_elapsed_seconds: number;
}

export interface AnalysisEvent {
  sequence: number;
  event_type: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface AnalysisRun {
  run_id: string;
  status: AnalysisStatus;
  request: AnalysisRequest;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  events: AnalysisEvent[];
  result: Record<string, unknown> | null;
  error: string | null;
  current_stage: string | null;
  snapshot_id: string | null;
  mode: AnalysisMode;
}

export interface AnalysisAccepted {
  run_id: string;
  status: AnalysisStatus;
  status_url: string;
}

export interface RoleRecord {
  role_id: string;
  wave_id: string;
  name: string;
  mission: string;
  rationale: string;
  coverage_targets: string[];
  required_capabilities: string[];
  model_policy: string;
}

export interface TaskRecord {
  task_id: string;
  wave_id: string;
  role_id: string | null;
  task_type: string;
  status: string;
  attempt_count: number;
  max_attempts: number;
  attempt_usage: Record<string, Record<string, number>>;
  error: { message?: string } | null;
}

export interface WaveRecord {
  wave_id: string;
  wave_number: number;
  rationale: string;
  status: string;
  created_at: string;
  completed_at: string | null;
}

export interface CanonicalFinding {
  finding_id: string;
  fingerprint: string;
  title: string;
  claim: string;
  severity: "info" | "low" | "medium" | "high" | "critical";
  confidence: number;
  candidate_ids: string[];
  supporting_evidence_ids: string[];
  conflicting_candidate_ids: string[];
  recommendation: string;
}

export interface FindingCandidate {
  candidate_id: string;
  title: string;
  claim: string;
  category: string;
  affected_path: string | null;
  impact: string;
  proposed_severity: CanonicalFinding["severity"];
  proposed_confidence: number;
  recommendation: string;
}

export interface CandidateVerdict {
  disposition: "accepted" | "rejected" | "needs-more-evidence" | "suggestion-only";
  calibrated_confidence: number;
  calibrated_severity: CanonicalFinding["severity"];
  rationale: string;
}

export interface CandidateRecord {
  candidate: FindingCandidate;
  verdict: CandidateVerdict | null;
}

export interface CoverageRecord {
  assessment_id: string;
  run_id: string;
  wave_id: string;
  measured: Record<string, number>;
  exclusions: string[];
  unsupported_areas: string[];
  unresolved_uncertainty: string[];
  remaining_gaps: string[];
  proposed_follow_up_task_ids: string[];
  confidence: number;
  follow_up_decision: "launch" | "defer" | "complete";
  decision_rationale: string;
}

export interface ModelUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

export interface AnalysisIntelligence {
  run_id: string;
  status: AnalysisStatus;
  current_stage: string | null;
  snapshot_id: string | null;
  waves: WaveRecord[];
  roles: RoleRecord[];
  tasks: TaskRecord[];
  candidates: CandidateRecord[];
  findings: CanonicalFinding[];
  coverage: CoverageRecord[];
  model_calls: Array<Record<string, unknown>>;
  usage: ModelUsage;
}

export interface RecoveryResponse {
  recovered_runs: number;
  recovered_tasks: number;
  scheduled_runs: number;
}

export type FindingReviewState =
  | "new"
  | "validated"
  | "acknowledged"
  | "reviewed"
  | "dismissed"
  | "reopened"
  | "resolved";

export interface FindingSummary extends CanonicalFinding {
  run_id: string;
  review_state: FindingReviewState;
  review_note: string | null;
  review_version: number;
  reviewed_at: string | null;
}

export interface EvidenceDetail {
  evidence_id: string;
  relative_path: string;
  content_hash: string;
  start_line: number;
  end_line: number;
  evidence_kind: string;
  provenance: Record<string, unknown>;
  excerpt: string | null;
  integrity: "valid" | "invalid" | "unavailable";
  redacted: boolean;
  truncated?: boolean;
}

export interface FindingDetail extends FindingSummary {
  candidates: Array<{
    relationship: string;
    candidate: FindingCandidate;
    role: RoleRecord | null;
    verdict: CandidateVerdict | null;
  }>;
  evidence: EvidenceDetail[];
  review_history: Array<{
    review_event_id: number;
    previous_state: FindingReviewState | null;
    review_state: FindingReviewState;
    note: string | null;
    actor: string;
    created_at: string;
  }>;
}

export interface FindingPage {
  items: FindingSummary[];
  next_cursor: string | null;
  counts_by_severity: Partial<Record<CanonicalFinding["severity"], number>>;
}

export interface SnapshotSummary {
  snapshot_id: string;
  project_id: string;
  display_name: string;
  git_repository: string | null;
  base_commit: string | null;
  head_commit: string | null;
  dirty: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
  file_count: number;
  symbol_count: number;
  edge_count: number;
}

export interface ArchitectureNode {
  node_id: string;
  label: string;
  node_kind: "file";
  language: string | null;
  classification: string;
  support_tier: string;
  line_count: number;
  confidence: number;
  derivation: "repository-index";
  findings: Array<{ finding_id: string; severity: CanonicalFinding["severity"] }>;
}

export interface ArchitectureEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  edge_kind: "imports" | "calls";
  confidence: number;
  derivation: "repository-index";
}

export interface ArchitectureGraph {
  snapshot_id: string;
  display_name: string;
  summary: {
    file_count: number;
    language_counts: Record<string, number>;
    classification_counts: Record<string, number>;
    changed_paths: string[];
  };
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  truncated: boolean;
  limits: { depth: number; node_limit: number };
}

export interface ArchitectureTrace {
  snapshot_id: string;
  found: boolean;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  max_hops: number;
}
