export type AnalysisStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "needs_attention";

export type AnalysisMode = "quick" | "deep" | "security" | "change-set";

export interface RuntimeIdentity {
  api_contract_version: number;
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

export type ArchitectureSupportTier = "exact" | "inferred" | "partial" | "unsupported";
export type ArchitectureCompleteness = "complete" | "partial" | "unknown" | "unsupported";
export type SemanticComponentKind =
  | "service"
  | "datastore"
  | "external_system"
  | "library"
  | "queue"
  | "unknown";
export type SemanticRelationKind =
  | "request"
  | "event"
  | "data_access"
  | "dependency"
  | "call"
  | "unknown";

export interface SemanticGitIdentity {
  ref: string | null;
  head_commit: string | null;
  dirty: boolean;
  parent_snapshot_id: string | null;
}

export interface SemanticCompleteness {
  status: ArchitectureCompleteness;
  extractor_version: string | null;
}

export interface SemanticSummary {
  snapshot_id: string;
  display_name: string;
  git: SemanticGitIdentity;
  created_at: string;
  counts: {
    services: number;
    datastores: number;
    external_systems: number;
    queues: number;
    libraries: number;
    unknown: number;
    boundaries: number;
  };
  totals: {
    files: number;
    lines: number;
    languages: Record<string, number>;
  };
  completeness: SemanticCompleteness;
}

export interface SemanticFindingLink {
  finding_id: string;
  title: string;
  severity: CanonicalFinding["severity"];
  confidence: number;
  component_id: string;
}

export interface SemanticComponent {
  component_id: string;
  snapshot_id: string;
  stable_key: string;
  name: string;
  component_kind: SemanticComponentKind;
  support_tier: ArchitectureSupportTier;
  completeness: ArchitectureCompleteness;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface SemanticBoundary {
  boundary_id: string;
  snapshot_id: string;
  stable_key: string;
  name: string;
  boundary_kind: string;
  support_tier: ArchitectureSupportTier;
  completeness: ArchitectureCompleteness;
  confidence: number;
  component_ids: string[];
  metadata: Record<string, unknown>;
}

export interface SemanticRelation {
  relation_id: string;
  snapshot_id?: string;
  source_component_id: string;
  target_component_id: string;
  stable_key: string;
  relation_kind: SemanticRelationKind;
  transport: string | null;
  is_async: boolean;
  support_tier: ArchitectureSupportTier;
  completeness: ArchitectureCompleteness;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface SemanticMembership {
  membership_id: string;
  snapshot_id: string;
  component_id: string;
  file_id: string | null;
  symbol_id: string | null;
  membership_kind: string;
  confidence: number;
  provenance: Record<string, unknown>;
  relative_path?: string | null;
  language?: string | null;
  line_count?: number | null;
  owners?: string[];
}

export interface SemanticResource {
  resource_id: string;
  snapshot_id: string;
  component_id: string | null;
  stable_key: string;
  resource_kind: string;
  name: string;
  locator: string | null;
  support_tier: ArchitectureSupportTier;
  completeness: ArchitectureCompleteness;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface SemanticEndpoint {
  endpoint_id: string;
  snapshot_id: string;
  component_id: string | null;
  stable_key: string;
  protocol: string;
  method: string | null;
  route: string;
  direction: "inbound" | "outbound";
  support_tier: ArchitectureSupportTier;
  completeness: ArchitectureCompleteness;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface SemanticProvenance {
  provenance_id: string;
  snapshot_id?: string;
  entity_kind: "component" | "boundary" | "membership" | "resource" | "endpoint" | "relation";
  entity_id: string;
  file_id: string;
  relative_path: string | null;
  start_line: number;
  end_line: number;
  derivation: string;
  extractor_version: string;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface SemanticArchitectureGraph extends SemanticSummary {
  components: SemanticComponent[];
  boundaries: SemanticBoundary[];
  relations: SemanticRelation[];
  findings: SemanticFindingLink[];
  truncated: boolean;
  findings_truncated: boolean;
  limits: { depth: number; node_limit: number; finding_limit: number };
}

export interface SemanticComponentDetail extends SemanticComponent {
  memberships: SemanticMembership[];
  resources: SemanticResource[];
  endpoints: SemanticEndpoint[];
  provenance: SemanticProvenance[];
  findings: SemanticFindingLink[];
  annotation: ComponentAnnotation | null;
  evidence_truncated: boolean;
  limits: { detail_row_limit: number };
}

export interface ComponentAnnotation {
  annotation_id: string;
  snapshot_id: string;
  component_id: string;
  note: string;
  version: number;
  actor: string;
  created_at: string;
  updated_at: string;
}

export interface SemanticTrace {
  snapshot_id: string;
  status: "complete" | "no_path" | "truncated" | "unsupported";
  components: SemanticComponent[];
  relations: SemanticRelation[];
  endpoints: SemanticEndpoint[];
  resources: SemanticResource[];
  provenance: SemanticProvenance[];
  evidence_truncated: boolean;
  max_hops: number;
}

export interface HistoryRun {
  run_id: string;
  status: AnalysisStatus;
  mode: AnalysisMode;
  current_stage: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  snapshot_id: string | null;
  display_name: string | null;
  base_commit: string | null;
  head_commit: string | null;
  dirty: boolean | null;
  task_count: number;
  specialist_count: number;
  finding_count: number;
  duration_seconds: number | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

export interface HistoryPage {
  items: HistoryRun[];
  next_cursor: string | null;
}

export interface RunComparison {
  baseline_run_id: string;
  target_run_id: string;
  new: CanonicalFinding[];
  resolved: CanonicalFinding[];
  unchanged: CanonicalFinding[];
  reopened: CanonicalFinding[];
  severity_moved: Array<{
    fingerprint: string;
    title: string;
    from_severity: CanonicalFinding["severity"];
    to_severity: CanonicalFinding["severity"];
  }>;
}

export interface HistoryTrends {
  days: number;
  partial: boolean;
  buckets: Array<{
    date: string;
    review_count: number;
    finding_count: number;
    cost_usd: number;
    average_duration_seconds: number | null;
  }>;
}

export interface LocalSettings {
  default_mode: AnalysisMode;
  default_max_agents: number;
  default_max_waves: number;
  default_max_tasks: number;
  default_max_total_tokens: number;
  default_max_cost_usd: number;
  default_max_elapsed_seconds: number;
  evidence_excerpt_enabled: boolean;
  retention_days: number;
}

export interface SettingsResponse {
  settings: LocalSettings;
  version: number;
  updated_at: string | null;
}

export type PresetRequest = Omit<AnalysisRequest, "project_path">;

export interface ReviewPreset {
  preset_id: string;
  name: string;
  request: PresetRequest;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ProviderTestResponse {
  configured: boolean;
  reachable: boolean;
  status: string;
  latency_ms: number | null;
}
