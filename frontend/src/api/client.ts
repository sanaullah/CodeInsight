import type {
  AnalysisAccepted,
  AnalysisIntelligence,
  AnalysisRequest,
  AnalysisRun,
  ArchitectureGraph,
  ArchitectureTrace,
  CapabilitiesResponse,
  ComponentAnnotation,
  FindingDetail,
  FindingPage,
  FindingReviewState,
  HealthResponse,
  HistoryPage,
  HistoryTrends,
  LocalSettings,
  PresetRequest,
  ProviderTestResponse,
  RecoveryResponse,
  ReviewPreset,
  RunComparison,
  SemanticArchitectureGraph,
  SemanticComponentDetail,
  SemanticSummary,
  SemanticTrace,
  SettingsResponse,
  SnapshotSummary,
} from "./contracts";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Preserve the status message when the body is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const apiClient = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/v1/health", { signal }),
  capabilities: (signal?: AbortSignal) =>
    request<CapabilitiesResponse>("/api/v1/capabilities", { signal }),
  listRuns: (limit = 20, signal?: AbortSignal) =>
    request<AnalysisRun[]>(`/api/v1/analyses?limit=${limit}`, { signal }),
  getRun: (runId: string, signal?: AbortSignal) =>
    request<AnalysisRun>(`/api/v1/analyses/${encodeURIComponent(runId)}`, { signal }),
  getIntelligence: (runId: string, signal?: AbortSignal) =>
    request<AnalysisIntelligence>(`/api/v1/analyses/${encodeURIComponent(runId)}/intelligence`, {
      signal,
    }),
  submit: (payload: AnalysisRequest) =>
    request<AnalysisAccepted>("/api/v1/analyses", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  rerun: (runId: string) =>
    request<AnalysisAccepted>(`/api/v1/analyses/${encodeURIComponent(runId)}/rerun`, {
      method: "POST",
    }),
  cancel: (runId: string) =>
    request<AnalysisRun>(`/api/v1/analyses/${encodeURIComponent(runId)}`, {
      method: "DELETE",
    }),
  recover: () =>
    request<RecoveryResponse>("/api/v1/recovery", {
      method: "POST",
    }),
  findings: (params: URLSearchParams, signal?: AbortSignal) =>
    request<FindingPage>(`/api/v1/findings?${params.toString()}`, { signal }),
  finding: (findingId: string, signal?: AbortSignal) =>
    request<FindingDetail>(`/api/v1/findings/${encodeURIComponent(findingId)}`, { signal }),
  updateFinding: (
    findingId: string,
    payload: { review_state: FindingReviewState; note?: string | null; expected_version: number },
  ) =>
    request<FindingDetail>(`/api/v1/findings/${encodeURIComponent(findingId)}/review`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  snapshots: (signal?: AbortSignal) =>
    request<SnapshotSummary[]>("/api/v1/snapshots?limit=50", { signal }),
  architecture: (snapshotId: string, params: URLSearchParams, signal?: AbortSignal) =>
    request<ArchitectureGraph>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/architecture?${params}`,
      { signal },
    ),
  trace: (snapshotId: string, sourceId: string, targetId: string, signal?: AbortSignal) =>
    request<ArchitectureTrace>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/trace?source_id=${encodeURIComponent(sourceId)}&target_id=${encodeURIComponent(targetId)}`,
      { signal },
    ),
  semanticSummary: (snapshotId: string, signal?: AbortSignal) =>
    request<SemanticSummary>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/semantic-summary`,
      { signal },
    ),
  semanticArchitecture: (snapshotId: string, params: URLSearchParams, signal?: AbortSignal) =>
    request<SemanticArchitectureGraph>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/semantic-architecture?${params}`,
      { signal },
    ),
  semanticComponent: (snapshotId: string, componentId: string, signal?: AbortSignal) =>
    request<SemanticComponentDetail>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/semantic-components/${encodeURIComponent(componentId)}`,
      { signal },
    ),
  updateComponentAnnotation: (
    snapshotId: string,
    componentId: string,
    payload: { note: string; expected_version: number },
  ) =>
    request<ComponentAnnotation>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/semantic-components/${encodeURIComponent(componentId)}/annotation`,
      { method: "PUT", body: JSON.stringify(payload) },
    ),
  semanticTrace: (
    snapshotId: string,
    sourceId: string,
    targetId: string,
    maxHops = 8,
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams({
      source_id: sourceId,
      target_id: targetId,
      max_hops: String(maxHops),
    });
    return request<SemanticTrace>(
      `/api/v1/snapshots/${encodeURIComponent(snapshotId)}/semantic-trace?${params}`,
      { signal },
    );
  },
  history: (params: URLSearchParams, signal?: AbortSignal) =>
    request<HistoryPage>(`/api/v1/history?${params}`, { signal }),
  compareRuns: (baselineRunId: string, targetRunId: string, signal?: AbortSignal) =>
    request<RunComparison>(
      `/api/v1/history/compare?baseline_run_id=${encodeURIComponent(baselineRunId)}&target_run_id=${encodeURIComponent(targetRunId)}`,
      { signal },
    ),
  historyTrends: (days = 30, signal?: AbortSignal) =>
    request<HistoryTrends>(`/api/v1/history/trends?days=${days}`, { signal }),
  settings: (signal?: AbortSignal) => request<SettingsResponse>("/api/v1/settings", { signal }),
  updateSettings: (settings: LocalSettings, expectedVersion: number) =>
    request<SettingsResponse>("/api/v1/settings", {
      method: "PUT",
      body: JSON.stringify({ settings, expected_version: expectedVersion }),
    }),
  testProvider: () =>
    request<ProviderTestResponse>("/api/v1/settings/provider-test", { method: "POST" }),
  presets: (signal?: AbortSignal) => request<ReviewPreset[]>("/api/v1/presets", { signal }),
  createPreset: (name: string, presetRequest: PresetRequest) =>
    request<ReviewPreset>("/api/v1/presets", {
      method: "POST",
      body: JSON.stringify({ name, request: presetRequest }),
    }),
  updatePreset: (preset: ReviewPreset) =>
    request<ReviewPreset>(`/api/v1/presets/${encodeURIComponent(preset.preset_id)}`, {
      method: "PUT",
      body: JSON.stringify({
        name: preset.name,
        request: preset.request,
        expected_version: preset.version,
      }),
    }),
  deletePreset: (presetId: string, expectedVersion: number) =>
    fetch(`/api/v1/presets/${encodeURIComponent(presetId)}?expected_version=${expectedVersion}`, {
      method: "DELETE",
    }).then((response) => {
      if (!response.ok)
        throw new ApiError(`Unable to delete preset (${response.status})`, response.status);
    }),
};
