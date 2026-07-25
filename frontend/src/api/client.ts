import type {
  AnalysisAccepted,
  AnalysisIntelligence,
  AnalysisRequest,
  AnalysisRun,
  ArchitectureGraph,
  ArchitectureTrace,
  CapabilitiesResponse,
  FindingDetail,
  FindingPage,
  FindingReviewState,
  HealthResponse,
  RecoveryResponse,
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
};
