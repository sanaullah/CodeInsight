import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiClient } from "./client";
import { healthFixture } from "./fixtures";

afterEach(() => vi.restoreAllMocks());

function response(body: unknown, status = 200): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: { "Content-Type": typeof body === "string" ? "text/plain" : "application/json" },
  });
}

describe("apiClient", () => {
  it("loads the typed health contract", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(response(healthFixture));

    await expect(apiClient.health()).resolves.toEqual(healthFixture);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/health",
      expect.objectContaining({ headers: { Accept: "application/json" } }),
    );
  });

  it("encodes run identifiers and forwards cancellation", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(response({ run_id: "run/one", status: "cancelled" }));

    await apiClient.cancel("run/one");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/analyses/run%2Fone",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("maps list, intelligence, and recovery endpoints", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response([]));

    await apiClient.listRuns(7);
    await apiClient.getRun("run 1");
    await apiClient.getIntelligence("run 1");
    await apiClient.recover();

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/analyses?limit=7",
      "/api/v1/analyses/run%201",
      "/api/v1/analyses/run%201/intelligence",
      "/api/v1/recovery",
    ]);
  });

  it("submits JSON with the exact durable request", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(response({ run_id: "run-1", status: "queued", status_url: "/run-1" }));
    const payload = {
      project_path: "H:\\repo",
      max_agents: 4,
      mode: "deep" as const,
      max_waves: 2,
      max_tasks: 100,
      max_total_tokens: 1000,
      max_cost_usd: 2,
      max_elapsed_seconds: 60,
    };

    await apiClient.submit(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/analyses",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(payload),
        headers: { Accept: "application/json", "Content-Type": "application/json" },
      }),
    );
  });

  it("surfaces server detail and preserves status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response({ detail: "Budget rejected" }, 422));

    const error = await apiClient.recover().catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message: "Budget rejected", status: 422 });
  });

  it("uses a stable fallback for non-JSON failures", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response("offline", 503));

    await expect(apiClient.health()).rejects.toThrow("Request failed (503)");
  });

  it("maps durable settings, provider checks, and preset lifecycle contracts", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response({}));
    const settings = {
      default_mode: "deep" as const,
      default_max_agents: 4,
      default_max_waves: 2,
      default_max_tasks: 100,
      default_max_total_tokens: 1_000_000,
      default_max_cost_usd: 25,
      default_max_elapsed_seconds: 3_600,
      evidence_excerpt_enabled: true,
      retention_days: 90,
    };
    const preset = {
      preset_id: "preset/one",
      name: "Deep",
      request: {
        goal: null,
        model_name: null,
        file_extensions: null,
        selected_directories: null,
        mode: "deep" as const,
        max_agents: 4,
        max_waves: 2,
        max_tasks: 100,
        max_total_tokens: 1_000_000,
        max_cost_usd: 25,
        max_elapsed_seconds: 3_600,
      },
      version: 2,
      created_at: "now",
      updated_at: "now",
    };

    await apiClient.settings();
    await apiClient.updateSettings(settings, 3);
    await apiClient.testProvider();
    await apiClient.presets();
    await apiClient.createPreset("Deep", preset.request);
    await apiClient.updatePreset(preset);
    await apiClient.deletePreset("preset/one", 2);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/settings",
      "/api/v1/settings",
      "/api/v1/settings/provider-test",
      "/api/v1/presets",
      "/api/v1/presets",
      "/api/v1/presets/preset%2Fone",
      "/api/v1/presets/preset%2Fone?expected_version=2",
    ]);
    expect(fetchMock.mock.calls[1][1]).toEqual(
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ settings, expected_version: 3 }),
      }),
    );
  });

  it("maps bounded semantic architecture contracts with encoded identifiers", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => response({}));
    const filters = new URLSearchParams([
      ["component_kind", "service"],
      ["relation_kind", "data_access"],
      ["focus", "service:orders"],
      ["depth", "2"],
      ["limit", "120"],
    ]);

    await apiClient.semanticSummary("snapshot/one");
    await apiClient.semanticArchitecture("snapshot/one", filters);
    await apiClient.semanticComponent("snapshot/one", "service/orders");
    await apiClient.semanticTrace("snapshot/one", "service/orders", "store/main", 5);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/v1/snapshots/snapshot%2Fone/semantic-summary",
      `/api/v1/snapshots/snapshot%2Fone/semantic-architecture?${filters}`,
      "/api/v1/snapshots/snapshot%2Fone/semantic-components/service%2Forders",
      "/api/v1/snapshots/snapshot%2Fone/semantic-trace?source_id=service%2Forders&target_id=store%2Fmain&max_hops=5",
    ]);
  });

  it("reports preset deletion conflicts", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(response({}, 409));
    await expect(apiClient.deletePreset("stale", 1)).rejects.toMatchObject({
      message: "Unable to delete preset (409)",
      status: 409,
    });
  });
});
