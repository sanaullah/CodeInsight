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
});
