import { describe, expect, it } from "vitest";
import { capabilitiesFixture, healthFixture, runStateFixtures } from "./fixtures";

describe("frontend contract fixtures", () => {
  it("covers every terminal lifecycle and the queued/running boundaries", () => {
    expect(new Set(Object.values(runStateFixtures).map((run) => run.status))).toEqual(
      new Set(["queued", "running", "succeeded", "needs_attention", "failed", "cancelled"]),
    );
  });

  it("keeps operational fixture claims explicit and non-reachable", () => {
    expect(healthFixture.runtime).toMatchObject({
      application_server: "FastAPI",
      environment_manager: "uv",
      database_engine: "SQLite",
      database_journal_mode: "WAL",
      read_only_analysis: true,
    });
    expect(capabilitiesFixture.model_provider_configured).toBe(false);
    expect(capabilitiesFixture.langfuse_enabled).toBe(false);
  });
});
