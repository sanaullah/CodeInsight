import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { apiClient } from "./api/client";
import {
  capabilitiesFixture,
  healthFixture,
  runningIntelligenceFixture,
  runningRunFixture,
} from "./api/fixtures";
import { CommandCenterPage } from "./pages/CommandCenterPage";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(apiClient, "health").mockResolvedValue(healthFixture);
  vi.spyOn(apiClient, "capabilities").mockResolvedValue(capabilitiesFixture);
  vi.spyOn(apiClient, "listRuns").mockResolvedValue([]);
  vi.spyOn(apiClient, "recover").mockResolvedValue({
    recovered_runs: 0,
    recovered_tasks: 0,
    scheduled_runs: 0,
  });
  vi.spyOn(apiClient, "getRun").mockResolvedValue(runningRunFixture);
  vi.spyOn(apiClient, "getIntelligence").mockResolvedValue(runningIntelligenceFixture);
  vi.spyOn(apiClient, "findings").mockResolvedValue({
    items: [],
    next_cursor: null,
    counts_by_severity: {},
  });
  vi.spyOn(apiClient, "finding").mockRejectedValue(new Error("Finding not selected"));
  vi.spyOn(apiClient, "updateFinding").mockRejectedValue(new Error("Finding not selected"));
  vi.spyOn(apiClient, "snapshots").mockResolvedValue([]);
  vi.spyOn(apiClient, "architecture").mockRejectedValue(new Error("Snapshot not selected"));
  vi.spyOn(apiClient, "trace").mockRejectedValue(new Error("Trace not selected"));
  vi.spyOn(apiClient, "semanticSummary").mockRejectedValue(new Error("Snapshot not selected"));
  vi.spyOn(apiClient, "semanticArchitecture").mockRejectedValue(new Error("Snapshot not selected"));
  vi.spyOn(apiClient, "semanticComponent").mockRejectedValue(new Error("Component not selected"));
  vi.spyOn(apiClient, "semanticTrace").mockRejectedValue(new Error("Trace not selected"));
  vi.spyOn(apiClient, "history").mockResolvedValue({ items: [], next_cursor: null });
  vi.spyOn(apiClient, "historyTrends").mockResolvedValue({
    days: 30,
    partial: true,
    buckets: [],
  });
  vi.spyOn(apiClient, "compareRuns").mockRejectedValue(new Error("Runs not selected"));
  vi.spyOn(apiClient, "settings").mockResolvedValue({
    settings: {
      default_mode: "deep",
      default_max_agents: 4,
      default_max_waves: 2,
      default_max_tasks: 100,
      default_max_total_tokens: 1_000_000,
      default_max_cost_usd: 25,
      default_max_elapsed_seconds: 3_600,
      evidence_excerpt_enabled: true,
      retention_days: 90,
    },
    version: 0,
    updated_at: null,
  });
  vi.spyOn(apiClient, "presets").mockResolvedValue([]);
  vi.spyOn(apiClient, "updateSettings").mockRejectedValue(new Error("No settings update"));
  vi.spyOn(apiClient, "testProvider").mockResolvedValue({
    configured: false,
    reachable: false,
    status: "Provider endpoint is not configured.",
    latency_ms: null,
  });
  vi.spyOn(apiClient, "createPreset").mockRejectedValue(new Error("No preset create"));
  vi.spyOn(apiClient, "updatePreset").mockRejectedValue(new Error("No preset update"));
  vi.spyOn(apiClient, "deletePreset").mockRejectedValue(new Error("No preset delete"));
});

describe("application shell", () => {
  it("renders truthful runtime identity and future routes without invented controls", async () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /repository intelligence/i })).toBeInTheDocument();
    expect(screen.getByText("Read-only by design")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "API documentation" })).toHaveAttribute(
      "href",
      "/api/docs",
    );
    await waitFor(() => expect(screen.getByText("API ready")).toBeInTheDocument());
    expect(screen.getByText("v0.1.0-alpha")).toBeInTheDocument();
    expect(screen.queryByText(/checkout branch/i)).not.toBeInTheDocument();
  });

  it("invokes durable recovery and reports exact ledger outcomes", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.recover).mockResolvedValue({
      recovered_runs: 1,
      recovered_tasks: 3,
      scheduled_runs: 1,
    });
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Recover interrupted work" }));
    expect(
      await screen.findByText("Recovered 1 run(s) and 3 task(s); scheduled 1."),
    ).toBeInTheDocument();
  });

  it("reports durable recovery failures without hiding recent reviews", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.recover).mockRejectedValue(new Error("Recovery is busy"));
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Recover interrupted work" }));
    expect(await screen.findByText("Recovery is busy")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Recent reviews" })).toBeInTheDocument();
  });

  it("navigates client-side and submits only supported review fields", async () => {
    const user = userEvent.setup();
    const submit = vi
      .spyOn(apiClient, "submit")
      .mockResolvedValue({ run_id: "run-new", status: "queued", status_url: "/run-new" });
    render(<App />);

    await user.click(screen.getByRole("link", { name: /new review/i }));
    await user.type(screen.getByLabelText("Repository path"), "H:\\repo");
    await user.type(screen.getByLabelText("Review goal"), "Inspect trust boundaries");
    await user.click(screen.getByLabelText(/Security/));
    await user.click(screen.getByText("Scope and durable budgets"));
    await user.type(screen.getByLabelText("File extensions"), ".py, ts");
    await user.click(screen.getByRole("button", { name: "Start code review" }));

    await waitFor(() => expect(submit).toHaveBeenCalledOnce());
    expect(submit).toHaveBeenCalledWith(
      expect.objectContaining({
        project_path: "H:\\repo",
        goal: "Inspect trust boundaries",
        mode: "security",
        file_extensions: [".py", "ts"],
      }),
    );
    expect(window.location.pathname).toBe("/reviews/run-new");
  });

  it("applies a path-free durable preset to a new review", async () => {
    window.history.replaceState({}, "", "/new-review");
    const user = userEvent.setup();
    vi.mocked(apiClient.presets).mockResolvedValue([
      {
        preset_id: "preset-security",
        name: "Security triage",
        request: {
          goal: "Inspect authorization",
          model_name: null,
          file_extensions: [".py"],
          selected_directories: ["src"],
          mode: "security",
          max_agents: 6,
          max_waves: 3,
          max_tasks: 80,
          max_total_tokens: 200_000,
          max_cost_usd: 10,
          max_elapsed_seconds: 900,
        },
        version: 1,
        created_at: "2026-07-25T12:00:00Z",
        updated_at: "2026-07-25T12:00:00Z",
      },
    ]);
    render(<App />);

    await user.selectOptions(await screen.findByLabelText("Review preset"), "preset-security");
    expect(screen.getByLabelText("Review goal")).toHaveValue("Inspect authorization");
    expect(screen.getByLabelText("Maximum specialists")).toHaveValue("6");
    expect(screen.getByLabelText("File extensions")).toHaveValue(".py");
    expect(screen.getByLabelText(/Security/)).toBeChecked();
  });

  it("shows durable command-center intelligence and cancellation", async () => {
    window.history.replaceState({}, "", "/reviews/run-fixture-1");
    const user = userEvent.setup();
    const cancel = vi
      .spyOn(apiClient, "cancel")
      .mockResolvedValue({ ...runningRunFixture, status: "cancelled" });
    render(<App />);

    expect(await screen.findByText("Repository Architecture Specialist")).toBeInTheDocument();
    expect(screen.getByText("124")).toBeInTheDocument();
    expect(screen.getByText("600 tokens")).toBeInTheDocument();
    expect(screen.getByText("Verification in progress")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel review" }));
    await waitFor(() => expect(cancel).toHaveBeenCalledWith("run-fixture-1"));
  });

  it("has no serious or critical axe violations on the operational overview", async () => {
    const { container } = render(<App />);
    await screen.findByText("API ready");

    const result = await axe.run(container, {
      rules: {
        "color-contrast": { enabled: false },
      },
    });
    const blocking = result.violations.filter(
      (violation) => violation.impact === "serious" || violation.impact === "critical",
    );
    expect(blocking).toEqual([]);
  });

  it.each([
    ["/new-review", "Start a new review"],
    ["/findings", "Findings"],
    ["/architecture", "Architecture explorer"],
    ["/history", "Review history"],
    ["/settings", "Settings"],
  ])("has no serious or critical axe violations on %s", async (path, heading) => {
    window.history.replaceState({}, "", path);
    const view = render(<App />);
    await screen.findByRole("heading", { name: heading });
    const result = await axe.run(view.container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(
      result.violations.filter(
        (violation) => violation.impact === "serious" || violation.impact === "critical",
      ),
    ).toEqual([]);
    view.unmount();
  });

  it.each([
    ["/", "Repository intelligence you can inspect."],
    ["/new-review", "Start a new review"],
    ["/settings", "Settings"],
    ["/history", "Review history"],
    ["/findings", "Findings"],
    ["/architecture", "Architecture explorer"],
  ])("keeps %s inside the shared content column", async (path, heading) => {
    window.history.replaceState({}, "", path);
    const view = render(<App />);
    const pageHeading = await screen.findByRole("heading", { name: heading });

    expect(pageHeading.closest(".page")).not.toBeNull();
    expect(pageHeading.closest("main")).not.toBeNull();
    view.unmount();
  });

  it.each([
    ["/history", "Review history"],
    ["/findings", "Findings"],
    ["/architecture", "Architecture explorer"],
  ])("lets the dense %s workspace use the available application width", async (path, heading) => {
    window.history.replaceState({}, "", path);
    const view = render(<App />);
    const pageHeading = await screen.findByRole("heading", { name: heading });

    expect(pageHeading.closest(".page")).toHaveClass("dense-workspace");
    view.unmount();
  });

  it("explains an API contract mismatch without hiding runtime diagnostics", async () => {
    vi.mocked(apiClient.health).mockResolvedValue({
      ...healthFixture,
      runtime: { ...healthFixture.runtime, api_contract_version: 2 },
    });
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("API update required")).toBeInTheDocument();
    await user.click(screen.getByText("Runtime details"));
    expect(screen.getByRole("alert")).toHaveTextContent("This frontend supports API contract 1");
  });

  it("provides an accessible mobile menu escape path", async () => {
    const user = userEvent.setup();
    render(<App />);
    const menu = screen.getByRole("button", { name: "Open navigation" });
    await user.click(menu);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(menu).toHaveAttribute("aria-expanded", "false");
  });

  it("renders a truthful missing route", async () => {
    window.history.replaceState({}, "", "/missing");
    render(<App />);
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByText(/not yet implemented/i)).toBeInTheDocument();
  });

  it("saves durable settings, tests the provider, and manages a preset", async () => {
    window.history.replaceState({}, "", "/settings");
    const user = userEvent.setup();
    const updatedSettings = {
      settings: {
        default_mode: "security" as const,
        default_max_agents: 4,
        default_max_waves: 2,
        default_max_tasks: 100,
        default_max_total_tokens: 1_000_000,
        default_max_cost_usd: 25,
        default_max_elapsed_seconds: 3_600,
        evidence_excerpt_enabled: true,
        retention_days: 90,
      },
      version: 1,
      updated_at: "2026-07-25T12:00:00Z",
    };
    vi.mocked(apiClient.updateSettings).mockResolvedValue(updatedSettings);
    const createdPreset = {
      preset_id: "preset-1",
      name: "Security defaults",
      request: {
        goal: null,
        model_name: null,
        file_extensions: null,
        selected_directories: null,
        mode: "security" as const,
        max_agents: 4,
        max_waves: 2,
        max_tasks: 100,
        max_total_tokens: 1_000_000,
        max_cost_usd: 25,
        max_elapsed_seconds: 3_600,
      },
      version: 1,
      created_at: "2026-07-25T12:00:00Z",
      updated_at: "2026-07-25T12:00:00Z",
    };
    vi.mocked(apiClient.createPreset).mockResolvedValue(createdPreset);
    vi.mocked(apiClient.deletePreset).mockResolvedValue();
    render(<App />);

    expect(await screen.findByText("Analysis guardrails")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Default mode"), "security");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save defaults" }));
    await waitFor(() =>
      expect(apiClient.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ default_mode: "security" }),
        0,
      ),
    );
    await user.click(screen.getByRole("button", { name: "Test provider" }));
    expect(await screen.findByText("Provider endpoint is not configured.")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Preset name"), "Security defaults");
    await user.click(screen.getByRole("button", { name: "Save current defaults" }));
    expect(await screen.findByText("Security defaults")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(apiClient.deletePreset).toHaveBeenCalledWith("preset-1", 1));
  });

  it("filters durable history and compares two runs", async () => {
    window.history.replaceState({}, "", "/history");
    const user = userEvent.setup();
    const historyRuns = ["run-001", "run-002"].map((runId, index) => ({
      run_id: runId,
      status: "succeeded" as const,
      mode: index ? ("security" as const) : ("deep" as const),
      current_stage: "complete",
      created_at: `2026-07-2${4 + index}T12:00:00Z`,
      started_at: `2026-07-2${4 + index}T12:00:00Z`,
      completed_at: `2026-07-2${4 + index}T12:01:00Z`,
      snapshot_id: "snapshot-1",
      display_name: "Fixture repository",
      base_commit: "base",
      head_commit: "head",
      dirty: false,
      task_count: 2,
      specialist_count: 2,
      finding_count: index + 1,
      duration_seconds: 60,
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cost_usd: 0.01,
    }));
    vi.mocked(apiClient.history)
      .mockResolvedValue({ items: historyRuns, next_cursor: null })
      .mockResolvedValueOnce({ items: historyRuns, next_cursor: "history-cursor" })
      .mockResolvedValueOnce({
        items: [{ ...historyRuns[0], run_id: "run-003", display_name: "Next page" }],
        next_cursor: null,
      });
    vi.mocked(apiClient.historyTrends).mockResolvedValue({
      days: 30,
      partial: false,
      buckets: [
        {
          date: "2026-07-25",
          review_count: 2,
          finding_count: 3,
          cost_usd: 0.02,
          average_duration_seconds: 60,
        },
      ],
    });
    vi.mocked(apiClient.compareRuns).mockResolvedValue({
      baseline_run_id: "run-001",
      target_run_id: "run-002",
      new: [],
      resolved: [],
      unchanged: [],
      reopened: [],
      severity_moved: [],
    });

    render(<App />);
    expect(await screen.findByText("30-day measured trend")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load more reviews" }));
    expect(await screen.findByText("Next page")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Compare run-001"));
    await user.click(screen.getByLabelText("Compare run-002"));
    await user.click(screen.getByRole("button", { name: "Compare selected (2/2)" }));
    expect(await screen.findByText("Run comparison")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Mode"), "security");
    await waitFor(() => {
      const params = vi.mocked(apiClient.history).mock.calls.at(-1)?.[0];
      expect(params?.get("mode")).toBe("security");
    });
  });

  it("explores semantic architecture, component evidence, and a typed trace", async () => {
    window.history.replaceState({}, "", "/architecture");
    const user = userEvent.setup();
    vi.mocked(apiClient.snapshots).mockResolvedValue([
      {
        snapshot_id: "snapshot-1",
        project_id: "project-1",
        display_name: "Fixture",
        git_repository: "git",
        base_commit: "base",
        head_commit: "head",
        dirty: false,
        metadata: {},
        created_at: "2026-07-25T12:00:00Z",
        file_count: 2,
        symbol_count: 0,
        edge_count: 1,
      },
    ]);
    const components = [
      {
        component_id: "service-orders",
        snapshot_id: "snapshot-1",
        stable_key: "service:orders",
        name: "Orders service",
        component_kind: "service" as const,
        support_tier: "exact" as const,
        completeness: "complete" as const,
        confidence: 1,
        metadata: { framework: "FastAPI" },
      },
      {
        component_id: "store-orders",
        snapshot_id: "snapshot-1",
        stable_key: "datastore:orders",
        name: "Orders database",
        component_kind: "datastore" as const,
        support_tier: "inferred" as const,
        completeness: "partial" as const,
        confidence: 0.9,
        metadata: {},
      },
    ];
    const relation = {
      relation_id: "relation-1",
      source_component_id: "service-orders",
      target_component_id: "store-orders",
      stable_key: "data:orders",
      relation_kind: "data_access" as const,
      transport: "sqlite",
      is_async: false,
      support_tier: "exact" as const,
      completeness: "complete" as const,
      confidence: 0.98,
      metadata: {},
    };
    vi.mocked(apiClient.semanticArchitecture).mockResolvedValue({
      snapshot_id: "snapshot-1",
      display_name: "Fixture",
      git: { ref: "main", head_commit: "head", dirty: false, parent_snapshot_id: null },
      created_at: "2026-07-25T12:00:00Z",
      counts: {
        services: 1,
        datastores: 1,
        external_systems: 0,
        queues: 0,
        libraries: 0,
        unknown: 0,
        boundaries: 1,
      },
      totals: { files: 2, lines: 30, languages: { Python: 2 } },
      completeness: { status: "partial", extractor_version: "semantic-v1" },
      components,
      boundaries: [],
      relations: [relation],
      findings: [
        {
          finding_id: "finding-architecture",
          title: "Unsafe order query",
          severity: "high",
          confidence: 0.94,
          component_id: "service-orders",
        },
      ],
      truncated: false,
      findings_truncated: false,
      limits: { depth: 1, node_limit: 160, finding_limit: 500 },
    });
    vi.mocked(apiClient.semanticComponent).mockResolvedValue({
      ...components[0],
      memberships: [
        {
          membership_id: "membership-1",
          snapshot_id: "snapshot-1",
          component_id: "service-orders",
          file_id: "file-1",
          symbol_id: null,
          membership_kind: "implementation",
          confidence: 1,
          provenance: {},
          relative_path: "app/orders.py",
          language: "Python",
          line_count: 30,
          owners: [],
        },
      ],
      resources: [],
      endpoints: [],
      provenance: [],
      findings: [
        {
          finding_id: "finding-architecture",
          title: "Unsafe order query",
          severity: "high",
          confidence: 0.94,
          component_id: "service-orders",
        },
      ],
      evidence_truncated: false,
      limits: { detail_row_limit: 250 },
    });
    vi.mocked(apiClient.semanticTrace).mockResolvedValue({
      snapshot_id: "snapshot-1",
      status: "complete",
      components,
      relations: [relation],
      endpoints: [],
      resources: [],
      provenance: [],
      evidence_truncated: false,
      max_hops: 8,
    });
    vi.mocked(apiClient.finding).mockResolvedValue({
      finding_id: "finding-architecture",
      fingerprint: "unsafe-order-query",
      title: "Unsafe order query",
      claim: "The order query accepts an unverified identifier.",
      severity: "high",
      confidence: 0.94,
      candidate_ids: ["candidate-architecture"],
      supporting_evidence_ids: ["evidence-architecture"],
      conflicting_candidate_ids: [],
      recommendation: "Validate ownership before querying.",
      run_id: "run-architecture",
      review_state: "validated",
      review_note: null,
      review_version: 1,
      reviewed_at: "2026-07-25T12:10:00Z",
      candidates: [],
      evidence: [
        {
          evidence_id: "evidence-architecture",
          relative_path: "app/orders.py",
          content_hash: "abc123",
          start_line: 12,
          end_line: 14,
          evidence_kind: "source_span",
          provenance: { extractor: "semantic-v1" },
          excerpt: "find_order(user_id)",
          integrity: "valid",
          redacted: false,
        },
      ],
      review_history: [],
    });

    const view = render(<App />);
    expect(await screen.findByRole("heading", { name: "Semantic topology" })).toBeInTheDocument();
    expect(screen.getByText(/partial projection/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: /Orders service/i })[0]);
    expect(await screen.findByText("app/orders.py")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Unsafe order query/i }));
    expect(await screen.findByRole("dialog", { name: "Unsafe order query" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export finding" })).toHaveAttribute(
      "href",
      "/api/v1/findings/finding-architecture/export",
    );
    await user.keyboard("{Escape}");
    await user.selectOptions(screen.getByLabelText("Trace source"), "service-orders");
    await user.selectOptions(screen.getByLabelText("Trace target"), "store-orders");
    await user.click(screen.getByRole("button", { name: "Trace evidence" }));
    expect((await screen.findAllByText("data access")).length).toBeGreaterThan(0);
    expect(window.location.search).toContain("selected=service-orders");
    const accessibility = await axe.run(view.container, {
      rules: { "color-contrast": { enabled: false } },
    });
    expect(
      accessibility.violations.filter(
        (violation) => violation.impact === "serious" || violation.impact === "critical",
      ),
    ).toEqual([]);
  });

  it("filters findings through the typed API and records durable review state", async () => {
    window.history.replaceState({}, "", "/findings");
    const user = userEvent.setup();
    const finding = {
      finding_id: "finding-1",
      fingerprint: "authorization",
      title: "Authorization check is bypassed",
      claim: "The handler reaches a protected path without enforcing policy.",
      severity: "high" as const,
      confidence: 0.94,
      candidate_ids: ["candidate-1"],
      supporting_evidence_ids: ["evidence-1"],
      conflicting_candidate_ids: [],
      recommendation: "Enforce policy before dispatch.",
      run_id: "run-fixture-1",
      review_state: "new" as const,
      review_note: null,
      review_version: 0,
      reviewed_at: null,
    };
    const findings = vi
      .mocked(apiClient.findings)
      .mockResolvedValue({
        items: [finding],
        next_cursor: null,
        counts_by_severity: { high: 1 },
      })
      .mockResolvedValueOnce({
        items: [finding],
        next_cursor: "finding-cursor",
        counts_by_severity: { high: 2 },
      })
      .mockResolvedValueOnce({
        items: [{ ...finding, finding_id: "finding-2", title: "Second page finding" }],
        next_cursor: null,
        counts_by_severity: { high: 2 },
      });
    vi.mocked(apiClient.finding).mockResolvedValue({
      ...finding,
      candidates: [
        {
          relationship: "supporting",
          candidate: {
            candidate_id: "candidate-1",
            title: finding.title,
            claim: finding.claim,
            category: "security",
            affected_path: "src/auth.py",
            impact: "Unauthorized access",
            proposed_severity: "high",
            proposed_confidence: 0.94,
            recommendation: finding.recommendation,
          },
          role: {
            role_id: "role-1",
            wave_id: "wave-1",
            name: "Security specialist",
            mission: "Review authorization",
            rationale: "Security mode",
            coverage_targets: ["src/auth.py"],
            required_capabilities: [],
            model_policy: "balanced",
          },
          verdict: {
            disposition: "accepted",
            calibrated_confidence: 0.94,
            calibrated_severity: "high",
            rationale: "Direct source support.",
          },
        },
      ],
      evidence: [
        {
          evidence_id: "evidence-1",
          relative_path: "src/auth.py",
          content_hash: "abc123",
          start_line: 4,
          end_line: 5,
          evidence_kind: "source",
          provenance: { run_id: "run-fixture-1" },
          excerpt: "return allow",
          integrity: "valid",
          redacted: false,
        },
      ],
      review_history: [],
    });
    vi.mocked(apiClient.updateFinding).mockResolvedValue({
      ...(await apiClient.finding("finding-1")),
      review_state: "reviewed",
      review_version: 1,
    });

    render(<App />);
    expect(await screen.findByText(finding.title)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load more findings" }));
    expect(await screen.findByText("Second page finding")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/api/v1/findings-export?format=csv",
    );
    await user.type(screen.getByLabelText("Search"), "authorization");
    await waitFor(() =>
      expect(findings).toHaveBeenLastCalledWith(
        expect.objectContaining({}),
        expect.any(AbortSignal),
      ),
    );
    expect(window.location.search).toContain("search=authorization");
    await user.click(screen.getByRole("button", { name: /Authorization check is bypassed/ }));
    expect(await screen.findByText("return allow")).toBeInTheDocument();
    const clipboard = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
    await user.click(screen.getByRole("button", { name: "Copy verified excerpt" }));
    expect(clipboard).toHaveBeenCalledWith("return allow");
    expect(screen.getByText("Copied evidence from src/auth.py.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Reviewed" }));
    await waitFor(() =>
      expect(apiClient.updateFinding).toHaveBeenCalledWith("finding-1", {
        review_state: "reviewed",
        expected_version: 0,
      }),
    );
    await user.click(screen.getByLabelText(`Select ${finding.title}`));
    await user.click(screen.getByRole("button", { name: "Mark selected reviewed (1)" }));
    expect(screen.getByText("Mark 1 selected finding(s) as reviewed?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Confirm bulk review" }));
  });

  it("renders accepted findings, measured gaps, and synthesis from durable data", async () => {
    window.history.replaceState({}, "", "/reviews/run-fixture-1");
    vi.mocked(apiClient.getRun).mockResolvedValue({
      ...runningRunFixture,
      status: "succeeded",
      completed_at: "2026-07-25T12:02:05Z",
      result: {
        provider_mode: "model",
        synthesized_report: "Prioritize the verified authorization defect.",
      },
      current_stage: "complete",
    });
    vi.mocked(apiClient.getIntelligence).mockResolvedValue({
      ...runningIntelligenceFixture,
      status: "succeeded",
      current_stage: "complete",
      candidates: [
        {
          candidate: {
            candidate_id: "candidate-rejected",
            title: "Unverified concern",
            claim: "A claim without sufficient evidence.",
            category: "security",
            affected_path: "src/auth.py",
            impact: "Unknown",
            proposed_severity: "high",
            proposed_confidence: 0.4,
            recommendation: "Collect evidence.",
          },
          verdict: {
            disposition: "rejected",
            calibrated_confidence: 0.2,
            calibrated_severity: "info",
            rationale: "No supporting span.",
          },
        },
      ],
      findings: [
        {
          finding_id: "finding-1",
          fingerprint: "auth-defect",
          title: "Authorization check is bypassed",
          claim: "The handler reaches the write path before policy enforcement.",
          severity: "high",
          confidence: 0.94,
          candidate_ids: ["candidate-1"],
          supporting_evidence_ids: ["evidence-1"],
          conflicting_candidate_ids: [],
          recommendation: "Enforce policy before dispatch.",
        },
      ],
      coverage: [
        {
          assessment_id: "coverage-1",
          run_id: "run-fixture-1",
          wave_id: "wave-1",
          measured: { repository_files: 0.92 },
          exclusions: [],
          unsupported_areas: ["generated parser"],
          unresolved_uncertainty: [],
          remaining_gaps: ["Review deployment policy"],
          proposed_follow_up_task_ids: [],
          confidence: 0.9,
          follow_up_decision: "complete",
          decision_rationale: "Coverage budget satisfied.",
        },
      ],
      usage: { input_tokens: 1000, output_tokens: 250, total_tokens: 1250, cost_usd: 0.0123 },
    });

    render(<App />);

    expect(await screen.findByText("Authorization check is bypassed")).toBeInTheDocument();
    expect(screen.getByText("94% confidence · 1 verified evidence span(s)")).toBeInTheDocument();
    expect(screen.getByText("Review deployment policy")).toBeInTheDocument();
    expect(screen.getByText("Unsupported: generated parser")).toBeInTheDocument();
    expect(screen.getByText("Prioritize the verified authorization defect.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel review" })).not.toBeInTheDocument();
  });

  it("surfaces submit failures and re-enables the form", async () => {
    window.history.replaceState({}, "", "/new-review");
    const user = userEvent.setup();
    vi.spyOn(apiClient, "submit").mockRejectedValue(new Error("Repository is not accessible"));
    render(<App />);

    await user.type(screen.getByLabelText("Repository path"), "H:\\missing");
    await user.click(screen.getByRole("button", { name: "Start code review" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Repository is not accessible");
    expect(screen.getByRole("button", { name: "Start code review" })).toBeEnabled();
  });

  it("keeps current run status visible when intelligence is temporarily unavailable", async () => {
    vi.mocked(apiClient.getIntelligence).mockRejectedValue(new Error("Projection unavailable"));
    render(<CommandCenterPage runId="run-fixture-1" />);

    expect(
      await screen.findByRole("heading", { name: "Review command center" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Review command center" }).closest(".page"),
    ).toHaveClass("dense-workspace");
    expect(screen.getByText("124")).toBeInTheDocument();
    expect(
      screen.getByText(/run status is current; intelligence details are unavailable/i),
    ).toHaveTextContent("Projection unavailable");
  });

  it("surfaces cancellation failure without hiding the active run", async () => {
    window.history.replaceState({}, "", "/reviews/run-fixture-1");
    const user = userEvent.setup();
    vi.spyOn(apiClient, "cancel").mockRejectedValue(new Error("Cancellation request failed"));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Cancel review" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Cancellation request failed");
    expect(screen.getByText("Repository Architecture Specialist")).toBeInTheDocument();
  });

  it("makes API failure explicit and supports a recovery retry", async () => {
    vi.mocked(apiClient.health)
      .mockRejectedValueOnce(new Error("Local API is offline"))
      .mockResolvedValue(healthFixture);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("API unavailable")).toBeInTheDocument();
    await user.click(screen.getByText("Runtime details"));
    expect(screen.getByText("Local API is offline")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("API ready")).toBeInTheDocument();
  });
});
