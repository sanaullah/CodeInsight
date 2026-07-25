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
  vi.spyOn(apiClient, "history").mockResolvedValue({ items: [], next_cursor: null });
  vi.spyOn(apiClient, "historyTrends").mockResolvedValue({
    days: 30,
    partial: true,
    buckets: [],
  });
  vi.spyOn(apiClient, "compareRuns").mockRejectedValue(new Error("Runs not selected"));
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

  it("provides an accessible mobile menu escape path", async () => {
    const user = userEvent.setup();
    render(<App />);
    const menu = screen.getByRole("button", { name: "Open navigation" });
    await user.click(menu);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(menu).toHaveAttribute("aria-expanded", "false");
  });

  it.each([
    ["/settings", "Settings"],
    ["/missing", "Page not found"],
  ])("establishes the %s route without claiming the future feature is ready", async (path, heading) => {
    window.history.replaceState({}, "", path);
    render(<App />);
    expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
    expect(screen.getByText(/not yet implemented/i)).toBeInTheDocument();
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
    vi.mocked(apiClient.history).mockResolvedValue({
      items: historyRuns,
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

  it("explores persisted architecture and traces a directed path", async () => {
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
    const nodes = [
      {
        node_id: "file-a",
        label: "a.py",
        node_kind: "file" as const,
        language: "python",
        classification: "source",
        support_tier: "parsed",
        line_count: 10,
        confidence: 1,
        derivation: "repository-index" as const,
        findings: [],
      },
      {
        node_id: "file-b",
        label: "b.py",
        node_kind: "file" as const,
        language: "python",
        classification: "source",
        support_tier: "parsed",
        line_count: 20,
        confidence: 1,
        derivation: "repository-index" as const,
        findings: [{ finding_id: "finding-1", severity: "high" as const }],
      },
    ];
    vi.mocked(apiClient.architecture).mockResolvedValue({
      snapshot_id: "snapshot-1",
      display_name: "Fixture",
      summary: {
        file_count: 2,
        language_counts: { python: 2 },
        classification_counts: { source: 2 },
        changed_paths: ["a.py"],
      },
      nodes,
      edges: [
        {
          edge_id: "edge-1",
          source_id: "file-a",
          target_id: "file-b",
          edge_kind: "imports",
          confidence: 1,
          derivation: "repository-index",
        },
      ],
      truncated: false,
      limits: { depth: 1, node_limit: 250 },
    });
    vi.mocked(apiClient.trace).mockResolvedValue({
      snapshot_id: "snapshot-1",
      found: true,
      nodes,
      edges: [],
      max_hops: 8,
    });

    render(<App />);
    expect(await screen.findByText("Bounded dependency map")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "b.py" }));
    expect(screen.getByText("repository-index")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Trace source"), "file-a");
    await user.selectOptions(screen.getByLabelText("Trace target"), "file-b");
    await user.click(screen.getByRole("button", { name: "Trace path" }));
    expect(await screen.findByText("a.py → b.py")).toBeInTheDocument();
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
    const findings = vi.mocked(apiClient.findings).mockResolvedValue({
      items: [finding],
      next_cursor: null,
      counts_by_severity: { high: 1 },
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
    await user.click(screen.getByRole("button", { name: "Reviewed" }));
    await waitFor(() =>
      expect(apiClient.updateFinding).toHaveBeenCalledWith("finding-1", {
        review_state: "reviewed",
        expected_version: 0,
      }),
    );
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
