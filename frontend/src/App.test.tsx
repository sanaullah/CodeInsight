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
    ["/architecture", "Architecture explorer"],
    ["/history", "Review history"],
    ["/settings", "Settings"],
    ["/missing", "Page not found"],
  ])("establishes the %s route without claiming the future feature is ready", async (path, heading) => {
    window.history.replaceState({}, "", path);
    render(<App />);
    expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
    expect(screen.getByText(/not yet implemented/i)).toBeInTheDocument();
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
