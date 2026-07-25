import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "../api/client";
import type { FindingDetail, FindingSummary } from "../api/contracts";
import { FindingsPage } from "./FindingsPage";

const summary: FindingSummary = {
  finding_id: "finding-1",
  fingerprint: "auth",
  title: "Authorization is bypassed",
  claim: "Every user is permitted.",
  severity: "high",
  confidence: 0.94,
  candidate_ids: ["candidate-1"],
  supporting_evidence_ids: ["evidence-1"],
  conflicting_candidate_ids: [],
  recommendation: "Enforce policy.",
  run_id: "run-1",
  review_state: "new",
  review_note: null,
  review_version: 0,
  reviewed_at: null,
};

const detail: FindingDetail = {
  ...summary,
  candidates: [
    {
      relationship: "supporting",
      candidate: {
        candidate_id: "candidate-1",
        title: summary.title,
        claim: summary.claim,
        category: "security",
        affected_path: "src/auth.py",
        impact: "Unauthorized access",
        proposed_severity: "high",
        proposed_confidence: 0.94,
        recommendation: summary.recommendation,
      },
      role: null,
      verdict: null,
    },
  ],
  evidence: [
    {
      evidence_id: "evidence-1",
      relative_path: "src/auth.py",
      content_hash: "abcdef123456",
      start_line: 2,
      end_line: 2,
      evidence_kind: "source",
      provenance: {},
      excerpt: null,
      integrity: "unavailable",
      redacted: true,
    },
  ],
  review_history: [],
};

beforeEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState({}, "", "/findings");
  vi.spyOn(apiClient, "findings").mockResolvedValue({
    items: [summary],
    next_cursor: "finding-1",
    counts_by_severity: { high: 1 },
  });
  vi.spyOn(apiClient, "finding").mockResolvedValue(detail);
  vi.spyOn(apiClient, "updateFinding").mockImplementation(async (_id, payload) => ({
    ...detail,
    review_state: payload.review_state,
    review_version: payload.expected_version + 1,
  }));
});

describe("findings workspace", () => {
  it("filters by severity and review state and reports pagination", async () => {
    const user = userEvent.setup();
    render(<FindingsPage />);

    expect(await screen.findByRole("button", { name: "Load more findings" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "High1" }));
    await user.selectOptions(screen.getByLabelText("Review state"), "new");
    await waitFor(() => {
      const params = vi.mocked(apiClient.findings).mock.calls.at(-1)?.[0];
      expect(params?.get("severity")).toBe("high");
      expect(params?.get("review_state")).toBe("new");
    });
  });

  it("selects and bulk-reviews findings", async () => {
    const user = userEvent.setup();
    render(<FindingsPage />);
    await screen.findByText(summary.title);
    await user.click(screen.getByLabelText(`Select ${summary.title}`));
    await user.click(screen.getByRole("button", { name: "Mark selected reviewed (1)" }));
    await user.click(screen.getByRole("button", { name: "Confirm bulk review" }));

    await waitFor(() =>
      expect(apiClient.updateFinding).toHaveBeenCalledWith("finding-1", {
        review_state: "reviewed",
        expected_version: 0,
      }),
    );
  });

  it("opens evidence, changes lifecycle state, and closes the drawer", async () => {
    const user = userEvent.setup();
    render(<FindingsPage />);
    await user.click(await screen.findByRole("button", { name: /Authorization is bypassed/ }));

    expect(await screen.findByRole("dialog", { name: "Finding detail" })).toBeInTheDocument();
    expect(screen.getByText(/failed integrity validation/i)).toBeInTheDocument();
    expect(screen.getByText("No verifier verdict is available.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Validated" }));
    await waitFor(() => expect(apiClient.updateFinding).toHaveBeenCalled());
    await user.click(screen.getAllByRole("button", { name: "Close finding detail" })[0]);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(window.location.search).not.toContain("finding=");
  });

  it("opens a durable finding deep link without waiting for list selection", async () => {
    window.history.replaceState({}, "", "/findings?finding=finding-1");
    render(<FindingsPage />);

    expect(await screen.findByRole("dialog", { name: "Finding detail" })).toBeInTheDocument();
    expect(apiClient.finding).toHaveBeenCalledWith("finding-1", expect.any(AbortSignal));
    expect(window.location.search).toContain("finding=finding-1");
  });

  it("surfaces list and detail failures with retry", async () => {
    vi.mocked(apiClient.findings)
      .mockRejectedValueOnce(new Error("Findings unavailable"))
      .mockResolvedValueOnce({ items: [], next_cursor: null, counts_by_severity: {} });
    const user = userEvent.setup();
    render(<FindingsPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Findings unavailable");
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("No findings in this view")).toBeInTheDocument();

    vi.mocked(apiClient.findings).mockResolvedValue({
      items: [summary],
      next_cursor: null,
      counts_by_severity: { high: 1 },
    });
    vi.mocked(apiClient.finding).mockRejectedValue(new Error("Evidence unavailable"));
    await user.type(screen.getByLabelText("Search"), "auth");
    await user.click(await screen.findByRole("button", { name: /Authorization is bypassed/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Evidence unavailable");
  });

  it("surfaces lifecycle update conflicts", async () => {
    vi.mocked(apiClient.updateFinding).mockRejectedValue(new Error("Refresh and retry"));
    const user = userEvent.setup();
    render(<FindingsPage />);
    await user.click(await screen.findByRole("button", { name: /Authorization is bypassed/ }));
    await user.click(await screen.findByRole("button", { name: "Reviewed" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Refresh and retry");
  });

  it("preserves the current page when cursor loading fails", async () => {
    vi.mocked(apiClient.findings)
      .mockResolvedValueOnce({
        items: [summary],
        next_cursor: "next",
        counts_by_severity: { high: 1 },
      })
      .mockRejectedValueOnce(new Error("Next page unavailable"));
    const user = userEvent.setup();
    render(<FindingsPage />);
    await user.click(await screen.findByRole("button", { name: "Load more findings" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Next page unavailable");
    expect(screen.getByText(summary.title)).toBeInTheDocument();
  });
});
