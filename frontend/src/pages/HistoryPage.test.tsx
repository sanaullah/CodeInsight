import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { apiClient } from "../api/client";
import type { HistoryRun } from "../api/contracts";
import { HistoryPage } from "./HistoryPage";

const runs: HistoryRun[] = ["run-001", "run-002"].map((runId) => ({
  run_id: runId,
  status: "succeeded",
  mode: "deep",
  current_stage: "complete",
  created_at: "2026-07-25T12:00:00Z",
  started_at: null,
  completed_at: null,
  snapshot_id: null,
  display_name: null,
  base_commit: null,
  head_commit: null,
  dirty: null,
  task_count: 0,
  specialist_count: 0,
  finding_count: 0,
  duration_seconds: null,
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  cost_usd: 0,
}));

beforeEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState({}, "", "/history");
});

it("shows truthful empty and partial-history states", async () => {
  vi.spyOn(apiClient, "history").mockResolvedValue({ items: [], next_cursor: null });
  vi.spyOn(apiClient, "historyTrends").mockResolvedValue({
    days: 30,
    partial: true,
    buckets: [],
  });
  render(<HistoryPage />);
  expect(await screen.findByText("No reviews in this view")).toBeInTheDocument();
  expect(screen.getByText(/no regression conclusion is implied/i)).toBeInTheDocument();
  expect(screen.getByText("—")).toBeInTheDocument();
});

it("surfaces history loading failures", async () => {
  vi.spyOn(apiClient, "history").mockRejectedValue(new Error("History unavailable"));
  vi.spyOn(apiClient, "historyTrends").mockResolvedValue({
    days: 30,
    partial: true,
    buckets: [],
  });
  render(<HistoryPage />);
  expect(await screen.findByRole("alert")).toHaveTextContent("History unavailable");
});

it("supports deselection and reports comparison failures", async () => {
  vi.spyOn(apiClient, "history").mockResolvedValue({ items: runs, next_cursor: null });
  vi.spyOn(apiClient, "historyTrends").mockResolvedValue({
    days: 30,
    partial: false,
    buckets: [],
  });
  vi.spyOn(apiClient, "compareRuns").mockRejectedValue(new Error("Comparison unavailable"));
  const user = userEvent.setup();
  render(<HistoryPage />);
  await user.click(await screen.findByLabelText("Compare run-001"));
  await user.click(screen.getByLabelText("Compare run-002"));
  await user.click(screen.getByLabelText("Compare run-002"));
  expect(screen.getByRole("button", { name: "Compare selected (1/2)" })).toBeDisabled();
  await user.click(screen.getByLabelText("Compare run-002"));
  await user.click(screen.getByRole("button", { name: "Compare selected (2/2)" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Comparison unavailable");
});
