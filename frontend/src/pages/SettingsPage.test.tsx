import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "../api/client";
import type { LocalSettings, ReviewPreset } from "../api/contracts";
import { capabilitiesFixture, healthFixture } from "../api/fixtures";
import { SettingsPage } from "./SettingsPage";

const defaults: LocalSettings = {
  default_mode: "deep",
  default_max_agents: 4,
  default_max_waves: 2,
  default_max_tasks: 100,
  default_max_total_tokens: 1_000_000,
  default_max_cost_usd: 25,
  default_max_elapsed_seconds: 3_600,
  evidence_excerpt_enabled: true,
  retention_days: 90,
};

const preset: ReviewPreset = {
  preset_id: "preset-1",
  name: "Quick triage",
  request: {
    goal: null,
    model_name: null,
    file_extensions: null,
    selected_directories: null,
    mode: "quick",
    max_agents: 2,
    max_waves: 1,
    max_tasks: 20,
    max_total_tokens: 10_000,
    max_cost_usd: 1,
    max_elapsed_seconds: 120,
  },
  version: 1,
  created_at: "2026-07-25T12:00:00Z",
  updated_at: "2026-07-25T12:00:00Z",
};

const operational = {
  health: healthFixture,
  capabilities: { ...capabilitiesFixture, model_provider_configured: true, langfuse_enabled: true },
  loading: false,
  error: null,
  refresh: vi.fn(),
};

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(apiClient, "settings").mockResolvedValue({
    settings: defaults,
    version: 3,
    updated_at: "2026-07-25T12:00:00Z",
  });
  vi.spyOn(apiClient, "presets").mockResolvedValue([preset]);
  vi.spyOn(apiClient, "updateSettings").mockResolvedValue({
    settings: defaults,
    version: 4,
    updated_at: "2026-07-25T12:01:00Z",
  });
  vi.spyOn(apiClient, "testProvider").mockResolvedValue({
    configured: true,
    reachable: true,
    status: "Provider responded with HTTP 200.",
    latency_ms: 8,
  });
  vi.spyOn(apiClient, "createPreset").mockResolvedValue(preset);
  vi.spyOn(apiClient, "updatePreset").mockImplementation(async (value) => ({
    ...value,
    version: value.version + 1,
  }));
  vi.spyOn(apiClient, "deletePreset").mockResolvedValue();
});

describe("SettingsPage", () => {
  it("edits every bounded default, discards, applies a preset, and saves", async () => {
    const user = userEvent.setup();
    render(<SettingsPage operational={operational} />);
    expect(await screen.findByText("Quick triage")).toBeInTheDocument();

    for (const [label, value] of [
      ["Maximum specialists", "6"],
      ["Maximum waves", "3"],
      ["Maximum tasks", "120"],
      ["Maximum model tokens", "20000"],
      ["Maximum cost (USD)", "12"],
      ["Maximum elapsed seconds", "900"],
      ["Retention days", "30"],
    ] as const) {
      const input = screen.getByLabelText(label);
      await user.clear(input);
      await user.type(input, value);
    }
    await user.click(screen.getByRole("checkbox", { name: /show evidence excerpts/i }));
    await user.click(screen.getByRole("button", { name: "Discard" }));
    expect(screen.getByLabelText("Maximum specialists")).toHaveValue(4);

    await user.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByLabelText("Default mode")).toHaveValue("quick");
    expect(screen.getByLabelText("Maximum specialists")).toHaveValue(2);
    await user.click(screen.getByRole("button", { name: "Save defaults" }));
    await waitFor(() =>
      expect(apiClient.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ default_mode: "quick", default_max_agents: 2 }),
        3,
      ),
    );
    await user.click(screen.getByRole("button", { name: "Rename" }));
    const name = screen.getByLabelText("Rename Quick triage");
    await user.clear(name);
    await user.type(name, "Fast triage");
    await user.click(screen.getByRole("button", { name: "Save name" }));
    expect(await screen.findByText("Fast triage")).toBeInTheDocument();
  });

  it("shows truthful configured services and bounded provider latency", async () => {
    const user = userEvent.setup();
    render(<SettingsPage operational={operational} />);
    await screen.findByText("Optional export active");
    expect(screen.getByText("Configured")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Test provider" }));
    expect(await screen.findByText("Provider responded with HTTP 200. 8 ms")).toBeInTheDocument();
  });

  it("surfaces load, save, provider, create, and delete failures", async () => {
    vi.mocked(apiClient.settings).mockRejectedValueOnce(new Error("Settings unavailable"));
    const { unmount } = render(<SettingsPage operational={operational} />);
    expect(await screen.findByText("Settings unavailable")).toBeInTheDocument();
    unmount();

    const user = userEvent.setup();
    vi.mocked(apiClient.settings).mockResolvedValue({
      settings: defaults,
      version: 3,
      updated_at: null,
    });
    vi.mocked(apiClient.updateSettings).mockRejectedValueOnce(new Error("Refresh and retry"));
    vi.mocked(apiClient.testProvider).mockRejectedValueOnce(new Error("Probe failed"));
    vi.mocked(apiClient.createPreset).mockRejectedValueOnce(new Error("Duplicate preset"));
    vi.mocked(apiClient.deletePreset).mockRejectedValueOnce(new Error("Preset changed"));
    render(<SettingsPage operational={operational} />);
    await screen.findByText("Quick triage");

    await user.selectOptions(screen.getByLabelText("Default mode"), "security");
    await user.click(screen.getByRole("button", { name: "Save defaults" }));
    expect(await screen.findByText("Refresh and retry")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Test provider" }));
    expect(await screen.findByText("Probe failed")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Preset name"), "Duplicate");
    await user.click(screen.getByRole("button", { name: "Save current defaults" }));
    expect(await screen.findByText("Duplicate preset")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(await screen.findByText("Preset changed")).toBeInTheDocument();
  });
});
