import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SemanticArchitectureGraph } from "../../api/contracts";
import { ArchitectureComparisonPanel } from "./ArchitectureComparisonPanel";
import { compareArchitectureGraphs } from "./model";

const component = {
  component_id: "service-current",
  snapshot_id: "current",
  stable_key: "service:orders",
  name: "Orders",
  component_kind: "service" as const,
  support_tier: "exact" as const,
  completeness: "complete" as const,
  confidence: 1,
  metadata: {},
};

function graph(snapshotId: string): SemanticArchitectureGraph {
  return {
    snapshot_id: snapshotId,
    display_name: snapshotId,
    git: { ref: "main", head_commit: snapshotId, dirty: false, parent_snapshot_id: null },
    created_at: "2026-07-25T12:00:00Z",
    counts: {
      services: 1,
      datastores: 0,
      external_systems: 0,
      queues: 0,
      libraries: 0,
      unknown: 0,
      boundaries: 0,
    },
    totals: { files: 1, lines: 10, languages: { TypeScript: 1 } },
    completeness: { status: "complete", extractor_version: "semantic-v1" },
    components: [{ ...component, snapshot_id: snapshotId }],
    boundaries: [],
    relations: [],
    findings: [],
    truncated: false,
    findings_truncated: false,
    limits: { depth: 1, node_limit: 160, finding_limit: 500 },
  };
}

describe("ArchitectureComparisonPanel", () => {
  it("shows stable-key changes with bounded-comparison disclosure", () => {
    const baseline = graph("baseline");
    const current = {
      ...graph("current"),
      components: [
        { ...component, name: "Orders API" },
        { ...component, component_id: "worker", stable_key: "service:worker", name: "Worker" },
      ],
    };

    render(
      <ArchitectureComparisonPanel
        baselineLabel="baseline"
        comparison={compareArchitectureGraphs(baseline, current)}
        currentLabel="current"
      />,
    );

    expect(screen.getByRole("heading", { name: "Snapshot comparison" })).toBeInTheDocument();
    expect(screen.getByText("Worker")).toBeInTheDocument();
    expect(screen.getByText("Orders API: name")).toBeInTheDocument();
    expect(screen.getByText(/not a raw repository diff/i)).toBeInTheDocument();
  });

  it("surfaces bounded comparison failures", () => {
    render(
      <ArchitectureComparisonPanel
        baselineLabel="baseline"
        comparison={null}
        currentLabel="current"
        error="Comparison unavailable"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Comparison unavailable");
  });
});
