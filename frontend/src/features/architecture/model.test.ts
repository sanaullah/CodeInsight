import { describe, expect, it } from "vitest";
import type {
  SemanticArchitectureGraph,
  SemanticComponent,
  SemanticTrace,
} from "../../api/contracts";
import {
  componentLabel,
  componentMap,
  graphLayers,
  layoutComponents,
  projectionStatus,
  relationIsRuntimeFlow,
  relationLabel,
  traceStatus,
} from "./model";

const components: SemanticComponent[] = [
  {
    component_id: "one",
    snapshot_id: "snapshot",
    stable_key: "service:one",
    name: "One",
    component_kind: "service",
    support_tier: "exact",
    completeness: "complete",
    confidence: 1,
    metadata: {},
  },
  {
    component_id: "two",
    snapshot_id: "snapshot",
    stable_key: "library:two",
    name: "Two",
    component_kind: "library",
    support_tier: "inferred",
    completeness: "partial",
    confidence: 0.7,
    metadata: {},
  },
];

describe("architecture view model", () => {
  it("builds constant-time component and position lookups", () => {
    expect(componentMap(components).get("two")?.name).toBe("Two");
    expect(layoutComponents(components, 1).get("two")).toEqual({ x: 32, y: 166 });
    expect(layoutComponents(components, 0).get("two")).toEqual({ x: 32, y: 166 });
    expect(componentLabel({ ...components[0], name: "" })).toBe("service:one");
    expect(componentLabel({ ...components[0], name: "", stable_key: "" })).toBe("one");
  });

  it("only treats supported semantic relations as runtime flow", () => {
    expect(relationIsRuntimeFlow("request")).toBe(true);
    expect(relationIsRuntimeFlow("event")).toBe(true);
    expect(relationIsRuntimeFlow("data_access")).toBe(true);
    expect(relationIsRuntimeFlow("call")).toBe(false);
    expect(relationLabel("request")).toBe("request");
    expect(relationLabel("event")).toBe("event");
    expect(relationLabel("data_access")).toBe("data access");
    expect(relationLabel("call")).toBe("static call reference");
    expect(relationLabel("calls")).toBe("static call reference");
    expect(relationLabel("import")).toBe("static import");
    expect(relationLabel("imports")).toBe("static import");
    expect(relationLabel("dependency")).toBe("declared dependency");
    expect(relationLabel("custom_relation")).toBe("custom relation");
  });

  it("reports projection and trace completeness without hiding bounded states", () => {
    const graph = {
      components,
      findings: [],
      truncated: false,
      completeness: { status: "complete" },
    } as unknown as SemanticArchitectureGraph;
    expect(projectionStatus(graph)).toBe("complete");
    expect(projectionStatus({ ...graph, truncated: true })).toBe("truncated");
    expect(projectionStatus({ ...graph, findings_truncated: true })).toBe("truncated");
    expect(
      projectionStatus({ ...graph, completeness: { status: "partial", extractor_version: null } }),
    ).toBe("partial");
    expect(
      projectionStatus({
        ...graph,
        completeness: { status: "unsupported", extractor_version: null },
      }),
    ).toBe("unsupported");
    expect(
      projectionStatus({
        ...graph,
        completeness: { status: "unknown", extractor_version: null },
      }),
    ).toBe("complete");

    const trace = { status: "complete" } as SemanticTrace;
    expect(traceStatus(null)).toBe("complete");
    expect(traceStatus(trace)).toBe("complete");
    expect(traceStatus({ ...trace, status: "truncated" })).toBe("truncated");
    expect(traceStatus({ ...trace, status: "unsupported" })).toBe("unsupported");
    expect(traceStatus({ ...trace, status: "no_path" })).toBe("no_path");
  });

  it("derives stable, sorted layer counts from semantic components", () => {
    expect(
      graphLayers({
        components: [...components, components[0]],
      } as unknown as SemanticArchitectureGraph),
    ).toEqual([
      { id: "library", label: "library", count: 1, active: true },
      { id: "service", label: "service", count: 2, active: true },
    ]);
  });
});
