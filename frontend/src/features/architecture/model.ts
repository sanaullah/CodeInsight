import type {
  SemanticArchitectureGraph,
  SemanticComponent,
  SemanticTrace,
} from "../../api/contracts";
import type { ArchitectureLens } from "./viewState";

export type ArchitectureProjectionStatus =
  | "complete"
  | "partial"
  | "truncated"
  | "unsupported"
  | "no_path";

export interface TopologyPosition {
  x: number;
  y: number;
}

export interface ArchitectureLayer {
  id: string;
  label: string;
  count: number;
  active: boolean;
  description?: string;
}

export function componentId(component: SemanticComponent): string {
  return component.component_id;
}

export function componentLabel(component: SemanticComponent): string {
  return component.name || component.stable_key || component.component_id;
}

export function projectionStatus(graph: SemanticArchitectureGraph): ArchitectureProjectionStatus {
  if (graph.truncated || graph.findings_truncated) return "truncated";
  const status = graph.completeness?.status;
  if (status === "partial" || status === "unsupported" || status === "complete") {
    return status;
  }
  return "complete";
}

export function traceStatus(trace: SemanticTrace | null): ArchitectureProjectionStatus {
  if (!trace) return "complete";
  if (
    trace.status === "truncated" ||
    trace.status === "unsupported" ||
    trace.status === "no_path"
  ) {
    return trace.status;
  }
  return "complete";
}

export function relationLabel(relationKind: string): string {
  switch (relationKind) {
    case "request":
      return "request";
    case "event":
      return "event";
    case "data_access":
      return "data access";
    case "dependency":
      return "declared dependency";
    case "imports":
    case "import":
      return "static import";
    case "calls":
    case "call":
      return "static call reference";
    default:
      return relationKind.replaceAll("_", " ");
  }
}

export function relationIsRuntimeFlow(relationKind: string): boolean {
  return relationKind === "request" || relationKind === "event" || relationKind === "data_access";
}

export function layoutComponents(
  components: SemanticComponent[],
  columns: number,
): Map<string, TopologyPosition> {
  const safeColumns = Math.max(1, columns);
  const positions = new Map<string, TopologyPosition>();
  components.forEach((component, index) => {
    positions.set(componentId(component), {
      x: 32 + (index % safeColumns) * 220,
      y: 34 + Math.floor(index / safeColumns) * 132,
    });
  });
  return positions;
}

export function componentMap(components: SemanticComponent[]): Map<string, SemanticComponent> {
  return new Map(components.map((component) => [componentId(component), component]));
}

export function graphLayers(graph: SemanticArchitectureGraph): ArchitectureLayer[] {
  const counts = new Map<string, number>();
  for (const component of graph.components) {
    counts.set(component.component_kind, (counts.get(component.component_kind) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, count]) => ({
      id,
      label: id.replaceAll("_", " "),
      count,
      active: true,
    }));
}

export function applyArchitectureLens(
  graph: SemanticArchitectureGraph,
  lens: ArchitectureLens,
  selectedComponentId: string,
): SemanticArchitectureGraph {
  if (lens === "topology") return graph;

  const componentIds = new Set<string>();
  if (lens === "security") {
    for (const finding of graph.findings) componentIds.add(finding.component_id);
  } else if (
    selectedComponentId &&
    graph.components.some((component) => component.component_id === selectedComponentId)
  ) {
    componentIds.add(selectedComponentId);
    for (const relation of graph.relations) {
      if (relation.source_component_id === selectedComponentId) {
        componentIds.add(relation.target_component_id);
      }
      if (relation.target_component_id === selectedComponentId) {
        componentIds.add(relation.source_component_id);
      }
    }
  }

  return {
    ...graph,
    components: graph.components.filter((component) => componentIds.has(component.component_id)),
    boundaries: graph.boundaries
      .map((boundary) => ({
        ...boundary,
        component_ids: boundary.component_ids.filter((id) => componentIds.has(id)),
      }))
      .filter((boundary) => boundary.component_ids.length > 0),
    relations: graph.relations.filter(
      (relation) =>
        componentIds.has(relation.source_component_id) &&
        componentIds.has(relation.target_component_id),
    ),
    findings: graph.findings.filter((finding) => componentIds.has(finding.component_id)),
  };
}
