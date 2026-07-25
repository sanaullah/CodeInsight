import type { SemanticComponentKind, SemanticRelationKind } from "../../api/contracts";

export const ARCHITECTURE_VIEW_VERSION = "1";

export const COMPONENT_KINDS: SemanticComponentKind[] = [
  "service",
  "datastore",
  "external_system",
  "queue",
  "library",
  "unknown",
];

export const RELATION_KINDS: SemanticRelationKind[] = [
  "request",
  "event",
  "data_access",
  "dependency",
  "call",
  "unknown",
];

export interface ArchitectureViewState {
  snapshotId: string;
  focus: string;
  selectedId: string;
  showBoundaries: boolean;
  componentKinds: Set<SemanticComponentKind>;
  relationKinds: Set<SemanticRelationKind>;
  traceSource: string;
  traceTarget: string;
}

function isComponentKind(value: string): value is SemanticComponentKind {
  return COMPONENT_KINDS.includes(value as SemanticComponentKind);
}

function isRelationKind(value: string): value is SemanticRelationKind {
  return RELATION_KINDS.includes(value as SemanticRelationKind);
}

function decodeKinds<T extends string>(
  params: URLSearchParams,
  key: string,
  supported: T[],
  isSupported: (value: string) => value is T,
): Set<T> {
  const values = params.getAll(key);
  if (values.includes("none")) return new Set();
  const selected = values.filter(isSupported);
  return new Set(selected.length ? selected : supported);
}

export function decodeArchitectureView(search: string): ArchitectureViewState {
  const params = new URLSearchParams(search);
  return {
    snapshotId: params.get("snapshot") ?? "",
    focus: params.get("focus") ?? "",
    selectedId: params.get("selected") ?? "",
    showBoundaries: params.get("boundaries") !== "false",
    componentKinds: decodeKinds(params, "component_kind", COMPONENT_KINDS, isComponentKind),
    relationKinds: decodeKinds(params, "relation_kind", RELATION_KINDS, isRelationKind),
    traceSource: params.get("trace_source") ?? "",
    traceTarget: params.get("trace_target") ?? "",
  };
}

function encodeKinds<T extends string>(
  params: URLSearchParams,
  key: string,
  selected: Set<T>,
  supported: T[],
): void {
  if (selected.size === supported.length && supported.every((kind) => selected.has(kind))) return;
  if (selected.size === 0) {
    params.set(key, "none");
    return;
  }
  for (const kind of supported) {
    if (selected.has(kind)) params.append(key, kind);
  }
}

export function encodeArchitectureView(state: ArchitectureViewState): URLSearchParams {
  const params = new URLSearchParams({ view: ARCHITECTURE_VIEW_VERSION });
  if (state.snapshotId) params.set("snapshot", state.snapshotId);
  encodeKinds(params, "component_kind", state.componentKinds, COMPONENT_KINDS);
  encodeKinds(params, "relation_kind", state.relationKinds, RELATION_KINDS);
  if (state.focus) params.set("focus", state.focus);
  if (state.selectedId) params.set("selected", state.selectedId);
  if (!state.showBoundaries) params.set("boundaries", "false");
  if (state.traceSource) params.set("trace_source", state.traceSource);
  if (state.traceTarget) params.set("trace_target", state.traceTarget);
  return params;
}
