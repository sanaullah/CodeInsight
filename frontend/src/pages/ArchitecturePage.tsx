import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type {
  FindingDetail,
  SemanticArchitectureGraph,
  SemanticComponent,
  SemanticComponentDetail,
  SemanticComponentKind,
  SemanticRelationKind,
  SemanticTrace,
  SnapshotSummary,
} from "../api/contracts";
import { EmptyState, ErrorNotice, PageHeader, Panel, StatCard } from "../components/primitives";
import {
  ArchitectureTextAlternative,
  ArchitectureWorkspace,
  ComponentInspector,
  FindingEvidenceDrawer,
  LayerRail,
  TopologyViewport,
  TraceRail,
} from "../features/architecture";
import { shortId } from "../format";

const COMPONENT_KINDS: SemanticComponentKind[] = [
  "service",
  "datastore",
  "external_system",
  "queue",
  "library",
  "unknown",
];
const RELATION_KINDS: SemanticRelationKind[] = [
  "request",
  "event",
  "data_access",
  "dependency",
  "call",
  "unknown",
];
const KIND_LABELS: Record<SemanticComponentKind, string> = {
  service: "Services",
  datastore: "Data stores",
  external_system: "External systems",
  queue: "Queues and events",
  library: "Libraries",
  unknown: "Unknown",
};

function initialValues() {
  const params = new URLSearchParams(window.location.search);
  const componentKinds = params.getAll("component_kind").filter(isComponentKind);
  const relationKinds = params.getAll("relation_kind").filter(isRelationKind);
  return {
    snapshotId: params.get("snapshot") ?? "",
    focus: params.get("focus") ?? "",
    selectedId: params.get("selected") ?? "",
    showBoundaries: params.get("boundaries") !== "false",
    componentKinds: new Set<SemanticComponentKind>(
      componentKinds.length ? componentKinds : COMPONENT_KINDS,
    ),
    relationKinds: new Set<SemanticRelationKind>(
      relationKinds.length ? relationKinds : RELATION_KINDS,
    ),
  };
}

function isComponentKind(value: string): value is SemanticComponentKind {
  return COMPONENT_KINDS.includes(value as SemanticComponentKind);
}

function isRelationKind(value: string): value is SemanticRelationKind {
  return RELATION_KINDS.includes(value as SemanticRelationKind);
}

export function ArchitecturePage() {
  const initial = useMemo(initialValues, []);
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [snapshotId, setSnapshotId] = useState(initial.snapshotId);
  const [focus, setFocus] = useState(initial.focus);
  const [selectedId, setSelectedId] = useState(initial.selectedId);
  const [componentKinds, setComponentKinds] = useState(initial.componentKinds);
  const [relationKinds, setRelationKinds] = useState(initial.relationKinds);
  const [showBoundaries, setShowBoundaries] = useState(initial.showBoundaries);
  const [graph, setGraph] = useState<SemanticArchitectureGraph | null>(null);
  const [detail, setDetail] = useState<SemanticComponentDetail | null>(null);
  const [finding, setFinding] = useState<FindingDetail | null>(null);
  const [trace, setTrace] = useState<SemanticTrace | null>(null);
  const [traceSource, setTraceSource] = useState("");
  const [traceTarget, setTraceTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [findingLoading, setFindingLoading] = useState(false);
  const [traceLoading, setTraceLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [findingError, setFindingError] = useState<string | null>(null);
  const [traceError, setTraceError] = useState<string | null>(null);

  const visibleGraph = useMemo(() => {
    if (!graph) return null;
    const components = graph.components.filter((component) =>
      componentKinds.has(component.component_kind),
    );
    const componentIds = new Set(components.map((component) => component.component_id));
    return {
      ...graph,
      components,
      boundaries: showBoundaries
        ? graph.boundaries
            .map((boundary) => ({
              ...boundary,
              component_ids: boundary.component_ids.filter((id) => componentIds.has(id)),
            }))
            .filter((boundary) => boundary.component_ids.length > 0)
        : [],
      relations: graph.relations.filter(
        (relation) =>
          relationKinds.has(relation.relation_kind) &&
          componentIds.has(relation.source_component_id) &&
          componentIds.has(relation.target_component_id),
      ),
      findings: graph.findings.filter((findingLink) => componentIds.has(findingLink.component_id)),
    };
  }, [componentKinds, graph, relationKinds, showBoundaries]);
  const selected =
    visibleGraph?.components.find((component) => component.component_id === selectedId) ?? null;

  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .snapshots(controller.signal)
      .then((items) => {
        setSnapshots(items);
        setSnapshotId((current) => current || items[0]?.snapshot_id || "");
      })
      .catch((reason) =>
        setError(reason instanceof Error ? reason.message : "Unable to load snapshots"),
      );
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const route = new URLSearchParams();
    if (snapshotId) route.set("snapshot", snapshotId);
    for (const kind of COMPONENT_KINDS) {
      if (componentKinds.has(kind) && componentKinds.size < COMPONENT_KINDS.length) {
        route.append("component_kind", kind);
      }
    }
    for (const kind of RELATION_KINDS) {
      if (relationKinds.has(kind) && relationKinds.size < RELATION_KINDS.length) {
        route.append("relation_kind", kind);
      }
    }
    if (focus) route.set("focus", focus);
    if (selectedId) route.set("selected", selectedId);
    if (!showBoundaries) route.set("boundaries", "false");
    const query = route.toString();
    window.history.replaceState({}, "", query ? `/architecture?${query}` : "/architecture");
  }, [componentKinds, focus, relationKinds, selectedId, showBoundaries, snapshotId]);

  useEffect(() => {
    if (!snapshotId) return;
    const controller = new AbortController();
    const params = new URLSearchParams({ limit: "160", depth: focus ? "2" : "1" });
    if (componentKinds.size < COMPONENT_KINDS.length) {
      for (const kind of COMPONENT_KINDS) {
        if (componentKinds.has(kind)) params.append("component_kind", kind);
      }
    }
    if (relationKinds.size < RELATION_KINDS.length) {
      for (const kind of RELATION_KINDS) {
        if (relationKinds.has(kind)) params.append("relation_kind", kind);
      }
    }
    if (focus) params.set("focus", focus);
    setLoading(true);
    apiClient
      .semanticArchitecture(snapshotId, params, controller.signal)
      .then((value) => {
        setGraph(value);
        setError(null);
        setTrace(null);
        setSelectedId((current) =>
          current && !value.components.some((item) => item.component_id === current) ? "" : current,
        );
      })
      .catch((reason) =>
        setError(reason instanceof Error ? reason.message : "Unable to load architecture"),
      )
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [componentKinds, focus, relationKinds, snapshotId]);

  useEffect(() => {
    if (!snapshotId || !selectedId) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    setDetailLoading(true);
    setDetailError(null);
    apiClient
      .semanticComponent(snapshotId, selectedId, controller.signal)
      .then(setDetail)
      .catch((reason) =>
        setDetailError(reason instanceof Error ? reason.message : "Unable to load component"),
      )
      .finally(() => setDetailLoading(false));
    return () => controller.abort();
  }, [selectedId, snapshotId]);

  function selectComponent(component: SemanticComponent) {
    setSelectedId(component.component_id);
    setTraceSource((current) => current || component.component_id);
  }

  function focusComponent(component: SemanticComponent) {
    setFocus(component.component_id);
    setSelectedId(component.component_id);
  }

  function toggleComponentKind(kind: string, active: boolean) {
    if (!isComponentKind(kind)) return;
    setComponentKinds((current) => {
      const next = new Set(current);
      if (active) next.add(kind);
      else next.delete(kind);
      return next;
    });
  }

  function toggleRelationKind(kind: SemanticRelationKind) {
    setRelationKinds((current) => {
      const next = new Set(current);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });
  }

  async function openFinding(findingId: string) {
    setFinding(null);
    setFindingLoading(true);
    setFindingError(null);
    try {
      setFinding(await apiClient.finding(findingId));
    } catch (reason) {
      setFindingError(reason instanceof Error ? reason.message : "Unable to load finding");
    } finally {
      setFindingLoading(false);
    }
  }

  async function loadTrace() {
    if (!snapshotId || !traceSource || !traceTarget) return;
    setTraceLoading(true);
    setTraceError(null);
    try {
      setTrace(await apiClient.semanticTrace(snapshotId, traceSource, traceTarget, 8));
    } catch (reason) {
      setTraceError(reason instanceof Error ? reason.message : "Unable to load trace");
    } finally {
      setTraceLoading(false);
    }
  }

  const layers = COMPONENT_KINDS.map((kind) => ({
    id: kind,
    label: KIND_LABELS[kind],
    count: graph
      ? kind === "service"
        ? graph.counts.services
        : kind === "datastore"
          ? graph.counts.datastores
          : kind === "external_system"
            ? graph.counts.external_systems
            : kind === "queue"
              ? graph.counts.queues
              : kind === "library"
                ? graph.counts.libraries
                : graph.counts.unknown
      : 0,
    active: componentKinds.has(kind),
  }));

  const traceControls = visibleGraph ? (
    <Panel className="architecture-trace-shell">
      <div className="architecture-trace-controls">
        <label>
          <span className="architecture-control-label">Trace source</span>
          <select onChange={(event) => setTraceSource(event.target.value)} value={traceSource}>
            <option value="">Select source</option>
            {visibleGraph.components.map((component) => (
              <option key={component.component_id} value={component.component_id}>
                {component.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="architecture-control-label">Trace target</span>
          <select onChange={(event) => setTraceTarget(event.target.value)} value={traceTarget}>
            <option value="">Select target</option>
            {visibleGraph.components.map((component) => (
              <option key={component.component_id} value={component.component_id}>
                {component.name}
              </option>
            ))}
          </select>
        </label>
        <button
          className="button button-secondary"
          disabled={!traceSource || !traceTarget || traceLoading}
          onClick={() => void loadTrace()}
          type="button"
        >
          {traceLoading ? "Tracing..." : "Trace evidence"}
        </button>
      </div>
      <TraceRail
        error={traceError}
        loading={traceLoading}
        onSelectComponent={selectComponent}
        sourceLabel={
          visibleGraph.components.find((item) => item.component_id === traceSource)?.name
        }
        targetLabel={
          visibleGraph.components.find((item) => item.component_id === traceTarget)?.name
        }
        trace={trace}
      />
    </Panel>
  ) : null;

  return (
    <div className="page architecture-page">
      <PageHeader
        description="Explore bounded, evidence-derived services, stores, external systems, queues, and typed relations. Static dependencies are shown as static facts, never runtime data flow."
        eyebrow="Repository evidence"
        title="Architecture explorer"
      />
      {error ? <ErrorNotice message={error} /> : null}
      <Panel className="architecture-semantic-controls">
        <label>
          <span className="architecture-control-label">Repository snapshot</span>
          <select onChange={(event) => setSnapshotId(event.target.value)} value={snapshotId}>
            <option value="">Select a snapshot</option>
            {snapshots.map((snapshot) => (
              <option key={snapshot.snapshot_id} value={snapshot.snapshot_id}>
                {snapshot.display_name} · {shortId(snapshot.snapshot_id)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="architecture-control-label">Focus neighborhood</span>
          <input
            onChange={(event) => setFocus(event.target.value)}
            placeholder="Component ID, stable key, or exact name"
            value={focus}
          />
        </label>
        <button
          className="button button-secondary"
          disabled={!focus}
          onClick={() => setFocus("")}
          type="button"
        >
          Clear focus
        </button>
        <fieldset className="architecture-relation-filters">
          <legend className="architecture-control-label">Typed relation layers</legend>
          {RELATION_KINDS.map((kind) => (
            <label key={kind}>
              <input
                checked={relationKinds.has(kind)}
                onChange={() => toggleRelationKind(kind)}
                type="checkbox"
              />
              <span>{kind.replaceAll("_", " ")}</span>
            </label>
          ))}
          <label>
            <input
              checked={showBoundaries}
              onChange={(event) => setShowBoundaries(event.target.checked)}
              type="checkbox"
            />
            <span>boundaries ({graph?.counts.boundaries ?? 0})</span>
          </label>
        </fieldset>
      </Panel>
      {!snapshotId ? (
        <EmptyState
          description="Run a review first to create an immutable repository snapshot."
          title="No repository snapshot selected"
        />
      ) : graph && visibleGraph ? (
        <>
          <div className="architecture-snapshot-strip">
            <div>
              <strong>{graph.display_name}</strong>
              <span className="architecture-snapshot-metadata">
                {graph.git.ref ?? "No Git ref"} ·{" "}
                {graph.git.head_commit?.slice(0, 12) ?? "No commit"}
                {graph.git.dirty ? " · dirty snapshot" : ""}
              </span>
            </div>
            <span className="architecture-snapshot-metadata">
              {new Date(graph.created_at).toLocaleString()} · {graph.completeness.status} projection
            </span>
          </div>
          <div className="stat-grid">
            <StatCard
              detail="Evidence-derived semantic services"
              icon="architecture"
              label="Services"
              value={graph.counts.services}
            />
            <StatCard
              detail="Datastores and durable resources"
              icon="database"
              label="Data stores"
              value={graph.counts.datastores}
            />
            <StatCard
              detail="Repository files in immutable snapshot"
              icon="files"
              label="Files"
              value={graph.totals.files}
            />
            <StatCard
              detail="Bounded semantic relations in view"
              icon="trace"
              label="Relations"
              value={visibleGraph.relations.length}
            />
          </div>
          {loading ? <output className="loading-state">Refreshing topology...</output> : null}
          <ArchitectureWorkspace
            inspector={
              <ComponentInspector
                component={selected}
                detail={detail}
                error={detailError}
                loading={detailLoading}
                onFocus={focusComponent}
                onOpenFinding={(id) => void openFinding(id)}
              />
            }
            layers={
              <div className="architecture-layer-stack">
                <LayerRail disabled={loading} layers={layers} onToggle={toggleComponentKind} />
                <Panel className="architecture-repository-summary">
                  <h2>Repository summary</h2>
                  <dl>
                    <div>
                      <dt className="architecture-summary-label">Files analyzed</dt>
                      <dd className="architecture-summary-value">
                        {graph.totals.files.toLocaleString()}
                      </dd>
                    </div>
                    <div>
                      <dt className="architecture-summary-label">Lines of code</dt>
                      <dd className="architecture-summary-value">
                        {graph.totals.lines.toLocaleString()}
                      </dd>
                    </div>
                  </dl>
                  <h3>Languages</h3>
                  {Object.keys(graph.totals.languages).length ? (
                    <ul>
                      {Object.entries(graph.totals.languages)
                        .sort((left, right) => right[1] - left[1])
                        .map(([language, count]) => (
                          <li key={language}>
                            <span className="architecture-summary-label">{language}</span>
                            <strong className="architecture-summary-value">
                              {count.toLocaleString()} files
                            </strong>
                          </li>
                        ))}
                    </ul>
                  ) : (
                    <p className="architecture-muted">No language totals are available.</p>
                  )}
                </Panel>
              </div>
            }
            textAlternative={
              <ArchitectureTextAlternative
                graph={visibleGraph}
                onSelectComponent={selectComponent}
                selectedComponentId={selectedId || null}
              />
            }
            topology={
              <TopologyViewport
                activeComponentKinds={componentKinds}
                graph={visibleGraph}
                onFocusComponent={focusComponent}
                onSelectComponent={selectComponent}
                selectedComponentId={selectedId || null}
              />
            }
            trace={traceControls}
          />
        </>
      ) : (
        <output className="loading-state">Loading semantic architecture...</output>
      )}
      <FindingEvidenceDrawer
        error={findingError}
        exportHref={
          finding ? `/api/v1/findings/${encodeURIComponent(finding.finding_id)}/export` : undefined
        }
        finding={finding}
        loading={findingLoading}
        openEvidenceHref={
          finding ? `/findings?finding=${encodeURIComponent(finding.finding_id)}` : undefined
        }
        onClose={() => {
          setFinding(null);
          setFindingError(null);
        }}
      />
    </div>
  );
}
