import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type {
  ArchitectureGraph,
  ArchitectureNode,
  ArchitectureTrace,
  SnapshotSummary,
} from "../api/contracts";
import { EmptyState, ErrorNotice, PageHeader, Panel, StatCard } from "../components/primitives";
import { shortId } from "../format";

export function ArchitecturePage() {
  const initial = useMemo(() => new URLSearchParams(window.location.search), []);
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [snapshotId, setSnapshotId] = useState(initial.get("snapshot") ?? "");
  const [edgeKind, setEdgeKind] = useState(initial.get("edge_kind") ?? "");
  const [focus, setFocus] = useState(initial.get("focus") ?? "");
  const [graph, setGraph] = useState<ArchitectureGraph | null>(null);
  const [selected, setSelected] = useState<ArchitectureNode | null>(null);
  const [traceSource, setTraceSource] = useState("");
  const [traceTarget, setTraceTarget] = useState("");
  const [trace, setTrace] = useState<ArchitectureTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    if (!snapshotId) return;
    const controller = new AbortController();
    const params = new URLSearchParams({ limit: "250", depth: "1" });
    if (edgeKind) params.set("edge_kind", edgeKind);
    if (focus) params.set("focus", focus);
    const route = new URLSearchParams();
    route.set("snapshot", snapshotId);
    if (edgeKind) route.set("edge_kind", edgeKind);
    if (focus) route.set("focus", focus);
    window.history.replaceState({}, "", `/architecture?${route}`);
    apiClient
      .architecture(snapshotId, params, controller.signal)
      .then((value) => {
        setGraph(value);
        setSelected(null);
        setError(null);
      })
      .catch((reason) =>
        setError(reason instanceof Error ? reason.message : "Unable to load architecture"),
      );
    return () => controller.abort();
  }, [snapshotId, edgeKind, focus]);

  async function loadTrace() {
    if (!snapshotId || !traceSource || !traceTarget) return;
    try {
      setTrace(await apiClient.trace(snapshotId, traceSource, traceTarget));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load trace");
    }
  }

  return (
    <div className="page">
      <PageHeader
        description="Explore bounded file, import, and call relationships derived from the immutable repository index."
        eyebrow="Repository evidence"
        title="Architecture explorer"
      />
      {error ? <ErrorNotice message={error} /> : null}
      <Panel className="architecture-controls">
        <label>
          <span>Repository snapshot</span>
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
          <span>Relationship layer</span>
          <select onChange={(event) => setEdgeKind(event.target.value)} value={edgeKind}>
            <option value="">Imports and calls</option>
            <option value="imports">Imports</option>
            <option value="calls">Calls</option>
          </select>
        </label>
        <label>
          <span>Focus file</span>
          <input
            onChange={(event) => setFocus(event.target.value)}
            placeholder="Exact repository-relative path"
            value={focus}
          />
        </label>
      </Panel>
      {!snapshotId ? (
        <EmptyState
          description="Run a review first to create an immutable repository snapshot."
          title="No repository snapshot selected"
        />
      ) : graph ? (
        <>
          <div className="stat-grid">
            <StatCard
              detail="Persisted repository files"
              icon="files"
              label="Files"
              value={graph.summary.file_count}
            />
            <StatCard
              detail="Bounded visible nodes"
              icon="architecture"
              label="Visible"
              value={graph.nodes.length}
            />
            <StatCard
              detail="Index-derived relationships"
              icon="trace"
              label="Edges"
              value={graph.edges.length}
            />
            <StatCard
              detail="Files changed in snapshot metadata"
              icon="history"
              label="Changed"
              value={graph.summary.changed_paths.length}
            />
          </div>
          <div className="architecture-layout">
            <Panel className="graph-panel">
              <div className="section-heading">
                <h2>Bounded dependency map</h2>
                <span>
                  {graph.truncated ? "Node limit applied" : "Complete current projection"}
                </span>
              </div>
              <svg
                aria-label="Repository relationship diagram"
                className="architecture-map"
                role="img"
                viewBox={`0 0 900 ${Math.max(360, Math.ceil(graph.nodes.length / 4) * 120)}`}
              >
                {graph.edges.map((edge) => {
                  const source = graph.nodes.findIndex((node) => node.node_id === edge.source_id);
                  const target = graph.nodes.findIndex((node) => node.node_id === edge.target_id);
                  if (source < 0 || target < 0) return null;
                  const x1 = 110 + (source % 4) * 220;
                  const y1 = 60 + Math.floor(source / 4) * 120;
                  const x2 = 110 + (target % 4) * 220;
                  const y2 = 60 + Math.floor(target / 4) * 120;
                  return <line key={edge.edge_id} x1={x1} x2={x2} y1={y1} y2={y2} />;
                })}
                {graph.nodes.map((node, index) => {
                  const x = 25 + (index % 4) * 220;
                  const y = 30 + Math.floor(index / 4) * 120;
                  return (
                    <g key={node.node_id}>
                      <rect
                        className={node.findings.length ? "graph-node has-findings" : "graph-node"}
                        height="62"
                        rx="9"
                        width="170"
                        x={x}
                        y={y}
                      />
                      <text x={x + 10} y={y + 26}>
                        {node.label.length > 22 ? `…${node.label.slice(-21)}` : node.label}
                      </text>
                      <text className="graph-node-meta" x={x + 10} y={y + 45}>
                        {node.language ?? "unknown"} · {node.line_count} lines
                      </text>
                    </g>
                  );
                })}
              </svg>
              <details className="graph-alternative">
                <summary>Text alternative: nodes and relationships</summary>
                <ul>
                  {graph.nodes.map((node) => (
                    <li key={node.node_id}>
                      <button onClick={() => setSelected(node)} type="button">
                        {node.label}
                      </button>{" "}
                      — {node.language ?? "unknown"}, {node.findings.length} correlated findings
                    </li>
                  ))}
                </ul>
                <ol>
                  {graph.edges.map((edge) => (
                    <li key={edge.edge_id}>
                      {graph.nodes.find((node) => node.node_id === edge.source_id)?.label} →{" "}
                      {graph.nodes.find((node) => node.node_id === edge.target_id)?.label} (
                      {edge.edge_kind}, {Math.round(edge.confidence * 100)}%)
                    </li>
                  ))}
                </ol>
              </details>
            </Panel>
            <Panel className="architecture-detail">
              <h2>{selected ? selected.label : "Node detail"}</h2>
              {selected ? (
                <dl>
                  <div>
                    <dt>Language</dt>
                    <dd>{selected.language ?? "Unknown"}</dd>
                  </div>
                  <div>
                    <dt>Classification</dt>
                    <dd>{selected.classification}</dd>
                  </div>
                  <div>
                    <dt>Support</dt>
                    <dd>{selected.support_tier}</dd>
                  </div>
                  <div>
                    <dt>Derivation</dt>
                    <dd>{selected.derivation}</dd>
                  </div>
                  <div>
                    <dt>Findings</dt>
                    <dd>{selected.findings.length}</dd>
                  </div>
                </dl>
              ) : (
                <p className="muted-copy">Select a node to inspect persisted index metadata.</p>
              )}
              <h3>Evidence trace</h3>
              <select
                aria-label="Trace source"
                onChange={(event) => setTraceSource(event.target.value)}
                value={traceSource}
              >
                <option value="">Source node</option>
                {graph.nodes.map((node) => (
                  <option key={node.node_id} value={node.node_id}>
                    {node.label}
                  </option>
                ))}
              </select>
              <select
                aria-label="Trace target"
                onChange={(event) => setTraceTarget(event.target.value)}
                value={traceTarget}
              >
                <option value="">Target node</option>
                {graph.nodes.map((node) => (
                  <option key={node.node_id} value={node.node_id}>
                    {node.label}
                  </option>
                ))}
              </select>
              <button
                className="button button-secondary"
                disabled={!traceSource || !traceTarget}
                onClick={loadTrace}
                type="button"
              >
                Trace path
              </button>
              {trace ? (
                <output className="trace-result">
                  {trace.found
                    ? trace.nodes.map((node) => node.label).join(" → ")
                    : `No directed path found within ${trace.max_hops} hops.`}
                </output>
              ) : null}
            </Panel>
          </div>
        </>
      ) : (
        <div className="loading-state">Loading architecture…</div>
      )}
    </div>
  );
}
