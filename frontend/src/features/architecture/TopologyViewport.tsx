import { useMemo, useState } from "react";
import type { SemanticArchitectureGraph, SemanticComponent } from "../../api/contracts";
import {
  componentId,
  componentLabel,
  layoutComponents,
  projectionStatus,
  relationIsRuntimeFlow,
  relationLabel,
} from "./model";
import { ProjectionNotice } from "./ProjectionNotice";

export interface TopologyViewportProps {
  graph: SemanticArchitectureGraph;
  selectedComponentId: string | null;
  onSelectComponent: (component: SemanticComponent) => void;
  onFocusComponent?: (component: SemanticComponent) => void;
  activeComponentKinds?: ReadonlySet<string>;
  heading?: string;
}

const MIN_ZOOM = 0.6;
const MAX_ZOOM = 1.6;

export function TopologyViewport({
  graph,
  selectedComponentId,
  onSelectComponent,
  onFocusComponent,
  activeComponentKinds,
  heading = "Semantic topology",
}: TopologyViewportProps) {
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const visibleComponents = useMemo(
    () =>
      activeComponentKinds
        ? graph.components.filter((component) => activeComponentKinds.has(component.component_kind))
        : graph.components,
    [activeComponentKinds, graph.components],
  );
  const visibleIds = useMemo(
    () => new Set(visibleComponents.map(componentId)),
    [visibleComponents],
  );
  const positions = useMemo(
    () => layoutComponents(visibleComponents, visibleComponents.length > 8 ? 4 : 3),
    [visibleComponents],
  );
  const visibleRelations = useMemo(
    () =>
      graph.relations.filter(
        (relation) =>
          visibleIds.has(relation.source_component_id) &&
          visibleIds.has(relation.target_component_id),
      ),
    [graph.relations, visibleIds],
  );
  const findingCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const finding of graph.findings) {
      counts.set(finding.component_id, (counts.get(finding.component_id) ?? 0) + 1);
    }
    return counts;
  }, [graph.findings]);
  const rowCount = Math.max(
    1,
    Math.ceil(visibleComponents.length / (visibleComponents.length > 8 ? 4 : 3)),
  );
  const contentHeight = Math.max(420, rowCount * 132 + 50);
  const contentWidth = visibleComponents.length > 8 ? 900 : 680;

  function changeZoom(delta: number) {
    setZoom((current) =>
      Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((current + delta).toFixed(1)))),
    );
  }

  return (
    <section aria-labelledby="semantic-topology-heading" className="architecture-topology">
      <div className="architecture-section-heading">
        <div>
          <h2 id="semantic-topology-heading">{heading}</h2>
          <span>
            {visibleComponents.length} components, {visibleRelations.length} typed relations
          </span>
        </div>
        <fieldset className="architecture-viewport-controls">
          <legend className="architecture-visually-hidden">Topology view controls</legend>
          <button
            aria-label="Zoom out"
            disabled={zoom <= MIN_ZOOM}
            onClick={() => changeZoom(-0.1)}
            type="button"
          >
            -
          </button>
          <output aria-live="polite">{Math.round(zoom * 100)}%</output>
          <button
            aria-label="Zoom in"
            disabled={zoom >= MAX_ZOOM}
            onClick={() => changeZoom(0.1)}
            type="button"
          >
            +
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setOffset({ x: 0, y: 0 });
            }}
            type="button"
          >
            Reset
          </button>
        </fieldset>
      </div>
      <ProjectionNotice status={projectionStatus(graph)} />
      {graph.boundaries.length ? (
        <ul aria-label="Visible architecture boundaries" className="architecture-boundary-strip">
          {graph.boundaries.map((boundary) => (
            <li key={boundary.boundary_id}>
              <strong>{boundary.name}</strong>
              <span>
                {boundary.boundary_kind.replaceAll("_", " ")} · {boundary.component_ids.length}{" "}
                component{boundary.component_ids.length === 1 ? "" : "s"}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      <div
        aria-describedby="topology-instructions"
        className="architecture-viewport"
        data-testid="topology-viewport"
      >
        <p className="architecture-visually-hidden" id="topology-instructions">
          The diagram is supplementary. Use the always-visible text alternative for a complete
          keyboard-readable list of components and relations.
        </p>
        {visibleComponents.length ? (
          <div
            className="architecture-canvas"
            style={{
              height: contentHeight,
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              width: contentWidth,
            }}
          >
            <svg
              aria-hidden="true"
              className="architecture-edge-layer"
              height={contentHeight}
              viewBox={`0 0 ${contentWidth} ${contentHeight}`}
              width={contentWidth}
            >
              <defs>
                <marker
                  id="architecture-arrow"
                  markerHeight="6"
                  markerWidth="6"
                  orient="auto"
                  refX="5"
                  refY="3"
                >
                  <path d="M0,0 L0,6 L6,3 z" />
                </marker>
              </defs>
              {visibleRelations.map((relation) => {
                const source = positions.get(relation.source_component_id);
                const target = positions.get(relation.target_component_id);
                if (!source || !target) return null;
                const runtime = relationIsRuntimeFlow(relation.relation_kind);
                return (
                  <g key={relation.relation_id}>
                    <line
                      className={runtime ? "runtime-relation" : "static-relation"}
                      markerEnd="url(#architecture-arrow)"
                      x1={source.x + 164}
                      x2={target.x}
                      y1={source.y + 38}
                      y2={target.y + 38}
                    />
                    <title>{relationLabel(relation.relation_kind)}</title>
                  </g>
                );
              })}
            </svg>
            {visibleComponents.map((component) => {
              const position = positions.get(componentId(component));
              if (!position) return null;
              const selected = selectedComponentId === componentId(component);
              const findingCount = findingCounts.get(componentId(component)) ?? 0;
              return (
                <button
                  aria-pressed={selected}
                  className={`architecture-node kind-${component.component_kind}${selected ? " is-selected" : ""}`}
                  key={componentId(component)}
                  onClick={() => onSelectComponent(component)}
                  onDoubleClick={() => onFocusComponent?.(component)}
                  style={{ left: position.x, top: position.y }}
                  type="button"
                >
                  <span className="architecture-node-kind">
                    {component.component_kind.replaceAll("_", " ")}
                  </span>
                  <strong>{componentLabel(component)}</strong>
                  <small>
                    {component.completeness} / {Math.round(component.confidence * 100)}%
                  </small>
                  {findingCount ? (
                    <>
                      <span aria-hidden="true" className="architecture-node-findings">
                        {findingCount}
                      </span>
                      <span className="architecture-visually-hidden">
                        {findingCount} correlated finding{findingCount === 1 ? "" : "s"}
                      </span>
                    </>
                  ) : null}
                </button>
              );
            })}
          </div>
        ) : (
          <div className="architecture-empty">
            <strong>No components match the active layers.</strong>
            <span>Enable a layer or adjust the server-side filters.</span>
          </div>
        )}
      </div>
      <fieldset className="architecture-pan-controls">
        <legend className="architecture-visually-hidden">Pan topology</legend>
        <button
          aria-label="Pan left"
          onClick={() => setOffset((current) => ({ ...current, x: current.x + 80 }))}
          type="button"
        >
          Left
        </button>
        <button
          aria-label="Pan up"
          onClick={() => setOffset((current) => ({ ...current, y: current.y + 80 }))}
          type="button"
        >
          Up
        </button>
        <button
          aria-label="Pan down"
          onClick={() => setOffset((current) => ({ ...current, y: current.y - 80 }))}
          type="button"
        >
          Down
        </button>
        <button
          aria-label="Pan right"
          onClick={() => setOffset((current) => ({ ...current, x: current.x - 80 }))}
          type="button"
        >
          Right
        </button>
      </fieldset>
    </section>
  );
}
