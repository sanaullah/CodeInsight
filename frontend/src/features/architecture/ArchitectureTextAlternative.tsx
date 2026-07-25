import { useMemo } from "react";
import type { SemanticArchitectureGraph, SemanticComponent } from "../../api/contracts";
import {
  componentId,
  componentLabel,
  componentMap,
  projectionStatus,
  relationIsRuntimeFlow,
  relationLabel,
} from "./model";
import { ProjectionNotice } from "./ProjectionNotice";

export interface ArchitectureTextAlternativeProps {
  graph: SemanticArchitectureGraph;
  selectedComponentId: string | null;
  onSelectComponent: (component: SemanticComponent) => void;
}

export function ArchitectureTextAlternative({
  graph,
  selectedComponentId,
  onSelectComponent,
}: ArchitectureTextAlternativeProps) {
  const components = useMemo(() => componentMap(graph.components), [graph.components]);
  return (
    <section aria-labelledby="architecture-text-heading" className="architecture-text-alternative">
      <div className="architecture-section-heading">
        <div>
          <h2 id="architecture-text-heading">Text architecture view</h2>
          <span>Complete keyboard-readable alternative to the diagram</span>
        </div>
      </div>
      <ProjectionNotice itemLabel="text alternative" status={projectionStatus(graph)} />
      {graph.boundaries.length ? (
        <section className="architecture-boundary-alternative">
          <h3>Boundaries ({graph.boundaries.length})</h3>
          <ul>
            {graph.boundaries.map((boundary) => (
              <li key={boundary.boundary_id}>
                <strong>{boundary.name}</strong>
                <span>
                  {boundary.boundary_kind.replaceAll("_", " ")} containing{" "}
                  {boundary.component_ids.length} visible component
                  {boundary.component_ids.length === 1 ? "" : "s"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <div className="architecture-text-columns">
        <section>
          <h3>Components ({graph.components.length})</h3>
          {graph.components.length ? (
            <ul className="architecture-component-list">
              {graph.components.map((component) => (
                <li key={componentId(component)}>
                  <button
                    aria-current={
                      selectedComponentId === componentId(component) ? "true" : undefined
                    }
                    onClick={() => onSelectComponent(component)}
                    type="button"
                  >
                    <strong>{componentLabel(component)}</strong>
                    <span>
                      {component.component_kind.replaceAll("_", " ")} / {component.completeness} /{" "}
                      {Math.round(component.confidence * 100)}%
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="architecture-muted">No semantic components are available.</p>
          )}
        </section>
        <section>
          <h3>Typed relations ({graph.relations.length})</h3>
          {graph.relations.length ? (
            <ol className="architecture-relation-list">
              {graph.relations.map((relation) => {
                const source = components.get(relation.source_component_id);
                const target = components.get(relation.target_component_id);
                const runtime = relationIsRuntimeFlow(relation.relation_kind);
                return (
                  <li key={relation.relation_id}>
                    <strong>
                      {source ? componentLabel(source) : relation.source_component_id}
                    </strong>
                    <span aria-hidden="true"> -&gt; </span>
                    <strong>
                      {target ? componentLabel(target) : relation.target_component_id}
                    </strong>
                    <span>
                      {relationLabel(relation.relation_kind)}
                      {relation.transport ? ` via ${relation.transport}` : ""}
                      {" / "}
                      {Math.round(relation.confidence * 100)}% confidence
                    </span>
                    {!runtime ? (
                      <small>
                        Static repository relationship; not evidence of runtime data flow.
                      </small>
                    ) : null}
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="architecture-muted">No typed relations match the current bounds.</p>
          )}
        </section>
      </div>
    </section>
  );
}
