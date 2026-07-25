import { useMemo } from "react";
import type { SemanticComponent, SemanticTrace } from "../../api/contracts";
import {
  componentId,
  componentLabel,
  componentMap,
  relationIsRuntimeFlow,
  relationLabel,
  traceStatus,
} from "./model";
import { ProjectionNotice } from "./ProjectionNotice";

export interface TraceRailProps {
  trace: SemanticTrace | null;
  loading?: boolean;
  error?: string | null;
  sourceLabel?: string;
  targetLabel?: string;
  onSelectComponent?: (component: SemanticComponent) => void;
}

export function TraceRail({
  trace,
  loading = false,
  error = null,
  sourceLabel,
  targetLabel,
  onSelectComponent,
}: TraceRailProps) {
  const components = useMemo(() => componentMap(trace?.components ?? []), [trace?.components]);
  const relationsByPair = useMemo(
    () =>
      new Map(
        (trace?.relations ?? []).map((relation) => [
          `${relation.source_component_id}\u0000${relation.target_component_id}`,
          relation,
        ]),
      ),
    [trace?.relations],
  );

  return (
    <section aria-labelledby="architecture-trace-heading" className="architecture-trace-rail">
      <div className="architecture-section-heading">
        <div>
          <h2 id="architecture-trace-heading">Evidence trace</h2>
          <span>
            {sourceLabel && targetLabel
              ? `${sourceLabel} to ${targetLabel}`
              : "Select endpoints to request a bounded semantic trace"}
          </span>
        </div>
      </div>
      {loading ? <output>Loading bounded evidence trace...</output> : null}
      {error ? (
        <p className="architecture-inline-error" role="alert">
          {error}
        </p>
      ) : null}
      {trace ? <ProjectionNotice itemLabel="trace" status={traceStatus(trace)} /> : null}
      {trace?.status === "complete" || trace?.status === "truncated" ? (
        trace.components.length ? (
          <ol className="architecture-trace-list">
            {trace.components.map((component, index) => {
              const next = trace.components[index + 1];
              const relation = next
                ? relationsByPair.get(`${componentId(component)}\u0000${componentId(next)}`)
                : null;
              return (
                <li key={componentId(component)}>
                  <button
                    disabled={!onSelectComponent}
                    onClick={() => onSelectComponent?.(component)}
                    type="button"
                  >
                    <span>{index + 1}</span>
                    <strong>{componentLabel(component)}</strong>
                    <small>{component.component_kind.replaceAll("_", " ")}</small>
                  </button>
                  {relation ? (
                    <div className="architecture-trace-relation">
                      <span>{relationLabel(relation.relation_kind)}</span>
                      <small>
                        {relation.transport ? `${relation.transport} / ` : ""}
                        {Math.round(relation.confidence * 100)}% confidence
                      </small>
                      {!relationIsRuntimeFlow(relation.relation_kind) ? (
                        <em>Static relationship, not runtime data flow</em>
                      ) : null}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="architecture-muted">The trace contains no supported components.</p>
        )
      ) : null}
      {trace?.provenance?.length ? (
        <details className="architecture-trace-provenance">
          <summary>Trace evidence ({trace.provenance.length})</summary>
          <ul>
            {trace.provenance.map((record) => (
              <li key={record.provenance_id}>
                <strong>
                  {record.relative_path}:{record.start_line}-{record.end_line}
                </strong>
                <span>
                  {record.derivation.replaceAll("_", " ")} / {Math.round(record.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {trace && trace.status === "complete" && trace.components.length > 0 ? (
        <p className="architecture-visually-hidden">
          Trace contains {components.size} unique semantic components.
        </p>
      ) : null}
    </section>
  );
}
