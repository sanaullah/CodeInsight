import type { SemanticComponent, SemanticComponentDetail } from "../../api/contracts";
import { componentLabel } from "./model";

export interface ComponentInspectorProps {
  component: SemanticComponent | null;
  detail: SemanticComponentDetail | null;
  loading?: boolean;
  error?: string | null;
  onFocus?: (component: SemanticComponent) => void;
  onOpenFinding?: (findingId: string) => void;
}

function metadataEntries(metadata: Record<string, unknown> | null | undefined) {
  return Object.entries(metadata ?? {}).filter(
    ([, value]) =>
      typeof value === "string" || typeof value === "number" || typeof value === "boolean",
  );
}

function metadataString(metadata: Record<string, unknown>, key: string): string | null {
  const value = metadata[key];
  return typeof value === "string" && value ? value : null;
}

export function ComponentInspector({
  component,
  detail,
  loading = false,
  error = null,
  onFocus,
  onOpenFinding,
}: ComponentInspectorProps) {
  return (
    <aside aria-label="Component inspector" className="architecture-inspector">
      <div className="architecture-section-heading">
        <h2>Inspector</h2>
        {component ? <span>{component.component_kind.replaceAll("_", " ")}</span> : null}
      </div>
      {!component ? (
        <p className="architecture-muted">
          Select a semantic component to inspect repository-derived facts and evidence.
        </p>
      ) : (
        <>
          <div className="architecture-inspector-title">
            <span className={`architecture-layer-dot kind-${component.component_kind}`} />
            <div>
              <h3>{componentLabel(component)}</h3>
              <p>{component.stable_key}</p>
            </div>
          </div>
          <dl className="architecture-fact-list">
            <div>
              <dt>Support tier</dt>
              <dd>{component.support_tier}</dd>
            </div>
            <div>
              <dt>Completeness</dt>
              <dd>{component.completeness}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{Math.round(component.confidence * 100)}%</dd>
            </div>
            {metadataEntries(component.metadata).map(([key, value]) => (
              <div key={key}>
                <dt>{key.replaceAll("_", " ")}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
          {onFocus ? (
            <button
              className="architecture-action"
              onClick={() => onFocus(component)}
              type="button"
            >
              Focus neighborhood
            </button>
          ) : null}
          {loading ? <output>Loading component evidence...</output> : null}
          {error ? (
            <p className="architecture-inline-error" role="alert">
              {error}
            </p>
          ) : null}
          {detail ? (
            <>
              <InspectorList
                empty="No file memberships were derived."
                heading="Repository membership"
                items={detail.memberships.map((membership) => ({
                  id: membership.relative_path ?? membership.file_id ?? membership.membership_id,
                  primary:
                    membership.relative_path ?? membership.file_id ?? "Unknown repository file",
                  secondary: `${membership.language ?? "unknown"} / ${
                    membership.line_count ?? "unknown"
                  } lines`,
                }))}
              />
              <InspectorList
                empty="No endpoints were derived."
                heading="Endpoints"
                items={detail.endpoints.map((endpoint) => ({
                  id: endpoint.endpoint_id,
                  primary: `${endpoint.method ?? "ANY"} ${endpoint.route}`,
                  secondary: `${
                    metadataString(endpoint.metadata, "framework") ?? "unknown framework"
                  } / ${endpoint.completeness}`,
                }))}
              />
              <InspectorList
                empty="No resources were derived."
                heading="Resources"
                items={detail.resources.map((resource) => ({
                  id: resource.resource_id,
                  primary: resource.name,
                  secondary: `${resource.resource_kind} / ${resource.completeness}`,
                }))}
              />
              <section className="architecture-inspector-section">
                <h3>Correlated findings ({detail.findings.length})</h3>
                {detail.findings.length ? (
                  <ul>
                    {detail.findings.map((finding) => (
                      <li key={finding.finding_id}>
                        <button
                          disabled={!onOpenFinding}
                          onClick={() => onOpenFinding?.(finding.finding_id)}
                          type="button"
                        >
                          <span className={`architecture-severity severity-${finding.severity}`}>
                            {finding.severity}
                          </span>
                          {finding.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="architecture-muted">No verified findings are linked.</p>
                )}
              </section>
            </>
          ) : null}
        </>
      )}
    </aside>
  );
}

interface InspectorListItem {
  id: string;
  primary: string;
  secondary: string;
}

function InspectorList({
  empty,
  heading,
  items,
}: {
  empty: string;
  heading: string;
  items: InspectorListItem[];
}) {
  return (
    <section className="architecture-inspector-section">
      <h3>
        {heading} ({items.length})
      </h3>
      {items.length ? (
        <ul>
          {items.map((item) => (
            <li key={item.id}>
              <strong>{item.primary}</strong>
              <span>{item.secondary}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="architecture-muted">{empty}</p>
      )}
    </section>
  );
}
