import type { ArchitectureComparison } from "./model";

export interface ArchitectureComparisonPanelProps {
  comparison: ArchitectureComparison | null;
  currentLabel: string;
  baselineLabel: string;
  loading?: boolean;
  error?: string | null;
}

function ChangeList({
  title,
  items,
}: {
  title: string;
  items: Array<{ key: string; label: string }>;
}) {
  return (
    <section>
      <h3>
        {title} <span>{items.length}</span>
      </h3>
      {items.length ? (
        <ul>
          {items.slice(0, 8).map((item) => (
            <li key={item.key}>{item.label}</li>
          ))}
        </ul>
      ) : (
        <p className="architecture-comparison-empty">None in this bounded projection.</p>
      )}
      {items.length > 8 ? (
        <small className="architecture-comparison-more">{items.length - 8} more not shown.</small>
      ) : null}
    </section>
  );
}

export function ArchitectureComparisonPanel({
  comparison,
  currentLabel,
  baselineLabel,
  loading = false,
  error = null,
}: ArchitectureComparisonPanelProps) {
  return (
    <section
      aria-labelledby="architecture-comparison-heading"
      className="panel architecture-comparison"
    >
      <header>
        <div>
          <h2 id="architecture-comparison-heading">Snapshot comparison</h2>
          <p className="architecture-comparison-meta">
            {baselineLabel} → {currentLabel}
          </p>
        </div>
        <span className="architecture-comparison-meta">Stable semantic keys</span>
      </header>
      {loading ? <output>Comparing bounded snapshot projections...</output> : null}
      {error ? (
        <p className="architecture-inline-error" role="alert">
          {error}
        </p>
      ) : null}
      {comparison ? (
        <>
          <p className="architecture-comparison-notice">
            {comparison.incomplete
              ? "At least one snapshot projection is partial or truncated. Counts describe only the matched bounded evidence returned by the API."
              : "Counts compare matched bounded semantic projections; they are not a raw repository diff."}
          </p>
          <div className="architecture-comparison-grid">
            <ChangeList
              items={comparison.addedComponents.map((component) => ({
                key: component.stable_key,
                label: component.name,
              }))}
              title="Components added"
            />
            <ChangeList
              items={comparison.removedComponents.map((component) => ({
                key: component.stable_key,
                label: component.name,
              }))}
              title="Components removed"
            />
            <ChangeList
              items={comparison.changedComponents.map((change) => ({
                key: change.stableKey,
                label: `${change.after.name}: ${change.fields.join(", ")}`,
              }))}
              title="Components changed"
            />
            <ChangeList
              items={comparison.addedRelations.map((relation) => ({
                key: relation.stable_key,
                label: relation.stable_key,
              }))}
              title="Relations added"
            />
            <ChangeList
              items={comparison.removedRelations.map((relation) => ({
                key: relation.stable_key,
                label: relation.stable_key,
              }))}
              title="Relations removed"
            />
          </div>
        </>
      ) : null}
    </section>
  );
}
