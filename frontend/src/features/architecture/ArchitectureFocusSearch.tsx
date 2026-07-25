import { useMemo } from "react";
import type { SemanticComponent } from "../../api/contracts";

export interface ArchitectureFocusSearchProps {
  components: SemanticComponent[];
  value: string;
  appliedValue: string;
  disabled?: boolean;
  onChange: (value: string) => void;
  onApply: (value: string) => void;
  onSelect: (component: SemanticComponent) => void;
}

export function ArchitectureFocusSearch({
  components,
  value,
  appliedValue,
  disabled = false,
  onChange,
  onApply,
  onSelect,
}: ArchitectureFocusSearchProps) {
  const normalized = value.trim().toLocaleLowerCase();
  const matches = useMemo(() => {
    if (!normalized) return [];
    return components
      .filter((component) =>
        [component.name, component.stable_key, component.component_kind].some((candidate) =>
          candidate.toLocaleLowerCase().includes(normalized),
        ),
      )
      .slice(0, 6);
  }, [components, normalized]);

  return (
    <div className="architecture-focus-search">
      <label>
        <span className="architecture-control-label">Search and focus components</span>
        <input
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Partial name, stable key, or component kind"
          value={value}
        />
      </label>
      <button
        className="button button-secondary"
        disabled={disabled || !value.trim() || value.trim() === appliedValue}
        onClick={() => onApply(value.trim())}
        type="button"
      >
        Apply exact focus
      </button>
      {normalized ? (
        <section aria-label="Matching components" className="architecture-focus-results">
          <span>{matches.length} local matches</span>
          {matches.length ? (
            <ul>
              {matches.map((component) => (
                <li key={component.component_id}>
                  <button onClick={() => onSelect(component)} type="button">
                    <strong>{component.name}</strong>
                    <small>
                      {component.component_kind.replaceAll("_", " ")} · {component.stable_key}
                    </small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p>No component in the current bounded projection matches locally.</p>
          )}
        </section>
      ) : null}
    </div>
  );
}
