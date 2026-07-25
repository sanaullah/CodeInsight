import type { ArchitectureLayer } from "./model";

export interface LayerRailProps {
  layers: ArchitectureLayer[];
  onToggle: (layerId: string, active: boolean) => void;
  disabled?: boolean;
  heading?: string;
}

export function LayerRail({
  layers,
  onToggle,
  disabled = false,
  heading = "Architecture layers",
}: LayerRailProps) {
  return (
    <aside aria-label={heading} className="architecture-layer-rail">
      <div className="architecture-section-heading">
        <h2>{heading}</h2>
        <span>{layers.filter((layer) => layer.active).length} visible</span>
      </div>
      {layers.length ? (
        <fieldset disabled={disabled}>
          <legend className="architecture-visually-hidden">Toggle component layers</legend>
          {layers.map((layer) => (
            <label className="architecture-layer-option" key={layer.id}>
              <input
                checked={layer.active}
                onChange={(event) => onToggle(layer.id, event.target.checked)}
                type="checkbox"
              />
              <span aria-hidden="true" className={`architecture-layer-dot kind-${layer.id}`} />
              <span className="architecture-layer-copy">
                <strong>{layer.label}</strong>
                {layer.description ? <small>{layer.description}</small> : null}
              </span>
              <span className="architecture-layer-count">{layer.count}</span>
            </label>
          ))}
        </fieldset>
      ) : (
        <p className="architecture-muted">No semantic layers are available.</p>
      )}
    </aside>
  );
}
