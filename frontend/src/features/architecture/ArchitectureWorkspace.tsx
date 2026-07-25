import type { ReactNode } from "react";

export interface ArchitectureWorkspaceProps {
  layers: ReactNode;
  topology: ReactNode;
  inspector: ReactNode;
  trace: ReactNode;
  textAlternative: ReactNode;
}

export function ArchitectureWorkspace({
  layers,
  topology,
  inspector,
  trace,
  textAlternative,
}: ArchitectureWorkspaceProps) {
  return (
    <div className="architecture-semantic-workspace">
      <div className="architecture-primary-grid">
        {layers}
        {topology}
        {inspector}
      </div>
      {trace}
      {textAlternative}
    </div>
  );
}
