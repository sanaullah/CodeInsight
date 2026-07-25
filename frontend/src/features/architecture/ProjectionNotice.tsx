import type { ArchitectureProjectionStatus } from "./model";

export interface ProjectionNoticeProps {
  status: ArchitectureProjectionStatus;
  itemLabel?: string;
}

const statusCopy: Record<ArchitectureProjectionStatus, { title: string; detail: string }> = {
  complete: {
    title: "Complete bounded projection",
    detail: "All results within the current server-side bounds are shown.",
  },
  partial: {
    title: "Partial evidence",
    detail: "Some repository facts are unsupported or could not be derived deterministically.",
  },
  truncated: {
    title: "Bounded result",
    detail:
      "The server limit was reached. Focus or filter the view to inspect a smaller neighborhood.",
  },
  unsupported: {
    title: "Unsupported evidence",
    detail: "The current extractor cannot make this claim from available repository evidence.",
  },
  no_path: {
    title: "No supported path",
    detail: "No directed semantic path was found within the requested hop limit.",
  },
};

export function ProjectionNotice({ status, itemLabel = "architecture" }: ProjectionNoticeProps) {
  if (status === "complete") return null;
  const copy = statusCopy[status];
  return (
    <output className={`architecture-notice architecture-notice-${status}`}>
      <strong>{copy.title}</strong>
      <span>
        {copy.detail} This {itemLabel} view does not infer missing runtime behavior.
      </span>
    </output>
  );
}
