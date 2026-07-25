import type { RoleRecord, TaskRecord } from "../api/contracts";
import { formatLabel } from "../format";
import { Icon } from "./Icon";
import { StatusBadge } from "./primitives";

export function SpecialistCard({ role, task }: { role: RoleRecord; task: TaskRecord | undefined }) {
  return (
    <article className="specialist-card">
      <div className="specialist-heading">
        <span className="specialist-icon">
          <Icon name="users" />
        </span>
        <div>
          <strong>{role.name}</strong>
          <span>{formatLabel(role.model_policy)} model policy</span>
        </div>
        <StatusBadge status={task?.status ?? "planned"} />
      </div>
      <p>{role.mission}</p>
      <div className="coverage-tags">
        {role.coverage_targets.map((target) => (
          <span key={target}>{formatLabel(target)}</span>
        ))}
      </div>
      {task?.error?.message ? <small className="task-error">{task.error.message}</small> : null}
    </article>
  );
}
