import type { HTMLAttributes, ReactNode } from "react";
import { formatLabel } from "../format";
import { Icon } from "./Icon";

export function Panel({ className = "", children, ...props }: HTMLAttributes<HTMLElement>) {
  return (
    <section className={`panel ${className}`.trim()} {...props}>
      {children}
    </section>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge status-${status}`}>{formatLabel(status)}</span>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function StatCard({
  icon,
  label,
  value,
  detail,
  tone = "teal",
}: {
  icon: Parameters<typeof Icon>[0]["name"];
  label: string;
  value: ReactNode;
  detail: string;
  tone?: "teal" | "amber" | "coral" | "blue";
}) {
  return (
    <article className="stat-card">
      <span className={`stat-icon tone-${tone}`}>
        <Icon name={icon} />
      </span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Icon name="trace" size={30} />
      </span>
      <h2>{title}</h2>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="notice notice-error" role="alert">
      <Icon name="alert" />
      <span>{message}</span>
      {retry ? (
        <button className="text-button" onClick={retry} type="button">
          Try again
        </button>
      ) : null}
    </div>
  );
}
