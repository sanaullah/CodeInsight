import type { OperationalState } from "../hooks/useOperationalState";
import { Icon } from "./Icon";

export function OperationalStatus({ state }: { state: OperationalState }) {
  const { health, capabilities, error, loading, refresh } = state;
  const online = Boolean(health) && !error;
  const buildLabel = health?.runtime.build_commit
    ? health.runtime.build_commit.slice(0, 10)
    : "local build";

  return (
    <div className="operational-status">
      <div className="status-summary" aria-live="polite">
        <span className={`status-dot ${online ? "online" : loading ? "loading" : "offline"}`} />
        <span>{online ? "API ready" : loading ? "Connecting" : "API unavailable"}</span>
        {health ? <span className="status-version">{health.version}</span> : null}
      </div>
      <details className="diagnostics">
        <summary>Runtime details</summary>
        <div className="diagnostic-panel">
          <div className="diagnostic-heading">
            <div>
              <strong>Local runtime</strong>
              <span>
                {health?.version ?? "Version unavailable"} · {buildLabel}
              </span>
            </div>
            {error ? (
              <button className="text-button" onClick={refresh} type="button">
                Retry
              </button>
            ) : null}
          </div>
          {error ? (
            <p className="diagnostic-error" role="alert">
              {error}
            </p>
          ) : null}
          <dl className="service-list">
            <Service
              icon="server"
              label="API"
              status={online ? "Ready" : "Unavailable"}
              value={health ? health.runtime.application_server : "FastAPI"}
            />
            <Service
              icon="database"
              label="Durable ledger"
              status={online ? "Ready" : "Unknown"}
              value={
                health
                  ? `${health.runtime.database_engine} / ${health.runtime.database_journal_mode} / schema ${health.runtime.database_schema_version}`
                  : "SQLite / WAL"
              }
            />
            <Service
              icon="files"
              label="Artifacts"
              status={online ? "On demand" : "Unknown"}
              value={health?.runtime.artifact_store ?? "Local filesystem"}
            />
            <Service
              icon="model"
              label="Model provider"
              status={capabilities?.model_provider_configured ? "Configured" : "Index-only"}
              value={
                capabilities?.model_provider_configured
                  ? "OpenAI-compatible endpoint"
                  : "No model endpoint required"
              }
            />
            <Service
              icon="trace"
              label="Langfuse"
              status={capabilities?.langfuse_enabled ? "Enabled" : "Optional / off"}
              value="Non-blocking trace export"
            />
          </dl>
          <p className="runtime-note">
            {health?.runtime.environment_manager ?? "uv"}-managed · read-only analysis
            {health ? ` · ${health.active_analyses}/${health.max_concurrent_analyses} active` : ""}
          </p>
        </div>
      </details>
    </div>
  );
}

function Service({
  icon,
  label,
  value,
  status,
}: {
  icon: Parameters<typeof Icon>[0]["name"];
  label: string;
  value: string;
  status: string;
}) {
  return (
    <div>
      <dt>
        <Icon name={icon} />
        <span>{label}</span>
      </dt>
      <dd>
        <span>{value}</span>
        <strong>{status}</strong>
      </dd>
    </div>
  );
}
