import { useState } from "react";
import { apiClient } from "../api/client";
import { Icon } from "../components/Icon";
import {
  EmptyState,
  ErrorNotice,
  PageHeader,
  Panel,
  StatCard,
  StatusBadge,
} from "../components/primitives";
import { SpecialistCard } from "../components/SpecialistCard";
import { StageStepper } from "../components/StageStepper";
import { formatDuration, formatLabel, formatTokens, shortId } from "../format";
import { useRun } from "../hooks/useRun";

const terminal = new Set(["succeeded", "failed", "cancelled", "needs_attention"]);

export function CommandCenterPage({ runId }: { runId: string }) {
  const { run, intelligence, loading, error, intelligenceError, refresh, setRun } = useRun(runId);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function cancel() {
    setCancelling(true);
    setCancelError(null);
    try {
      const cancelled = await apiClient.cancel(runId);
      setRun(cancelled);
    } catch (reason) {
      setCancelError(reason instanceof Error ? reason.message : "Unable to cancel review");
    } finally {
      setCancelling(false);
    }
  }

  if (error) {
    return (
      <div className="page">
        <PageHeader
          description="The durable review could not be loaded."
          title="Review command center"
        />
        <ErrorNotice message={error} retry={refresh} />
      </div>
    );
  }

  if (loading || !run) {
    return (
      <div className="page">
        <PageHeader
          description="Loading durable run state from SQLite."
          title="Review command center"
        />
        <Panel>
          <output className="loading-state">
            <span className="spinner" />
            Loading review…
          </output>
        </Panel>
      </div>
    );
  }

  const result = run.result as {
    snapshot?: { file_count?: number };
    synthesized_report?: string;
    provider_mode?: string;
  } | null;
  const indexedEvent = [...run.events]
    .reverse()
    .find((event) => event.event_type === "repository_index_ready");
  const fileCount =
    result?.snapshot?.file_count ??
    (typeof indexedEvent?.data.file_count === "number" ? indexedEvent.data.file_count : null);
  const latestCoverage = intelligence?.coverage.at(-1);
  const active = !terminal.has(run.status);
  const findings = intelligence?.findings ?? [];
  const rejected =
    intelligence?.candidates.filter(
      (candidate) => candidate.verdict && candidate.verdict.disposition !== "accepted",
    ) ?? [];

  return (
    <div className="page command-center">
      <PageHeader
        actions={
          <div className="header-actions">
            <StatusBadge status={run.status} />
            {active ? (
              <button
                className="button button-danger"
                disabled={cancelling}
                onClick={cancel}
                type="button"
              >
                {cancelling ? "Cancelling…" : "Cancel review"}
              </button>
            ) : null}
          </div>
        }
        description={`${run.request.project_path} · ${formatLabel(run.mode)} · run ${shortId(run.run_id)}`}
        eyebrow="Read-only review"
        title="Review command center"
      />

      <section className="stat-grid" aria-label="Review summary">
        <StatCard
          detail={fileCount === null ? "Available after repository indexing" : "Indexed snapshot"}
          icon="files"
          label="Files indexed"
          value={fileCount === null ? "Pending" : fileCount.toLocaleString()}
        />
        <StatCard
          detail={`${intelligence?.tasks.length ?? 0} durable tasks`}
          icon="users"
          label="Specialists"
          value={intelligence?.roles.length ?? 0}
        />
        <StatCard
          detail={`${rejected.length} rejected or unresolved candidates`}
          icon="alert"
          label="Accepted findings"
          tone="coral"
          value={findings.length}
        />
        <StatCard
          detail={
            intelligence?.usage.cost_usd
              ? `$${intelligence.usage.cost_usd.toFixed(4)} recorded cost`
              : result?.provider_mode === "index-only"
                ? "Index-only run"
                : "No provider cost recorded"
          }
          icon="coins"
          label="Model usage"
          tone="amber"
          value={formatTokens(intelligence?.usage.total_tokens ?? 0)}
        />
      </section>

      <Panel className="stage-panel">
        <StageStepper currentStage={run.current_stage} terminal={run.status === "succeeded"} />
      </Panel>
      {intelligenceError ? (
        <ErrorNotice
          message={`Run status is current; intelligence details are unavailable: ${intelligenceError}`}
        />
      ) : null}
      {cancelError ? <ErrorNotice message={cancelError} /> : null}

      <div className="command-grid">
        <div className="command-main">
          <Panel className="section-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Repository-driven plan</p>
                <h2>Specialists</h2>
              </div>
              <span>{intelligence?.waves.length ?? 0} wave(s)</span>
            </div>
            {intelligence?.roles.length ? (
              <div className="specialist-grid">
                {intelligence.roles.map((role) => (
                  <SpecialistCard
                    key={role.role_id}
                    role={role}
                    task={intelligence.tasks.find((task) => task.role_id === role.role_id)}
                  />
                ))}
              </div>
            ) : (
              <EmptyState
                description="Roles appear after the repository snapshot has been indexed."
                title="Planning specialists"
              />
            )}
          </Panel>

          <Panel className="section-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Verified output</p>
                <h2>Evidence-backed findings</h2>
              </div>
              <span>{findings.length} accepted</span>
            </div>
            {findings.length ? (
              <div className="finding-list">
                {findings.map((finding) => (
                  <article className="finding-row" key={finding.finding_id}>
                    <span className={`severity severity-${finding.severity}`}>
                      {formatLabel(finding.severity)}
                    </span>
                    <div>
                      <strong>{finding.title}</strong>
                      <p>{finding.claim}</p>
                      <small>
                        {Math.round(finding.confidence * 100)}% confidence ·{" "}
                        {finding.supporting_evidence_ids.length} verified evidence span(s)
                      </small>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <EmptyState
                description="Only findings that pass evidence verification and deterministic correlation appear here."
                title={active ? "Verification in progress" : "No accepted findings"}
              />
            )}
          </Panel>

          {result?.synthesized_report ? (
            <Panel className="report-panel">
              <p className="eyebrow">Synthesis</p>
              <h2>Review report</h2>
              <pre>{result.synthesized_report}</pre>
            </Panel>
          ) : null}
        </div>

        <aside className="command-side" aria-label="Review detail">
          <Panel className="coverage-panel">
            <h2>Coverage and gaps</h2>
            {latestCoverage ? (
              <>
                <div className="coverage-decision">
                  <StatusBadge status={latestCoverage.follow_up_decision} />
                  <span>{latestCoverage.decision_rationale}</span>
                </div>
                <dl className="coverage-metrics">
                  {Object.entries(latestCoverage.measured).map(([name, value]) => (
                    <div key={name}>
                      <dt>{formatLabel(name)}</dt>
                      <dd>{Math.round(value * 100)}%</dd>
                    </div>
                  ))}
                </dl>
                {latestCoverage.remaining_gaps.map((gap) => (
                  <p className="gap-item" key={gap}>
                    <Icon name="alert" />
                    {gap}
                  </p>
                ))}
                {latestCoverage.unsupported_areas.map((area) => (
                  <p className="gap-item" key={area}>
                    <Icon name="alert" />
                    Unsupported: {area}
                  </p>
                ))}
              </>
            ) : (
              <p className="muted-copy">
                Coverage is assessed after specialist results have been verified.
              </p>
            )}
          </Panel>
          <Panel className="timeline-panel">
            <h2>Durable timeline</h2>
            <p className="timeline-elapsed">
              <Icon name="clock" />
              {formatDuration(run.started_at, run.completed_at)}
            </p>
            <ol className="event-timeline">
              {[...run.events]
                .reverse()
                .slice(0, 12)
                .map((event) => (
                  <li key={event.sequence}>
                    <strong>{formatLabel(event.event_type)}</strong>
                    <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                  </li>
                ))}
            </ol>
          </Panel>
        </aside>
      </div>
    </div>
  );
}
