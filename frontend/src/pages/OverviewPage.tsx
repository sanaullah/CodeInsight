import { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { AnalysisRun } from "../api/contracts";
import { AppLink } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { EmptyState, ErrorNotice, PageHeader, Panel, StatusBadge } from "../components/primitives";
import { formatDuration, formatLabel } from "../format";
import type { OperationalState } from "../hooks/useOperationalState";

export function OverviewPage({ operational }: { operational: OperationalState }) {
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    void revision;
    const controller = new AbortController();
    apiClient
      .listRuns(8, controller.signal)
      .then((value) => {
        setRuns(value);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Unable to load reviews");
        }
      });
    return () => controller.abort();
  }, [revision]);

  const support = operational.capabilities?.languages.reduce<Record<string, number>>(
    (result, language) => {
      result[language.support_level] = (result[language.support_level] ?? 0) + 1;
      return result;
    },
    {},
  );

  return (
    <div className="page">
      <PageHeader
        actions={
          <AppLink className="button button-primary" path="/new-review">
            <Icon name="plus" />
            Start a review
          </AppLink>
        }
        description="Run bounded, repository-specific specialists and keep every decision tied to durable evidence."
        eyebrow="Architecture-aware analysis"
        title="Repository intelligence you can inspect."
      />

      <section className="overview-grid" aria-label="Runtime capabilities">
        <Panel className="overview-intro">
          <span className="read-only-pill">
            <Icon name="shield" />
            Read-only by design
          </span>
          <h2>One durable review path</h2>
          <p>
            CodeInsight indexes the repository once, plans roles from its actual shape, verifies
            evidence, correlates findings, measures gaps, and records the complete lifecycle in
            SQLite.
          </p>
          <ol className="workflow-strip" aria-label="Review workflow">
            {["Index", "Plan", "Analyze", "Verify", "Synthesize"].map((label, index) => (
              <li key={label}>
                <b>{index + 1}</b>
                {label}
              </li>
            ))}
          </ol>
        </Panel>
        <Panel className="capability-panel">
          <h2>Language coverage</h2>
          <dl className="capability-stats">
            <div>
              <dt>Parsed</dt>
              <dd>{support?.parsed ?? "—"}</dd>
            </div>
            <div>
              <dt>Dependency-aware</dt>
              <dd>{support?.["dependency-aware"] ?? "—"}</dd>
            </div>
            <div>
              <dt>Discoverable</dt>
              <dd>{operational.capabilities?.languages.length ?? "—"}</dd>
            </div>
          </dl>
          <p>
            Provider mode:{" "}
            <strong>
              {operational.capabilities?.model_provider_configured
                ? "Model-backed"
                : "Index-only until configured"}
            </strong>
          </p>
        </Panel>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Durable ledger</p>
            <h2>Recent reviews</h2>
          </div>
          <button
            className="button button-secondary"
            onClick={() => setRevision((value) => value + 1)}
            type="button"
          >
            Refresh
          </button>
        </div>
        {error ? (
          <ErrorNotice message={error} retry={() => setRevision((value) => value + 1)} />
        ) : null}
        {!error && runs.length === 0 ? (
          <Panel>
            <EmptyState
              action={
                <AppLink className="button button-primary" path="/new-review">
                  Create the first review
                </AppLink>
              }
              description="Completed and active runs will remain available here after restarts."
              title="No reviews yet"
            />
          </Panel>
        ) : (
          <div className="run-list">
            {runs.map((run) => (
              <AppLink className="run-list-item" key={run.run_id} path={`/reviews/${run.run_id}`}>
                <div>
                  <strong>{run.request.project_path}</strong>
                  <span>
                    {formatLabel(run.mode)} · {run.request.max_agents} specialists ·{" "}
                    {new Date(run.created_at).toLocaleString()}
                  </span>
                </div>
                <div>
                  <StatusBadge status={run.status} />
                  <span>{formatDuration(run.started_at, run.completed_at)}</span>
                </div>
              </AppLink>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
