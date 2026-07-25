import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type {
  AnalysisMode,
  AnalysisStatus,
  HistoryRun,
  HistoryTrends,
  RunComparison,
} from "../api/contracts";
import { AppLink } from "../components/AppShell";
import {
  EmptyState,
  ErrorNotice,
  PageHeader,
  Panel,
  StatCard,
  StatusBadge,
} from "../components/primitives";
import { formatDuration, formatLabel, formatTokens, shortId } from "../format";

export function HistoryPage() {
  const initial = useMemo(() => new URLSearchParams(window.location.search), []);
  const [search, setSearch] = useState(initial.get("search") ?? "");
  const [status, setStatus] = useState(initial.get("status") ?? "");
  const [mode, setMode] = useState(initial.get("mode") ?? "");
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [trends, setTrends] = useState<HistoryTrends | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<RunComparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const params = historyParams(search, status, mode);
    window.history.replaceState({}, "", `/history?${params}`);
    Promise.all([
      apiClient.history(params, controller.signal),
      apiClient.historyTrends(30, controller.signal),
    ])
      .then(([page, trendData]) => {
        setRuns(page.items);
        setCursor(page.next_cursor);
        setTrends(trendData);
        setError(null);
      })
      .catch((reason) =>
        setError(reason instanceof Error ? reason.message : "Unable to load review history"),
      );
    return () => controller.abort();
  }, [search, status, mode]);

  async function compare() {
    if (selected.length !== 2) return;
    try {
      setComparison(await apiClient.compareRuns(selected[0], selected[1]));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to compare reviews");
    }
  }

  async function loadMore() {
    if (!cursor || loadingMore) return;
    setLoadingMore(true);
    try {
      const params = historyParams(search, status, mode);
      params.set("cursor", cursor);
      const page = await apiClient.history(params);
      setRuns((current) => [...current, ...page.items]);
      setCursor(page.next_cursor);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load more review history");
    } finally {
      setLoadingMore(false);
    }
  }

  const totalFindings = runs.reduce((sum, run) => sum + run.finding_count, 0);
  const totalCost = runs.reduce((sum, run) => sum + run.cost_usd, 0);
  const completed = runs.filter((run) => run.duration_seconds !== null);
  const averageDuration = completed.length
    ? completed.reduce((sum, run) => sum + (run.duration_seconds ?? 0), 0) / completed.length
    : null;

  return (
    <div className="page">
      <PageHeader
        description="Compare durable review outcomes and inspect measured trends without overstating partial history."
        eyebrow="Longitudinal intelligence"
        title="Review history"
      />
      {error ? <ErrorNotice message={error} /> : null}
      <div className="stat-grid">
        <StatCard
          detail="Current filtered result"
          icon="history"
          label="Reviews"
          value={runs.length}
        />
        <StatCard
          detail="Canonical findings"
          icon="findings"
          label="Findings"
          value={totalFindings}
        />
        <StatCard
          detail="Measured model usage"
          icon="coins"
          label="Cost"
          value={`$${totalCost.toFixed(4)}`}
        />
        <StatCard
          detail="Completed reviews only"
          icon="clock"
          label="Average time"
          value={averageDuration === null ? "—" : `${Math.round(averageDuration)}s`}
        />
      </div>
      <Panel className="history-filters">
        <label>
          <span>Project</span>
          <input
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Repository name or path"
            type="search"
            value={search}
          />
        </label>
        <label>
          <span>Status</span>
          <select onChange={(event) => setStatus(event.target.value)} value={status}>
            <option value="">All statuses</option>
            {(
              [
                "succeeded",
                "failed",
                "cancelled",
                "needs_attention",
                "running",
                "queued",
              ] as AnalysisStatus[]
            ).map((value) => (
              <option key={value} value={value}>
                {formatLabel(value)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Mode</span>
          <select onChange={(event) => setMode(event.target.value)} value={mode}>
            <option value="">All modes</option>
            {(["quick", "deep", "security", "change-set"] as AnalysisMode[]).map((value) => (
              <option key={value} value={value}>
                {formatLabel(value)}
              </option>
            ))}
          </select>
        </label>
        <button
          className="button button-secondary"
          disabled={selected.length !== 2}
          onClick={compare}
          type="button"
        >
          Compare selected ({selected.length}/2)
        </button>
      </Panel>
      <div className="history-layout">
        <Panel className="history-table-panel">
          {runs.length ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Compare</th>
                    <th scope="col">Review</th>
                    <th scope="col">Status</th>
                    <th scope="col">Findings</th>
                    <th scope="col">Usage</th>
                    <th scope="col">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.run_id}>
                      <td>
                        <input
                          aria-label={`Compare ${run.run_id}`}
                          checked={selected.includes(run.run_id)}
                          disabled={!selected.includes(run.run_id) && selected.length >= 2}
                          onChange={(event) =>
                            setSelected((current) =>
                              event.target.checked
                                ? [...current, run.run_id]
                                : current.filter((item) => item !== run.run_id),
                            )
                          }
                          type="checkbox"
                        />
                      </td>
                      <td>
                        <AppLink path={`/reviews/${run.run_id}`}>
                          <strong>{run.display_name ?? "Unbound review"}</strong>
                          <span>
                            {shortId(run.run_id)} · {formatLabel(run.mode)} ·{" "}
                            {new Date(run.created_at).toLocaleDateString()}
                          </span>
                        </AppLink>
                      </td>
                      <td>
                        <StatusBadge status={run.status} />
                      </td>
                      <td>{run.finding_count}</td>
                      <td>
                        {formatTokens(run.total_tokens)} · ${run.cost_usd.toFixed(4)}
                      </td>
                      <td>
                        {run.duration_seconds === null
                          ? "—"
                          : formatDuration(run.started_at, run.completed_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {cursor ? (
                <div className="pagination-note">
                  <button
                    className="button button-secondary"
                    disabled={loadingMore}
                    onClick={loadMore}
                    type="button"
                  >
                    {loadingMore ? "Loading more…" : "Load more reviews"}
                  </button>
                </div>
              ) : null}
            </div>
          ) : (
            <EmptyState
              description="No durable runs match the current server-derived filters."
              title="No reviews in this view"
            />
          )}
        </Panel>
        <Panel className="trend-panel">
          <h2>30-day measured trend</h2>
          {trends?.buckets.length ? (
            <div className="trend-bars">
              {trends.buckets.map((bucket) => (
                <div key={bucket.date}>
                  <span>{bucket.date.slice(5)}</span>
                  <i
                    aria-hidden="true"
                    style={{ height: `${Math.max(8, bucket.review_count * 18)}px` }}
                  />
                  <strong>{bucket.review_count}</strong>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-copy">No completed trend buckets are available in this period.</p>
          )}
          {trends?.partial ? (
            <p className="trend-note">
              History is empty or partial; no regression conclusion is implied.
            </p>
          ) : null}
        </Panel>
      </div>
      {comparison ? (
        <Panel className="comparison-panel">
          <div className="section-heading">
            <h2>Run comparison</h2>
            <span>
              {shortId(comparison.baseline_run_id)} → {shortId(comparison.target_run_id)}
            </span>
          </div>
          <div className="comparison-grid">
            {[
              ["New", comparison.new.length],
              ["Resolved", comparison.resolved.length],
              ["Unchanged", comparison.unchanged.length],
              ["Reopened", comparison.reopened.length],
              ["Severity moved", comparison.severity_moved.length],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

function historyParams(search: string, status: string, mode: string) {
  const params = new URLSearchParams({ limit: "50" });
  if (search) params.set("search", search);
  if (status) params.set("status", status);
  if (mode) params.set("mode", mode);
  return params;
}
