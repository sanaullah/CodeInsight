import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type {
  CanonicalFinding,
  FindingDetail,
  FindingReviewState,
  FindingSummary,
} from "../api/contracts";
import { EmptyState, ErrorNotice, PageHeader, Panel, StatusBadge } from "../components/primitives";
import { formatLabel, shortId } from "../format";

const severities: Array<CanonicalFinding["severity"] | ""> = [
  "",
  "critical",
  "high",
  "medium",
  "low",
  "info",
];
const reviewStates: Array<FindingReviewState | ""> = [
  "",
  "new",
  "validated",
  "acknowledged",
  "reviewed",
  "dismissed",
  "reopened",
  "resolved",
];

export function FindingsPage() {
  const initial = useMemo(() => new URLSearchParams(window.location.search), []);
  const [search, setSearch] = useState(initial.get("search") ?? "");
  const [severity, setSeverity] = useState(initial.get("severity") ?? "");
  const [reviewState, setReviewState] = useState(initial.get("review_state") ?? "");
  const [items, setItems] = useState<FindingSummary[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [cursor, setCursor] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkConfirming, setBulkConfirming] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);
  const [detail, setDetail] = useState<FindingDetail | null>(null);
  const [detailId, setDetailId] = useState(initial.get("finding") ?? "");
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: revision is an explicit retry/reload nonce.
  useEffect(() => {
    const controller = new AbortController();
    const params = findingParams(search, severity, reviewState);
    setLoading(true);
    apiClient
      .findings(params, controller.signal)
      .then((page) => {
        setItems(page.items);
        setCounts(page.counts_by_severity as Record<string, number>);
        setCursor(page.next_cursor);
        setSelected(new Set());
        setError(null);
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to load findings");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [search, severity, reviewState, revision]);

  useEffect(() => {
    const params = findingParams(search, severity, reviewState);
    if (detailId) params.set("finding", detailId);
    window.history.replaceState({}, "", `/findings${params.size ? `?${params}` : ""}`);
  }, [detailId, reviewState, search, severity]);

  useEffect(() => {
    if (!detailId) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    apiClient
      .finding(detailId, controller.signal)
      .then((value) => {
        setDetail(value);
        setError(null);
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to load finding");
      });
    return () => controller.abort();
  }, [detailId]);

  function openFinding(findingId: string) {
    setDetailId(findingId);
  }

  function closeFinding() {
    setDetailId("");
    setDetail(null);
  }

  async function loadMore() {
    if (!cursor || loadingMore) return;
    setLoadingMore(true);
    try {
      const params = findingParams(search, severity, reviewState);
      params.set("cursor", cursor);
      const page = await apiClient.findings(params);
      setItems((current) => [...current, ...page.items]);
      setCursor(page.next_cursor);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load more findings");
    } finally {
      setLoadingMore(false);
    }
  }

  async function updateFinding(
    finding: FindingSummary | FindingDetail,
    nextState: FindingReviewState,
  ) {
    try {
      const updated = await apiClient.updateFinding(finding.finding_id, {
        review_state: nextState,
        expected_version: finding.review_version,
      });
      setDetail(updated);
      setRevision((value) => value + 1);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to update finding");
    }
  }

  async function bulkReview() {
    const targets = items.filter((item) => selected.has(item.finding_id));
    try {
      await Promise.all(targets.map((item) => updateFinding(item, "reviewed")));
      setSelected(new Set());
    } catch {
      // Individual failures are surfaced by updateFinding.
    }
  }

  return (
    <div className="page">
      <PageHeader
        actions={
          <div className="button-row">
            <a
              className="button button-secondary"
              download
              href="/api/v1/findings-export?format=json"
            >
              Export JSON
            </a>
            <a
              className="button button-secondary"
              download
              href="/api/v1/findings-export?format=csv"
            >
              Export CSV
            </a>
          </div>
        }
        description="Search verified findings, inspect immutable evidence, and record local review decisions."
        eyebrow="Evidence lifecycle"
        title="Findings"
      />
      {error ? (
        <ErrorNotice message={error} retry={() => setRevision((value) => value + 1)} />
      ) : null}
      <fieldset className="severity-summary">
        <legend className="visually-hidden">Finding severity summary</legend>
        {(["critical", "high", "medium", "low", "info"] as const).map((value) => (
          <button key={value} onClick={() => setSeverity(value)} type="button">
            <span className={`severity severity-${value}`}>{formatLabel(value)}</span>
            <strong>{counts[value] ?? 0}</strong>
          </button>
        ))}
      </fieldset>
      <Panel className="findings-workspace">
        <form className="filter-bar" onSubmit={(event) => event.preventDefault()}>
          <label>
            <span>Search</span>
            <input
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Title or verified claim"
              type="search"
              value={search}
            />
          </label>
          <label>
            <span>Severity</span>
            <select onChange={(event) => setSeverity(event.target.value)} value={severity}>
              {severities.map((value) => (
                <option key={value} value={value}>
                  {value ? formatLabel(value) : "All severities"}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Review state</span>
            <select onChange={(event) => setReviewState(event.target.value)} value={reviewState}>
              {reviewStates.map((value) => (
                <option key={value} value={value}>
                  {value ? formatLabel(value) : "All states"}
                </option>
              ))}
            </select>
          </label>
          <button
            className="button button-secondary"
            disabled={!selected.size}
            onClick={() => setBulkConfirming(true)}
            type="button"
          >
            Mark selected reviewed ({selected.size})
          </button>
        </form>
        {bulkConfirming ? (
          <div className="confirmation-bar" role="alert">
            <span>Mark {selected.size} selected finding(s) as reviewed?</span>
            <div className="button-row">
              <button
                className="button button-primary"
                onClick={() => {
                  setBulkConfirming(false);
                  void bulkReview();
                }}
                type="button"
              >
                Confirm bulk review
              </button>
              <button className="button" onClick={() => setBulkConfirming(false)} type="button">
                Cancel
              </button>
            </div>
          </div>
        ) : null}
        {loading ? (
          <output className="loading-state">Loading verified findings…</output>
        ) : items.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Select</th>
                  <th scope="col">Severity</th>
                  <th scope="col">Finding</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Review</th>
                </tr>
              </thead>
              <tbody>
                {items.map((finding) => (
                  <tr key={finding.finding_id}>
                    <td>
                      <input
                        aria-label={`Select ${finding.title}`}
                        checked={selected.has(finding.finding_id)}
                        onChange={(event) =>
                          setSelected((current) => {
                            const next = new Set(current);
                            event.target.checked
                              ? next.add(finding.finding_id)
                              : next.delete(finding.finding_id);
                            return next;
                          })
                        }
                        type="checkbox"
                      />
                    </td>
                    <td>
                      <span className={`severity severity-${finding.severity}`}>
                        {formatLabel(finding.severity)}
                      </span>
                    </td>
                    <td>
                      <button
                        className="finding-link"
                        onClick={() => openFinding(finding.finding_id)}
                        type="button"
                      >
                        <strong>{finding.title}</strong>
                        <span>{finding.claim}</span>
                      </button>
                    </td>
                    <td>{Math.round(finding.confidence * 100)}%</td>
                    <td>
                      <StatusBadge status={finding.review_state} />
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
                  {loadingMore ? "Loading more…" : "Load more findings"}
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <EmptyState
            description="No canonical findings match the active server-derived filters."
            title="No findings in this view"
          />
        )}
      </Panel>
      {detail ? (
        <>
          <button
            aria-label="Close finding detail"
            className="drawer-scrim"
            onClick={closeFinding}
            type="button"
          />
          <aside
            aria-label="Finding detail"
            aria-modal="true"
            className="finding-drawer"
            role="dialog"
          >
            <div className="drawer-heading">
              <div>
                <span className={`severity severity-${detail.severity}`}>
                  {formatLabel(detail.severity)}
                </span>
                <h2>{detail.title}</h2>
                <p>
                  Finding {shortId(detail.finding_id)} · {Math.round(detail.confidence * 100)}%
                  confidence
                </p>
              </div>
              <button
                aria-label="Close finding detail"
                className="icon-button"
                onClick={closeFinding}
                type="button"
              >
                ×
              </button>
            </div>
            <section>
              <h3>Verified claim</h3>
              <p>{detail.claim}</p>
              <h3>Recommendation</h3>
              <p>{detail.recommendation}</p>
            </section>
            <section>
              <h3>Review decision</h3>
              <div className="review-actions">
                {(
                  [
                    "validated",
                    "acknowledged",
                    "reviewed",
                    "dismissed",
                    "reopened",
                    "resolved",
                  ] as const
                ).map((state) => (
                  <button
                    className="button button-secondary"
                    disabled={detail.review_state === state}
                    key={state}
                    onClick={() => updateFinding(detail, state)}
                    type="button"
                  >
                    {formatLabel(state)}
                  </button>
                ))}
              </div>
            </section>
            <section>
              <h3>Evidence ({detail.evidence.length})</h3>
              {detail.evidence.map((evidence) => (
                <article className="evidence-card" key={evidence.evidence_id}>
                  <strong>
                    {evidence.relative_path}:{evidence.start_line}-{evidence.end_line}
                  </strong>
                  <span>
                    Integrity: {evidence.integrity} · snapshot hash {shortId(evidence.content_hash)}
                  </span>
                  {evidence.excerpt ? (
                    <>
                      <pre>{evidence.excerpt}</pre>
                      <button
                        className="text-button"
                        onClick={async () => {
                          try {
                            await navigator.clipboard.writeText(evidence.excerpt ?? "");
                            setCopyStatus(`Copied evidence from ${evidence.relative_path}.`);
                          } catch {
                            setCopyStatus("Clipboard access is unavailable.");
                          }
                        }}
                        type="button"
                      >
                        Copy verified excerpt
                      </button>
                    </>
                  ) : (
                    <p>Source excerpt is unavailable or failed integrity validation.</p>
                  )}
                </article>
              ))}
            </section>
            {copyStatus ? <p aria-live="polite">{copyStatus}</p> : null}
            <section>
              <h3>Specialist consensus</h3>
              {detail.candidates.map((record) => (
                <article className="consensus-card" key={record.candidate.candidate_id}>
                  <strong>{record.role?.name ?? "Specialist"}</strong>
                  <StatusBadge status={record.verdict?.disposition ?? "unverified"} />
                  <p>{record.verdict?.rationale ?? "No verifier verdict is available."}</p>
                </article>
              ))}
            </section>
          </aside>
        </>
      ) : null}
    </div>
  );
}

function findingParams(search: string, severity: string, reviewState: string) {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  if (severity) params.set("severity", severity);
  if (reviewState) params.set("review_state", reviewState);
  params.set("limit", "50");
  return params;
}
