import { useEffect, useRef, useState } from "react";
import type { FindingDetail } from "../../api/contracts";

export interface FindingEvidenceDrawerProps {
  finding: FindingDetail | null;
  loading?: boolean;
  error?: string | null;
  exportHref?: string;
  openEvidenceHref?: string;
  onClose: () => void;
  onCopyEvidence?: (excerpt: string) => Promise<void> | void;
}

export function FindingEvidenceDrawer({
  finding,
  loading = false,
  error = null,
  exportHref,
  openEvidenceHref,
  onClose,
  onCopyEvidence,
}: FindingEvidenceDrawerProps) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const [copyStatus, setCopyStatus] = useState("");
  const open = Boolean(finding || loading || error);

  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;
  const impacts = finding
    ? [
        ...new Set(
          finding.candidates
            .map((record) => record.candidate.impact.trim())
            .filter((impact) => impact.length > 0),
        ),
      ]
    : [];

  async function copyExcerpt(excerpt: string) {
    try {
      if (onCopyEvidence) await onCopyEvidence(excerpt);
      else await navigator.clipboard.writeText(excerpt);
      setCopyStatus("Evidence excerpt copied.");
    } catch {
      setCopyStatus("Clipboard access is unavailable.");
    }
  }

  return (
    <>
      <button
        aria-label="Close finding evidence"
        className="architecture-drawer-scrim"
        onClick={onClose}
        type="button"
      />
      <aside
        aria-labelledby="architecture-finding-heading"
        aria-modal="true"
        className="architecture-finding-drawer"
        role="dialog"
      >
        <header>
          <div>
            <span className="architecture-eyebrow">Verified finding</span>
            <h2 id="architecture-finding-heading">{finding?.title ?? "Finding evidence"}</h2>
          </div>
          <button
            aria-label="Close finding evidence"
            className="architecture-icon-button"
            onClick={onClose}
            ref={closeButton}
            type="button"
          >
            Close
          </button>
        </header>
        {loading ? <output>Loading verified finding evidence...</output> : null}
        {error ? (
          <p className="architecture-inline-error" role="alert">
            {error}
          </p>
        ) : null}
        {finding ? (
          <>
            <div className="architecture-finding-summary">
              <span className={`architecture-severity severity-${finding.severity}`}>
                {finding.severity}
              </span>
              <span>{Math.round(finding.confidence * 100)}% confidence</span>
              <span>{finding.review_state}</span>
              {exportHref ? (
                <a className="architecture-action" download href={exportHref}>
                  Export finding
                </a>
              ) : null}
              {openEvidenceHref ? (
                <a className="architecture-action" href={openEvidenceHref}>
                  Open evidence workspace
                </a>
              ) : null}
            </div>
            <section>
              <h3>Verified claim</h3>
              <p>{finding.claim}</p>
            </section>
            <section>
              <h3>Recommendation</h3>
              <p>{finding.recommendation}</p>
            </section>
            {impacts.length ? (
              <section>
                <h3>Evidence-derived impact</h3>
                {impacts.map((impact) => (
                  <p key={impact}>{impact}</p>
                ))}
              </section>
            ) : null}
            <section>
              <h3>Evidence ({finding.evidence.length})</h3>
              {finding.evidence.length ? (
                finding.evidence.map((evidence) => (
                  <article className="architecture-evidence-card" key={evidence.evidence_id}>
                    <div>
                      <strong>
                        {evidence.relative_path}:{evidence.start_line}-{evidence.end_line}
                      </strong>
                      <span>{evidence.integrity}</span>
                    </div>
                    <p>
                      {evidence.evidence_kind.replaceAll("_", " ")}
                      {evidence.redacted ? " / redacted" : ""}
                      {evidence.truncated ? " / truncated" : ""}
                    </p>
                    {evidence.excerpt ? (
                      <>
                        <pre>{evidence.excerpt}</pre>
                        <button
                          className="architecture-action"
                          onClick={() => void copyExcerpt(evidence.excerpt ?? "")}
                          type="button"
                        >
                          Copy evidence
                        </button>
                      </>
                    ) : (
                      <p>No excerpt is available for this evidence record.</p>
                    )}
                  </article>
                ))
              ) : (
                <p className="architecture-muted">No evidence records are available.</p>
              )}
            </section>
            <section>
              <h3>Specialist verification ({finding.candidates.length})</h3>
              {finding.candidates.map((record) => (
                <article className="architecture-verdict-card" key={record.candidate.candidate_id}>
                  <strong>{record.role?.name ?? "Specialist"}</strong>
                  <span>{record.verdict?.disposition ?? "unverified"}</span>
                  <p>{record.verdict?.rationale ?? "No verifier rationale is available."}</p>
                </article>
              ))}
            </section>
          </>
        ) : null}
        <p aria-live="polite">{copyStatus}</p>
      </aside>
    </>
  );
}
