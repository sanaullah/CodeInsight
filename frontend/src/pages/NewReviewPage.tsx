import type { FormEvent } from "react";
import { useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type { AnalysisMode, AnalysisRequest } from "../api/contracts";
import { Icon } from "../components/Icon";
import { ErrorNotice, PageHeader, Panel } from "../components/primitives";
import { formatLabel } from "../format";
import type { OperationalState } from "../hooks/useOperationalState";
import { navigate } from "../router";

const defaults: AnalysisRequest = {
  project_path: "",
  goal: null,
  model_name: null,
  max_agents: 4,
  file_extensions: null,
  selected_directories: null,
  mode: "deep",
  max_waves: 2,
  max_tasks: 100,
  max_total_tokens: 1_000_000,
  max_cost_usd: 25,
  max_elapsed_seconds: 3_600,
};

const modes: Array<{ value: AnalysisMode; label: string; detail: string }> = [
  { value: "quick", label: "Quick", detail: "Focused first pass" },
  { value: "deep", label: "Deep", detail: "Broad evidence coverage" },
  { value: "security", label: "Security", detail: "Trust boundaries first" },
  { value: "change-set", label: "Change set", detail: "Changed neighborhood" },
];

export function NewReviewPage({ operational }: { operational: OperationalState }) {
  const [form, setForm] = useState(defaults);
  const [extensions, setExtensions] = useState("");
  const [directories, setDirectories] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const availableModes = useMemo(
    () => new Set(operational.capabilities?.analysis_modes ?? modes.map((mode) => mode.value)),
    [operational.capabilities],
  );

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const payload: AnalysisRequest = {
        ...form,
        goal: form.goal?.trim() || null,
        model_name: form.model_name?.trim() || null,
        file_extensions: splitValues(extensions),
        selected_directories: splitValues(directories),
      };
      const accepted = await apiClient.submit(payload);
      navigate(`/reviews/${accepted.run_id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start review");
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <PageHeader
        actions={
          <span className="read-only-pill">
            <Icon name="shield" />
            Read-only by default
          </span>
        }
        description="Define the repository, goal, mode, and durable limits. CodeInsight plans the specialist team from the repository itself."
        title="Start a new review"
      />
      <div className="review-layout">
        <Panel className="review-form-panel">
          <form onSubmit={submit}>
            <div className="form-section">
              <label htmlFor="project-path">Repository path</label>
              <input
                autoComplete="off"
                id="project-path"
                onChange={(event) => setForm({ ...form, project_path: event.target.value })}
                placeholder="H:\projects\payments-platform"
                required
                value={form.project_path}
              />
              <small>A directory accessible to the local FastAPI process.</small>
            </div>
            <div className="form-section">
              <div className="label-row">
                <label htmlFor="review-goal">Review goal</label>
                <span>{form.goal?.length ?? 0}/4000</span>
              </div>
              <textarea
                id="review-goal"
                maxLength={4000}
                onChange={(event) => setForm({ ...form, goal: event.target.value })}
                placeholder="Focus on authorization boundaries, reliability risks, and missing tests."
                rows={5}
                value={form.goal ?? ""}
              />
            </div>
            <fieldset className="form-section">
              <legend>Analysis mode</legend>
              <div className="mode-grid">
                {modes
                  .filter((mode) => availableModes.has(mode.value))
                  .map((mode) => (
                    <label
                      className={form.mode === mode.value ? "mode-card selected" : "mode-card"}
                      key={mode.value}
                    >
                      <input
                        checked={form.mode === mode.value}
                        name="mode"
                        onChange={() => setForm({ ...form, mode: mode.value })}
                        type="radio"
                        value={mode.value}
                      />
                      <strong>{mode.label}</strong>
                      <span>{mode.detail}</span>
                    </label>
                  ))}
              </div>
            </fieldset>
            <div className="field-grid">
              <div className="form-section">
                <label htmlFor="model-name">Model</label>
                <input
                  disabled={!operational.capabilities?.model_provider_configured}
                  id="model-name"
                  onChange={(event) => setForm({ ...form, model_name: event.target.value })}
                  placeholder={
                    operational.capabilities?.model_provider_configured
                      ? "Use configured default"
                      : "Index-only without a provider"
                  }
                  value={form.model_name ?? ""}
                />
                {!operational.capabilities?.model_provider_configured ? (
                  <small>Configure an OpenAI-compatible endpoint to enable model selection.</small>
                ) : null}
              </div>
              <div className="form-section">
                <label htmlFor="specialists">Maximum specialists</label>
                <select
                  id="specialists"
                  onChange={(event) => setForm({ ...form, max_agents: Number(event.target.value) })}
                  value={form.max_agents}
                >
                  {[2, 4, 6, 8, 10, 12]
                    .filter((value) => value <= (operational.capabilities?.max_agents ?? 12))
                    .map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                </select>
              </div>
            </div>
            <details className="advanced-fields">
              <summary>Scope and durable budgets</summary>
              <div className="field-grid">
                <NumberField
                  id="max-waves"
                  label="Maximum waves"
                  max={10}
                  min={1}
                  onChange={(value) => setForm({ ...form, max_waves: value })}
                  value={form.max_waves}
                />
                <NumberField
                  id="max-tasks"
                  label="Maximum tasks"
                  max={500}
                  min={1}
                  onChange={(value) => setForm({ ...form, max_tasks: value })}
                  value={form.max_tasks}
                />
                <NumberField
                  id="max-tokens"
                  label="Total model tokens"
                  min={1}
                  onChange={(value) => setForm({ ...form, max_total_tokens: value })}
                  value={form.max_total_tokens}
                />
                <NumberField
                  id="max-cost"
                  label="Maximum cost (USD)"
                  min={0}
                  onChange={(value) => setForm({ ...form, max_cost_usd: value })}
                  step={0.01}
                  value={form.max_cost_usd}
                />
                <NumberField
                  id="max-elapsed"
                  label="Maximum elapsed seconds"
                  min={1}
                  onChange={(value) => setForm({ ...form, max_elapsed_seconds: value })}
                  value={form.max_elapsed_seconds}
                />
                <div className="form-section">
                  <label htmlFor="extensions">File extensions</label>
                  <input
                    id="extensions"
                    onChange={(event) => setExtensions(event.target.value)}
                    placeholder=".py, .ts, .tsx"
                    value={extensions}
                  />
                </div>
              </div>
              <div className="form-section">
                <label htmlFor="directories">Selected directories</label>
                <input
                  id="directories"
                  onChange={(event) => setDirectories(event.target.value)}
                  placeholder="src, tests, packages/api"
                  value={directories}
                />
                <small>
                  Relative paths only. Leave blank to inspect the full supported repository.
                </small>
              </div>
            </details>
            {error ? <ErrorNotice message={error} /> : null}
            <button
              className="button button-primary submit-review"
              disabled={submitting}
              type="submit"
            >
              <span>{submitting ? "Starting review…" : "Start code review"}</span>
              <Icon name="arrow" />
            </button>
          </form>
        </Panel>
        <Panel className="review-plan">
          <h2>Review plan</h2>
          <p>The exact roles and task count are planned after the repository snapshot exists.</p>
          <ol>
            <PlanStep icon="files" title="Index repository">
              Read supported files, symbols, imports, calls, ownership, and Git state.
            </PlanStep>
            <PlanStep icon="architecture" title="Plan repository-specific roles">
              Select bounded specialists from the actual language and risk profile.
            </PlanStep>
            <PlanStep icon="users" title="Run bounded specialists">
              Respect task, wave, token, cost, and elapsed-time limits.
            </PlanStep>
            <PlanStep icon="database" title="Verify and synthesize">
              Accept only evidence-backed findings and record remaining gaps.
            </PlanStep>
          </ol>
          <div className="invariant-card">
            <Icon name="shield" />
            <div>
              <strong>Repository writes are unavailable</strong>
              <span>Analysis tools can read the selected repository; they cannot modify it.</span>
            </div>
          </div>
          <dl className="review-summary">
            <div>
              <dt>Mode</dt>
              <dd>{formatLabel(form.mode)}</dd>
            </div>
            <div>
              <dt>Specialist limit</dt>
              <dd>{form.max_agents}</dd>
            </div>
            <div>
              <dt>Wave limit</dt>
              <dd>{form.max_waves}</dd>
            </div>
            <div>
              <dt>Provider</dt>
              <dd>
                {operational.capabilities?.model_provider_configured ? "Configured" : "Index-only"}
              </dd>
            </div>
          </dl>
        </Panel>
      </div>
    </div>
  );
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="form-section">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        required
        step={step}
        type="number"
        value={value}
      />
    </div>
  );
}

function PlanStep({
  icon,
  title,
  children,
}: {
  icon: Parameters<typeof Icon>[0]["name"];
  title: string;
  children: string;
}) {
  return (
    <li>
      <span>
        <Icon name={icon} />
      </span>
      <div>
        <strong>{title}</strong>
        <p>{children}</p>
      </div>
    </li>
  );
}

function splitValues(value: string): string[] | null {
  const values = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return values.length ? values : null;
}
