import type { FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../api/client";
import type {
  LocalSettings,
  PresetRequest,
  ProviderTestResponse,
  ReviewPreset,
  SettingsResponse,
} from "../api/contracts";
import { ErrorNotice, PageHeader, Panel } from "../components/primitives";
import type { OperationalState } from "../hooks/useOperationalState";

export function SettingsPage({ operational }: { operational: OperationalState }) {
  const [record, setRecord] = useState<SettingsResponse | null>(null);
  const [draft, setDraft] = useState<LocalSettings | null>(null);
  const [presets, setPresets] = useState<ReviewPreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [editingPresetId, setEditingPresetId] = useState<string | null>(null);
  const [editingPresetName, setEditingPresetName] = useState("");
  const [providerTest, setProviderTest] = useState<ProviderTestResponse | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([apiClient.settings(controller.signal), apiClient.presets(controller.signal)])
      .then(([settings, savedPresets]) => {
        setRecord(settings);
        setDraft(settings.settings);
        setPresets(savedPresets);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Unable to load settings");
        }
      });
    return () => controller.abort();
  }, []);

  const dirty = useMemo(
    () => Boolean(record && draft && JSON.stringify(record.settings) !== JSON.stringify(draft)),
    [draft, record],
  );

  async function save() {
    if (!record || !draft || saving) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await apiClient.updateSettings(draft, record.version);
      setRecord(updated);
      setDraft(updated.settings);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to save settings");
    } finally {
      setSaving(false);
    }
  }

  async function createPreset(event: FormEvent) {
    event.preventDefault();
    if (!draft || !presetName.trim()) return;
    setError(null);
    try {
      const created = await apiClient.createPreset(presetName.trim(), toPresetRequest(draft));
      setPresets((current) => [...current, created].sort((a, b) => a.name.localeCompare(b.name)));
      setPresetName("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to create preset");
    }
  }

  async function removePreset(preset: ReviewPreset) {
    setError(null);
    try {
      await apiClient.deletePreset(preset.preset_id, preset.version);
      setPresets((current) => current.filter((item) => item.preset_id !== preset.preset_id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to delete preset");
    }
  }

  async function renamePreset(preset: ReviewPreset) {
    const name = editingPresetName.trim();
    if (!name) return;
    setError(null);
    try {
      const updated = await apiClient.updatePreset({ ...preset, name });
      setPresets((current) =>
        current
          .map((item) => (item.preset_id === updated.preset_id ? updated : item))
          .sort((a, b) => a.name.localeCompare(b.name)),
      );
      setEditingPresetId(null);
      setEditingPresetName("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to rename preset");
    }
  }

  async function testProvider() {
    setError(null);
    try {
      setProviderTest(await apiClient.testProvider());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to test provider");
    }
  }

  return (
    <div className="page">
      <PageHeader
        description="Manage non-secret local defaults, bounded budgets, and reusable review presets."
        title="Settings"
      />
      {error ? <ErrorNotice message={error} /> : null}
      {!draft || !record ? (
        <Panel>
          <p>Loading durable settings…</p>
        </Panel>
      ) : (
        <div className="settings-grid">
          <Panel className="settings-panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">SQLite-backed defaults</span>
                <h2>Analysis guardrails</h2>
              </div>
              <span className={dirty ? "status-badge warning" : "status-badge success"}>
                {dirty ? "Unsaved changes" : `Version ${record.version}`}
              </span>
            </div>
            <div className="field-grid">
              <label className="form-section">
                <span>Default mode</span>
                <select
                  aria-label="Default mode"
                  value={draft.default_mode}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      default_mode: event.target.value as LocalSettings["default_mode"],
                    })
                  }
                >
                  <option value="quick">Quick</option>
                  <option value="deep">Deep</option>
                  <option value="security">Security</option>
                  <option value="change-set">Change set</option>
                </select>
              </label>
              <NumberSetting
                label="Maximum specialists"
                value={draft.default_max_agents}
                min={1}
                max={12}
                onChange={(value) => setDraft({ ...draft, default_max_agents: value })}
              />
              <NumberSetting
                label="Maximum waves"
                value={draft.default_max_waves}
                min={1}
                max={10}
                onChange={(value) => setDraft({ ...draft, default_max_waves: value })}
              />
              <NumberSetting
                label="Maximum tasks"
                value={draft.default_max_tasks}
                min={1}
                max={500}
                onChange={(value) => setDraft({ ...draft, default_max_tasks: value })}
              />
              <NumberSetting
                label="Maximum model tokens"
                value={draft.default_max_total_tokens}
                min={1}
                onChange={(value) => setDraft({ ...draft, default_max_total_tokens: value })}
              />
              <NumberSetting
                label="Maximum cost (USD)"
                value={draft.default_max_cost_usd}
                min={0}
                step={0.01}
                onChange={(value) => setDraft({ ...draft, default_max_cost_usd: value })}
              />
              <NumberSetting
                label="Maximum elapsed seconds"
                value={draft.default_max_elapsed_seconds}
                min={1}
                onChange={(value) => setDraft({ ...draft, default_max_elapsed_seconds: value })}
              />
              <NumberSetting
                label="Retention days"
                value={draft.retention_days}
                min={1}
                max={3650}
                onChange={(value) => setDraft({ ...draft, retention_days: value })}
              />
            </div>
            <label className="toggle-row">
              <input
                checked={draft.evidence_excerpt_enabled}
                onChange={(event) =>
                  setDraft({ ...draft, evidence_excerpt_enabled: event.target.checked })
                }
                type="checkbox"
              />
              <span>
                <strong>Show evidence excerpts</strong>
                <small>
                  Controls local UI display only; source artifacts remain in the durable ledger.
                </small>
              </span>
            </label>
            <div className="button-row">
              <button
                className="button button-primary"
                disabled={!dirty || saving}
                onClick={save}
                type="button"
              >
                {saving ? "Saving…" : "Save defaults"}
              </button>
              <button
                className="button"
                disabled={!dirty}
                onClick={() => setDraft(record.settings)}
                type="button"
              >
                Discard
              </button>
            </div>
          </Panel>
          <div className="settings-stack">
            <Panel>
              <span className="eyebrow">Environment-owned</span>
              <h2>Services</h2>
              <dl className="settings-status">
                <div>
                  <dt>Model provider</dt>
                  <dd>
                    {operational.capabilities?.model_provider_configured
                      ? "Configured"
                      : "Index-only"}
                  </dd>
                </div>
                <div>
                  <dt>Langfuse</dt>
                  <dd>
                    {operational.capabilities?.langfuse_enabled
                      ? "Optional export active"
                      : "Disabled"}
                  </dd>
                </div>
                <div>
                  <dt>Runtime</dt>
                  <dd>FastAPI · uv · SQLite/WAL</dd>
                </div>
              </dl>
              <p className="muted-copy">
                Provider endpoints and credentials are read from process environment variables and
                are never stored in presets or settings.
              </p>
              <button className="button" onClick={testProvider} type="button">
                Test provider
              </button>
              {providerTest ? (
                <p
                  aria-live="polite"
                  className={
                    providerTest.reachable ? "inline-result success" : "inline-result warning"
                  }
                >
                  {providerTest.status}
                  {providerTest.latency_ms == null ? "" : ` ${providerTest.latency_ms} ms`}
                </p>
              ) : null}
            </Panel>
            <Panel>
              <span className="eyebrow">Reusable, path-free configuration</span>
              <h2>Review presets</h2>
              <form className="preset-form" onSubmit={createPreset}>
                <label className="form-section">
                  <span>Preset name</span>
                  <input
                    maxLength={100}
                    onChange={(event) => setPresetName(event.target.value)}
                    required
                    value={presetName}
                  />
                </label>
                <button className="button button-primary" type="submit">
                  Save current defaults
                </button>
              </form>
              {presets.length ? (
                <ul className="preset-list">
                  {presets.map((preset) => (
                    <li key={preset.preset_id}>
                      <div className="preset-identity">
                        {editingPresetId === preset.preset_id ? (
                          <input
                            aria-label={`Rename ${preset.name}`}
                            maxLength={100}
                            onChange={(event) => setEditingPresetName(event.target.value)}
                            value={editingPresetName}
                          />
                        ) : (
                          <strong>{preset.name}</strong>
                        )}
                        <small>
                          {preset.request.mode} · {preset.request.max_agents} specialists · v
                          {preset.version}
                        </small>
                      </div>
                      <div className="button-row">
                        {editingPresetId === preset.preset_id ? (
                          <>
                            <button
                              className="button"
                              onClick={() => renamePreset(preset)}
                              type="button"
                            >
                              Save name
                            </button>
                            <button
                              className="button"
                              onClick={() => setEditingPresetId(null)}
                              type="button"
                            >
                              Cancel rename
                            </button>
                          </>
                        ) : (
                          <button
                            className="button"
                            onClick={() => {
                              setEditingPresetId(preset.preset_id);
                              setEditingPresetName(preset.name);
                            }}
                            type="button"
                          >
                            Rename
                          </button>
                        )}
                        <button
                          className="button"
                          onClick={() => setDraft(fromPreset(preset.request, draft))}
                          type="button"
                        >
                          Apply
                        </button>
                        <button
                          className="button button-danger"
                          onClick={() => removePreset(preset)}
                          type="button"
                        >
                          Delete
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted-copy">No presets saved yet.</p>
              )}
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}

function NumberSetting({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="form-section">
      <span>{label}</span>
      <input
        aria-label={label}
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        required
        step={step}
        type="number"
        value={value}
      />
    </label>
  );
}

function toPresetRequest(settings: LocalSettings): PresetRequest {
  return {
    goal: null,
    model_name: null,
    file_extensions: null,
    selected_directories: null,
    mode: settings.default_mode,
    max_agents: settings.default_max_agents,
    max_waves: settings.default_max_waves,
    max_tasks: settings.default_max_tasks,
    max_total_tokens: settings.default_max_total_tokens,
    max_cost_usd: settings.default_max_cost_usd,
    max_elapsed_seconds: settings.default_max_elapsed_seconds,
  };
}

function fromPreset(preset: PresetRequest, current: LocalSettings): LocalSettings {
  return {
    ...current,
    default_mode: preset.mode,
    default_max_agents: preset.max_agents,
    default_max_waves: preset.max_waves,
    default_max_tasks: preset.max_tasks,
    default_max_total_tokens: preset.max_total_tokens,
    default_max_cost_usd: preset.max_cost_usd,
    default_max_elapsed_seconds: preset.max_elapsed_seconds,
  };
}
