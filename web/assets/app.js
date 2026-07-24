const state = {
  runId: null,
  status: "idle",
  pollHandle: null,
  startedAt: null,
  report: "",
};

const byId = (id) => document.getElementById(id);
const terminalStatuses = new Set(["succeeded", "failed", "cancelled"]);

function setApiState(mode, label) {
  const element = byId("api-state");
  element.className = `api-state ${mode}`;
  element.lastChild.textContent = ` ${label}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

function formatEventName(name) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function describeEvent(event) {
  const data = event.data || {};
  return data.message || data.agent || data.role || data.status || "";
}

function renderTimeline(events) {
  const timeline = byId("timeline");
  timeline.replaceChildren();
  for (const event of events.slice(-12).reverse()) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = formatEventName(event.event_type);
    const detail = document.createElement("small");
    const description = describeEvent(event);
    const time = new Date(event.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    detail.textContent = description ? `${time} · ${description}` : time;
    item.append(title, detail);
    timeline.append(item);
  }
}

function updateStatus(status) {
  const badge = byId("status-badge");
  badge.className = `status-badge ${status}`;
  badge.textContent = formatEventName(status);
  byId("cancel-run").classList.toggle("hidden", terminalStatuses.has(status));
}

function renderRun(run) {
  state.runId = run.run_id;
  state.status = run.status;
  state.startedAt = run.started_at ? new Date(run.started_at) : new Date(run.created_at);
  byId("empty-state").classList.add("hidden");
  byId("run-content").classList.remove("hidden");
  byId("run-id").textContent = run.run_id;
  byId("run-title").textContent =
    run.status === "succeeded"
      ? "Review complete"
      : run.status === "failed"
        ? "Review needs attention"
        : "Review in progress";
  updateStatus(run.status);
  renderTimeline(run.events || []);

  const resultCard = byId("result-card");
  const report = run.result?.synthesized_report || (run.error ? `Analysis failed:\n${run.error}` : "");
  if (report) {
    state.report = report;
    byId("report").textContent = report;
    resultCard.classList.remove("hidden");
  } else {
    resultCard.classList.add("hidden");
  }
  updateElapsed();
}

function updateElapsed() {
  if (!state.startedAt) return;
  const seconds = Math.max(0, Math.round((Date.now() - state.startedAt.getTime()) / 1000));
  byId("elapsed").textContent =
    seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

async function pollRun() {
  if (!state.runId) return;
  try {
    const run = await api(`/api/v1/analyses/${state.runId}`);
    renderRun(run);
    if (terminalStatuses.has(run.status)) {
      clearInterval(state.pollHandle);
      state.pollHandle = null;
      byId("start-review").disabled = false;
      await loadRecentRuns();
    }
  } catch (error) {
    byId("form-message").textContent = error.message;
  }
}

async function startReview(event) {
  event.preventDefault();
  byId("form-message").textContent = "";
  const button = byId("start-review");
  button.disabled = true;

  const strategy = byId("chunking-strategy").value;
  const payload = {
    project_path: byId("project-path").value,
    goal: byId("goal").value || null,
    model_name: byId("model-name").value || null,
    max_agents: Number(byId("max-agents").value),
    max_tokens_per_chunk: Number(byId("max-tokens").value),
    enable_chunking: strategy !== "NONE",
    chunking_strategy: strategy,
    auto_detect_languages: true,
    enable_dynamic_file_selection: byId("dynamic-files").checked,
    enable_tool_calling: byId("tool-calling").checked,
  };

  try {
    const accepted = await api("/api/v1/analyses", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.runId = accepted.run_id;
    state.startedAt = new Date();
    byId("empty-state").classList.add("hidden");
    byId("run-content").classList.remove("hidden");
    byId("run-id").textContent = accepted.run_id;
    byId("run-title").textContent = "Preparing review";
    updateStatus(accepted.status);
    renderTimeline([]);
    await pollRun();
    if (!terminalStatuses.has(state.status) && !state.pollHandle) {
      state.pollHandle = setInterval(pollRun, 1400);
    }
  } catch (error) {
    button.disabled = false;
    byId("form-message").textContent = error.message;
  }
}

async function cancelRun() {
  if (!state.runId) return;
  try {
    const run = await api(`/api/v1/analyses/${state.runId}`, { method: "DELETE" });
    renderRun(run);
    if (state.pollHandle) clearInterval(state.pollHandle);
    state.pollHandle = null;
    byId("start-review").disabled = false;
    await loadRecentRuns();
  } catch (error) {
    byId("form-message").textContent = error.message;
  }
}

async function loadCapabilities() {
  const capabilities = await api("/api/v1/capabilities");
  const totals = capabilities.languages.reduce(
    (result, language) => {
      result[language.support_level] = (result[language.support_level] || 0) + 1;
      return result;
    },
    {},
  );
  const values = byId("support-legend").querySelectorAll("strong");
  values[0].textContent = totals.parsed || 0;
  values[1].textContent = totals["dependency-aware"] || 0;
  values[2].textContent = capabilities.languages.length;
}

async function loadRecentRuns() {
  const runs = await api("/api/v1/analyses?limit=6");
  const container = byId("recent-runs");
  container.replaceChildren();
  if (!runs.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No review runs in this API process yet.";
    container.append(empty);
    return;
  }
  for (const run of runs) {
    const card = document.createElement("article");
    card.className = "recent-card";
    const button = document.createElement("button");
    button.type = "button";
    const title = document.createElement("strong");
    title.textContent = run.request.project_path;
    const meta = document.createElement("small");
    const date = document.createElement("span");
    date.textContent = new Date(run.created_at).toLocaleString();
    const status = document.createElement("span");
    status.textContent = formatEventName(run.status);
    meta.append(date, status);
    button.append(title, meta);
    button.addEventListener("click", () => renderRun(run));
    card.append(button);
    container.append(card);
  }
}

async function initialize() {
  try {
    const health = await api("/api/v1/health");
    setApiState("online", `${health.version} · ready`);
    await Promise.all([loadCapabilities(), loadRecentRuns()]);
  } catch (error) {
    setApiState("offline", "API unavailable");
    byId("form-message").textContent = error.message;
  }
}

byId("review-form").addEventListener("submit", startReview);
byId("cancel-run").addEventListener("click", cancelRun);
byId("refresh-runs").addEventListener("click", loadRecentRuns);
byId("copy-report").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.report);
  byId("copy-report").textContent = "Copied";
  setTimeout(() => (byId("copy-report").textContent = "Copy"), 1200);
});
setInterval(updateElapsed, 1000);
initialize();
