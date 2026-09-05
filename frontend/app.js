"use strict";

/* Multi-Agent AI Data Analyst - frontend controller */

const $ = (id) => document.getElementById(id);

const AGENTS = [
  { key: "manager", icon: "🧠", name: "Manager", role: "Orchestration", flow: "plans & assigns" },
  { key: "cleaner", icon: "🧹", name: "Cleaner", role: "Data quality", flow: "finds issues" },
  { key: "analyst", icon: "📊", name: "Data Analyst", role: "Statistics", flow: "discovers insights" },
  { key: "visualizer", icon: "📈", name: "Visualization", role: "Charts", flow: "builds visuals" },
  { key: "reviewer", icon: "🛡️", name: "Reviewer", role: "Critical gate", flow: "verifies work" },
  { key: "reporter", icon: "📝", name: "Report", role: "Final deliverable", flow: "writes report" },
];

let state = null;
let es = null;
let pollTimer = null;
let selectedFile = null;
let agentTaskMap = {};

/* ---------------- health ---------------- */
async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const key = d.openrouter_configured ? "AI ready" : "no API key";
    $("health-pill").textContent = `${key} · ${d.model}`;
    if (d.openrouter_configured) $("health-pill").style.color = "var(--green)";
    else $("health-pill").style.color = "var(--yellow)";
  } catch {
    $("health-pill").textContent = "backend offline";
    $("health-pill").style.color = "var(--red)";
  }
}

/* ---------------- upload screen ---------------- */
const dropZone = $("drop-zone");
const fileInput = $("file-input");

function onFile(file) {
  if (!file) return;
  const okExt = file.name.toLowerCase().endsWith(".csv");
  const okSize = file.size <= 50 * 1024 * 1024;
  const info = $("file-info");
  info.classList.remove("hidden", "bad");
  if (!okExt) {
    info.classList.add("bad");
    info.textContent = "Unsupported file type. Please upload a .csv file.";
    selectedFile = null;
    $("start-btn").disabled = true;
    return;
  }
  if (!okSize) {
    info.classList.add("bad");
    info.textContent = "File exceeds the 50 MB limit.";
    selectedFile = null;
    $("start-btn").disabled = true;
    return;
  }
  selectedFile = file;
  info.textContent = `✓ ${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  $("start-btn").disabled = false;
}

fileInput.addEventListener("change", () => onFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); })
);
dropZone.addEventListener("drop", (e) => onFile(e.dataTransfer.files[0]));
dropZone.addEventListener("click", () => fileInput.click());

$("start-btn").addEventListener("click", startAnalysis);
$("new-analysis-btn").addEventListener("click", () => location.reload());

/* ---------------- start / live stream ---------------- */
async function startAnalysis() {
  if (!selectedFile) return;
  $("start-btn").disabled = true;
  $("screen-upload").classList.add("hidden");
  $("screen-live").classList.remove("hidden");
  $("live-title").textContent = "Spinning up the agent team…";
  $("live-question").textContent = selectedFile.name;

  renderAgents();
  addActivity("info", "⚡ Dataset loaded: " + selectedFile.name);

  const question = $("question").value.trim();
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("question", question || "");

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: fd });
    const body = await resp.json();
    if (!resp.ok) { showError(body.detail || "Failed to start analysis."); return; }
    state = body;
    const wf = state.workflow_id;
    $("live-question").textContent = `${selectedFile.name} · ${question || "no question — general exploration"}`;
    applyState(state);
    connectStream(wf);
  } catch (err) {
    showError("Could not reach the backend: " + err.message);
  }
}

function connectStream(wf) {
  if (es) es.close();
  es = new EventSource(`/api/workflow/${wf}/events`);
  es.onmessage = (msg) => {
    let parsed;
    try { parsed = JSON.parse(msg.data); } catch { return; }
    routeEvent(parsed, wf);
  };
  es.onerror = () => {
    // SSE hiccup (Windows/uvicorn quirk): fall back to polling.
    if (es) { es.close(); es = null; }
    startPolling(wf);
  };
}

function startPolling(wf) {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/workflow/${wf}`);
      const s = await r.json();
      routeState(s);
      if (["completed", "failed", "partial"].includes(s.status)) {
        clearInterval(pollTimer);
        pollTimer = null;
        onFinished(s);
      }
    } catch { /* keep polling */ }
  }, 1500);
}

function routeEvent(evt, wf) {
  const { event, data } = evt;
  if (event === "state") routeState(data);
  else if (event === "agent_status") setAgentStatus(data.agent, data.status);
  else if (event === "activity") addActivity(data.level || "info", data.message, data.agent);
  else if (event === "error") showError(data.message || "Workflow error");
  else if (event === "done") {
    (async () => {
      const r = await fetch(`/api/workflow/${wf}`);
      const s = await r.json();
      routeState(s);
      onFinished(s);
    })();
  }
}

function routeState(s) {
  state = s;
  applyState(s);
}

/* ---------------- rendering ---------------- */
function applyState(s) {
  if (s.current_agent) setAgentStatus(s.current_agent, "working");
  else { /* waiting */ }

  for (const name of s.completed_tasks || []) {
    // tasks are descriptions; agent statuses come from agent_results
  }
  if (s.agent_results) {
    for (const [key, res] of Object.entries(s.agent_results)) {
      if (res.status === "completed") setAgentStatus(key, "completed");
      else if (res.status === "failed") setAgentStatus(key, "failed");
    }
    // any agent present in results but not completed marker
  }

  if (s.current_task) {
    const focused = $("agent-detail");
    focused.innerHTML = `<h4>Focus: ${iconFor(s.current_agent)} ${s.current_agent}</h4>
      <p>${escapeHtml(s.current_task)}</p>`;
  }

  const overall = $("overall-status");
  overall.className = "status-chip " + s.status;
  overall.textContent = capitalize(s.status || "running");

  if (s.status === "completed") onFinished(s);
}

function renderAgents() {
  const grid = $("agent-grid");
  grid.innerHTML = "";
  for (const a of AGENTS) {
    agentTaskMap[a.key] = "";
    const card = document.createElement("div");
    card.className = "agent-card";
    card.id = `card-${a.key}`;
    card.innerHTML = `
      <div class="agent-head">
        <span class="agent-icon">${a.icon}</span>
        <div>
          <div class="agent-name">${a.name}</div>
          <div class="agent-role">${a.role}</div>
        </div>
        <span class="agent-status status-waiting" id="status-${a.key}">Waiting</span>
      </div>
      <div class="agent-task" id="task-${a.key}">—</div>
      <div class="agent-flow">↳ ${a.flow}</div>`;
    grid.appendChild(card);
  }
}

function setAgentStatus(key, status) {
  const el = $("status-" + key);
  if (!el) return;
  const card = $("card-" + key);
  const map = { waiting: "Waiting", working: "Working", completed: "Completed", failed: "Failed" };
  el.textContent = map[status] || status;
  el.className = "agent-status status-" + status;
  card.className = "agent-card " + status;
}

const iconFor = (k) => (AGENTS.find((a) => a.key === k) || { icon: "🤖" }).icon;

function addActivity(level, message, agent) {
  const ul = $("activity");
  const li = document.createElement("li");
  li.className = level || "info";
  const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const agentPfx = agent ? ` <span class="muted">[${agent}]</span>` : "";
  li.innerHTML = `<span class="t">${t}</span><span class="m">${escapeHtml(message)}${agentPfx}</span>`;
  ul.prepend(li);
  while (ul.children.length > 50) ul.lastElementChild.remove();
}

/* ---------------- finished ---------------- */
function onFinished(s) {
  if (!$("screen-results").classList.contains("hidden")) return;
  if (es) { es.close(); es = null; }
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  $("screen-live").classList.add("hidden");
  $("screen-results").classList.remove("hidden");
  $("results-question").textContent = s.question || s.filename;
  renderResults(s);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderResults(s) {
  const body = $("results-body");
  const overview = (s.memory && s.memory.dataset_overview) || {};
  const cleaner = (s.agent_results || {}).cleaner;
  const analyst = (s.agent_results || {}).analyst;
  const visualizer = (s.agent_results || {}).visualizer;
  const reviewer = (s.agent_results || {}).reviewer;

  const charts = (visualizer && (visualizer.output || {}).charts) || [];
  const q = overview.quality_score || {};

  const reviewApproved = (s.review_status || "") === "approved";
  const reviewClass = reviewApproved ? "approved" : "caveat";
  const reviewLabel = reviewApproved
    ? "🟢 Analysis Verified"
    : "🟡 Corrections / Caveats";

  body.innerHTML = `
    <div class="results-grid">

      <div class="card">
        <h3>🐞 Dataset Overview</h3>
        <div class="metric-row"><span class="metric-key">Rows</span><span><b>${overview.rows ?? "—"}</b></span></div>
        <div class="metric-row"><span class="metric-key">Columns</span><span><b>${overview.columns ?? "—"}</b></span></div>
        <div class="metric-row"><span class="metric-key">Filename</span><span>${escapeHtml(s.filename)}</span></div>
        <div class="metric-row"><span class="metric-key">Reviewer</span><span class="review-badge ${reviewClass}">${reviewLabel}</span></div>
      </div>

      <div class="card">
        <h3>🏥 Data Quality</h3>
        <div class="quality-score">
          <div class="score-ring">${q.score ?? "—"}/100</div>
          <div class="score-label">${q.label || ""}</div>
        </div>
        ${(cleaner && cleaner.output && cleaner.output.findings.length) ? `
          <ul>
            ${cleaner.output.findings.slice(0, 6).map((f) =>
              `<li>${escapeHtml(f.title)} — <span class="muted">${escapeHtml((f.detail || "").slice(0, 90))}</span></li>`
            ).join("")}
          </ul>` : `<p class="muted">No cleaner results.</p>`}
      </div>

      ${charts.length ? `<div class="card card-span">
        <h3>📈 Visual Analysis</h3>
        <div class="charts-grid">
          ${charts.map((c, i) => `
            <figure class="chart-figure">
              <img src="${escapeAttr(c.relative_path)}" alt="${escapeAttr(c.title)}" loading="lazy" />
              <figcaption>${escapeHtml(c.title)} <span class="muted">(${chartsCountExplainer(c)})</span></figcaption>
            </figure>`).join("")}
        </div>
      </div>` : ""}

      ${(analyst && analyst.output && analyst.output.findings.length) ? `<div class="card">
        <h3>💡 Key Insights</h3>
        ${analyst.output.findings.slice(0, 8).map((f) => `
          <div class="insight-card">
            <b>${escapeHtml(f.title)}</b>
            ${f.detail ? `<p style="margin-top:4px">${escapeHtml(f.detail)}</p>` : ""}
            ${f.evidence ? `<div class="ev">evidence: ${escapeHtml(f.evidence)}</div>` : ""}
          </div>`).join("")}
      </div>` : ""}

      <div class="card">
        <h3>🤖 Agent Findings</h3>
        <div class="agent-findings">
          ${Object.entries(s.agent_results || {}).map(([key, res]) => `
            <div class="af">
              <div class="af-head">
                <span class="af-name">${iconFor(key)} ${key}</span>
                <span class="af-status">${res.status}</span>
              </div>
              <div class="af-conf">confidence ${(res.confidence * 100).toFixed(0)}% ${res.error ? "· ⚠ " + escapeHtml(res.error.slice(0, 80)) : ""}</div>
              ${res.summary ? `<div class="muted" style="font-size:0.78rem;margin-top:4px">${escapeHtml(res.summary.slice(0, 140))}</div>` : ""}
            </div>`).join("")}
        </div>
      </div>

      <div class="card">
        <h3>🧠 Workflow Summary</h3>
        <div class="metric-row"><span class="metric-key">Workflow</span><span>${escapeHtml(s.workflow_id)}</span></div>
        <div class="metric-row"><span class="metric-key">Steps used</span><span>${s.total_steps}</span></div>
        <div class="metric-row"><span class="metric-key">Review cycles</span><span>${s.retry_count}</span></div>
        <div class="metric-row"><span class="metric-key">Completed tasks</span><span>${(s.completed_tasks || []).length}</span></div>
        ${s.error ? `<div class="ev" style="color:var(--yellow);margin-top:8px">⚠ ${escapeHtml(s.error)}</div>` : ""}
      </div>
    </div>

    <div class="report-preview" id="report-preview"></div>
  `;

  if (s.final_report) $("report-preview").innerHTML = renderMarkdown(s.final_report);
  else $("report-preview").innerHTML = "<p class='muted'>No report was generated.</p>";
}

function chartsCountExplainer(c) {
  if (c.chart_type) return c.chart_type;
  return "chart";
}

/* ---------------- markdown (minimal, XSS-safe) ---------------- */
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function renderMarkdown(md) {
  const escaped = escapeHtml(md);
  let html = escaped
    .replace(/^###### (.*)$/gm, "<h6>$1</h6>")
    .replace(/^##### (.*)$/gm, "<h5>$1</h5>")
    .replace(/^#### (.*)$/gm, "<h4>$1</h4>")
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
      const safe = src.replace(/["']/g, "");
      return `<img src="${safe}" alt="${alt}" loading="lazy" />`;
    });
  // lists
  html = html.replace(/(?:^|\n)(?:[-*] )(.+)(?:\n|$)/g, "<li>$1</li>");
  html = html.replace(/(<li>[\s\S]*?)(<\/li>)/g, "<ul>$1$2</ul>");
  html = html.replace(/<\/ul><ul>/g, "");
  // paragraphs
  html = html.replace(/(^|\n)(?!<)([^\n]+)\n/g, (_, pre, line) =>
    line.trim() && !/<h\d|<ul|<li|<img|<strong|<p|^[-*]/.test(line.trim())
      ? `${pre}<p>${line}</p>\n`
      : pre + line + "\n"
  );
  return html;
}

/* ---------------- downloads ---------------- */
function downloadReport() {
  if (!state) return;
  const blob = new Blob([state.final_report || ""], { type: "text/markdown" });
  triggerDownload(blob, `ai_data_analysis_report_${state.workflow_id}.md`);
}
function downloadJson() {
  if (!state) return;
  const a = document.createElement("a");
  a.href = `/api/download/json/${state.workflow_id}`;
  a.download = "";
  a.click();
}
function triggerDownload(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

$("download-report-btn").addEventListener("click", downloadReport);
$("download-json-btn").addEventListener("click", downloadJson);

function showError(msg) {
  $("error-text").textContent = msg;
  $("error-modal").classList.remove("hidden");
  const status = $("overall-status");
  if (status) { status.className = "status-chip failed"; status.textContent = "Failed"; }
}
$("error-close").addEventListener("click", () => $("error-modal").classList.add("hidden"));

const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

checkHealth();