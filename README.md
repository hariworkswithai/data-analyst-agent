# Multi-Agent AI Data Analyst

An autonomous data-analysis application built with a **multi-agent LLM architecture**.
Upload a CSV (optionally with a business question) and let a team of specialized AI
agents — Manager, Cleaner, Analyst, Visualizer, Reviewer, Reporter — deliver a
full written analysis report with charts.

Every number in the report is computed by **Python**; the LLMs only *plan* which
tools to call and *interpret* the results. If an LLM call fails (no credits, timeouts),
each agent degrades gracefully to a deterministic fallback so the pipeline never dies.

---

## Architecture

```
┌──────────┐   CSV + question    ┌────────────────────────────────────────────┐
│  User UI │ ──────────────────▶ │            FastAPI orchestrator             │
└──────────┘                     │                                            │
   ▲  SSE / REST                 │   ┌─────────┐                              │
   │                             │   │ Manager │  builds the agent plan       │
   │                             │   └────┬────┘                              │
                                 │        │ delegates                          │
                                 │   ┌────▼─────────────┬───────────────┐      │
                                 │   │ Cleaner          │   Visualizer  │      │
                                 │   │ Analyst          └───────┬───────┘      │
                                 │   └────▲─────────────┬───────┘              │
                                 │        │             │ charts (PNG)         │
                                 │   ┌────┴─────┐       │                      │
                                 │   │ Reviewer │◀──────┘ (verifies, can send  │
                                 │   └────┬─────┘    corrections back)          │
                                 │        │ approved                            │
                                 │   ┌────▼─────┐                               │
                                 │   │ Reporter │  writes final markdown report │
                                 │   └──────────┘                               │
                                 └────────────────────────────────────────────┘
```

### The agent loop

Each working agent runs a bounded **plan → execute → interpret** cycle:

1. **Plan** — the LLM picks which controlled tools to call and why (structured JSON).
2. **Execute** — the backend runs the tools against the real DataFrame and returns safe, serializable results.
3. **Interpret** — the LLM reads the tool results and produces findings with `evidence`, `severity`, and `confidence`.

The **Reviewer** adds a non-negotiable quality gate: every finding must cite numeric
evidence, else the cycle is sent back to the Manager as a correction directive
(up to **3 review cycles**), with **hard limits** enforced throughout:

| Limit | Value |
|---|---|
| Max tool steps per agent | 20 |
| Max review cycles | 3 |
| Max tool retries | 2 |
| Max upload size | 50 MB |
| Max rows / columns | 200,000 / 200 |
| Allowed file type | CSV |

### The agents

| Agent | Responsibility |
|---|---|
| **Manager** | Reads the question, inspects the dataset profile, creates and assigns the plan, routes reviewer corrections. |
| **Cleaner** | Runs quality scans: missing values, duplicates, constants, outliers, high-cardinality columns, and category-formatting inconsistencies. |
| **Analyst** | Computes group summaries, correlations, trends, and anomaly detection; turns them into evidence-backed insights. |
| **Visualizer** | Picks the right chart for each insight (bar/line/scatter/box/heatmap) and renders publication-quality PNGs. |
| **Reviewer** | Mechanically verifies that every claimed insight has numeric evidence, then asks the LLM for a final judgment. |
| **Reporter** | Assembles the executive summary, findings, charts, recommendations, and limitations into a polished Markdown report. |

> **Live demo note:** uses `minimax/minimax-m3:free` by default — a free-tier model
> that requires no credits. Full LLM mode verified: cleaner catches real issues
> (missing values, duplicates, `North`/`north` category split, outliers), reviewer
> exercises a real correction loop, reporter writes a 9000+ char insight-driven report.

---

## Getting Started

### 1. Requirements

- Python 3.12+
- An [OpenRouter](https://openrouter.ai) API key (optional but recommended)

### 2. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure the API key

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=minimax/minimax-m3:free
```

Any OpenRouter key works out of the box with free `:free` models — no credits required.
For better output quality, funded keys can switch to paid models by setting
`OPENROUTER_MODEL=openai/gpt-4o-mini` in `.env`.

### 4. Run

```powershell
python -m uvicorn backend.main:app --port 8010
```

Open `http://127.0.0.1:8010` in your browser, drag in a CSV, ask a question, and
watch the agents work in real time over Server-Sent Events.

> Port 8000 is often occupied — the default here is **8010**.

### 5. Try the sample dataset

`sample_data/sales.csv` (126 rows) includes realistic quality issues — a missing
value, duplicate rows, an inconsistent `north`/`North` category, and outliers —
that the Cleaner and Reviewer are designed to catch.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload CSV + optional `question`, returns `workflow_id` |
| `GET` | `/api/workflow/{id}` | Current workflow state (polling) |
| `GET` | `/api/workflow/{id}/events` | Live Server-Sent Events stream |
| `POST` | `/api/workflow/{id}/replay` | Re-push full state (SSE reconnects) |
| `GET` | `/api/download/report/{id}` | Download the Markdown report |
| `GET` | `/api/download/json/{id}` | Download all agent results as JSON |
| `GET` | `/api/health` | Health, model, and limits info |

Outputs are written under `generated/`:

```
generated/
  charts/   ...*.png
  reports/  ...*.md
```

Full agent results are also exportable via `/api/download/json/{id}`.

---

## Safety & reliability by design

- **No arbitrary code execution.** Agents can only call a fixed, internally
  registered tool set; unknown tools, non-primitive arguments, and attempts to
  pass the DataFrame or touch file paths are rejected.
- **Backend-only API key.** The key never leaves the server; the frontend knows nothing about it.
- **Bounded everywhere.** Step, retry, and review caps keep a misbehaving model from looping forever.
- **Graceful degradation.** LLM outages, timeouts, quota errors, and malformed model
  JSON produce deterministic fallback outputs — the workflow still completes with charts and a report.
- **Evidence-checked review.** The Reviewer refuses to approve unsupported claims;
  ambiguous results trigger a correction round instead of a wrong report.

---

## Testing

```powershell
# 34 unit & integration tests (no LLM required)
python -m pytest -q

# deterministic end-to-end smoke test
python scripts/smoke_test.py

# end-to-end with real LLM calls (requires funded key)
python scripts/smoke_test.py llm

# LLM connectivity probe
python scripts/llm_check.py

# live SSE stream integration test (needs the server running)
python scripts/sse_test.py 8010
```

---

## Project layout

```
backend/
  config.py            # settings, limits, paths
  main.py              # FastAPI app entrypoint
  api/routes.py        # REST + SSE endpoints
  services/
    llm.py             # OpenRouter client, JSON parsing, retries
    orchestrator.py    # bounded multi-agent workflow
    jobs.py            # in-memory job registry + event hub
  agents/              # base + manager, cleaner, analyst, visualizer, reviewer, reporter
  tools/               # registry + controlled dataset/analysis/chart tools
  utils/               # validation, errors
frontend/
  index.html, style.css, app.js   # upload -> live control center -> results
sample_data/
  sales.csv            # demo dataset with embedded quality issues
tests/                 # 34 passing tests
scripts/               # smoke tests, LLM probe, SSE test
generated/             # outputs (charts, reports)
```

---

## Future improvements

- Persistent workflow history (SQLite) so results survive server restarts
- CSV cleaning preview + user-approved transformation step
- More chart types (pivot heatmaps, Paretos) and Excel export
- Streaming the report to the UI as it is written
- Dockerfile + `docker compose` for one-command setup