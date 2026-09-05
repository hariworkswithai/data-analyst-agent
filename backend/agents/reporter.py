"""REPORT AGENT - writes the final business report after approval.

The report is composed in Python from the structured, validated agent
results so every number is real. When the LLM is available it is used
to polish the Executive Summary and Recommendations; otherwise those
sections are generated deterministically from the findings.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from backend.agents.base import AgentContext, ToolPlanAgent
from backend.config import REPORT_DIR
from backend.models import AgentResult
from backend.utils.errors import LLMError

logger = logging.getLogger("analyst.agents")


class ReportAgent(ToolPlanAgent):
    name = "reporter"
    label = "Report Agent"

    async def run(self, ctx: AgentContext) -> AgentResult:
        self._emit(ctx, "agent_start", {"agent": self.name, "task": ctx.task})
        try:
            markdown = await self._compose_report(ctx)
            path = self._save_report(ctx, markdown)
            output = {"markdown": markdown, "report_path": str(path)}
            result = AgentResult(
                agent=self.name,
                status="completed",
                output=output,
                confidence=1.0,
                summary=f"Report written ({len(markdown)} chars).",
            )
            self._emit(ctx, "agent_done", {"agent": self.name, "status": "done"})
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Report generation failed: %s", exc)
            self._emit(ctx, "agent_done", {"agent": self.name, "status": "failed"})
            return AgentResult(
                agent=self.name,
                status="failed",
                output={},
                confidence=0.0,
                error=str(exc),
                summary="Report generation failed.",
            )

    async def _compose_report(self, ctx: AgentContext) -> str:
        """Build markdown from structured results (real data only)."""
        profile = ctx.profile or {}
        overview = profile.get("overview", {})
        cleaner = (ctx.prior_results.get("cleaner") or AgentResult(agent="cleaner", status="skipped")).output
        analyst = (ctx.prior_results.get("analyst") or AgentResult(agent="analyst", status="skipped")).output
        visualizer = (ctx.prior_results.get("visualizer") or AgentResult(agent="visualizer", status="skipped")).output

        exec_summary = ""
        if self.llm:
            exec_summary = await self._exec_summary(ctx, analyst)
        if not exec_summary:
            exec_summary = _fallback_executive_summary(analyst)

        sections = []
        sections.append("# AI DATA ANALYSIS REPORT\n")
        sections.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
        sections.append(f"**Dataset:** `{ctx.df.shape[0]}` rows × `{ctx.df.shape[1]}` columns  ")
        sections.append(f"**User question:** {ctx.question}\n")

        sections.append("## Executive Summary\n")
        sections.append(exec_summary + "\n")

        sections.append("## Dataset Overview\n")
        sections.append(_overview_section(profile))

        sections.append("## Data Quality\n")
        sections.append(_quality_section(profile, cleaner))

        sections.append("## Key Insights\n")
        sections.append(_insights_section(analyst))

        sections.append("## Visual Analysis\n")
        sections.append(_visual_section(ctx, visualizer))

        sections.append("## Important Patterns\n")
        sections.append(_patterns_section(analyst, profile))

        sections.append("## Recommendations\n")
        recs = ""
        if self.llm:
            recs = await self._recommendations(ctx, analyst)
        if not recs:
            recs = _fallback_recommendations(cleaner, analyst)
        sections.append(recs)

        sections.append("## Data Limitations\n")
        sections.append(_limitations_section(profile, cleaner, ctx))

        sections.append("## Final Conclusion\n")
        sections.append(_conclusion_section(exec_summary))

        return "\n".join(sections)

    async def _exec_summary(self, ctx: AgentContext, analyst: dict) -> str:
        findings = analyst.get("findings", [])[:6]
        try:
            return await self.llm.chat(
                system=(
                    "You write concise executive summaries for data analysis reports. "
                    "Only use the findings given. Never invent numbers."
                ),
                user=(
                    "Findings:\n" + json.dumps(findings, default=str) +
                    "\n\nWrite a 3-5 sentence executive summary, plain text."
                ),
                temperature=0.3,
                max_tokens=300,
            )
        except (LLMError, Exception):  # noqa: BLE001
            return ""

    async def _recommendations(self, ctx: AgentContext, analyst: dict) -> str:
        findings = analyst.get("findings", [])[:8]
        try:
            text = await self.llm.chat(
                system=(
                    "You write practical business recommendations based ONLY on "
                    "the provided data-driven findings. Never add unverified claims."
                ),
                user=(
                    "Findings:\n" + json.dumps(findings, default=str) +
                    "\n\nWrite 3-5 actionable recommendations as a markdown bullet list."
                ),
                temperature=0.3,
                max_tokens=400,
            )
            return text.strip() + "\n"
        except (LLMError, Exception):  # noqa: BLE001
            return ""

    def _save_report(self, ctx: AgentContext, markdown: str):
        import uuid

        wf_id = f"{ctx.workflow_id or 'wf'}_{uuid.uuid4().hex[:8]}"
        path = REPORT_DIR / f"report_{wf_id}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def deterministic_result(self, ctx: AgentContext) -> dict:
        return {"markdown": "# Report\n(no content)", "report_path": ""}


# ---------------------------------------------------------------------------
# Section builders (everything from computed values, never guessed)
# ---------------------------------------------------------------------------


def _overview_section(profile: dict) -> str:
    o = profile.get("overview", {})
    lines = [
        f"- **Rows:** {o.get('rows', '?')}",
        f"- **Columns:** {o.get('columns', '?')}",
        "- **Columns & types:**",
    ]
    for col, info in (profile.get("columns") or {}).items():
        lines.append(f"  - `{col}` — {info.get('dtype', 'unknown')} "
                     f"({info.get('kind', '')}, {info.get('non_null', '?')} non-null)")
    return "\n".join(lines) + "\n"


def _quality_section(profile: dict, cleaner: dict) -> str:
    q = profile.get("quality_score", {})
    lines = [
        f"- **Quality score:** {q.get('score', '?')} / 100 ({q.get('label', 'n/a')})",
    ]
    missing = profile.get("missing_values", {})
    dups = profile.get("duplicates", {})
    outliers = profile.get("outliers", {})
    lines.append(f"- **Missing values:** {missing.get('total_missing', 0)} cells")
    lines.append(f"- **Duplicate rows:** {dups.get('duplicate_row_count', 0)}")
    outlier_cols = [c["column"] for c in outliers.get("columns", []) if c.get("outlier_count", 0) > 0]
    lines.append(f"- **Columns with outliers:** {', '.join(outlier_cols) if outlier_cols else 'none detected'}")

    findings = cleaner.get("findings") or []
    if findings:
        lines.append("\n**Cleaner findings:**")
        for f in findings[:8]:
            lines.append(f"- [{f.get('severity', 'info')}] {f.get('title', '')}: {f.get('detail', '')}")
    return "\n".join(lines) + "\n"


def _insights_section(analyst: dict) -> str:
    findings = analyst.get("findings") or []
    if not findings:
        return "_No structured insights were produced._\n"
    lines = []
    for f in findings[:10]:
        title = f.get("title")
        detail = f.get("detail")
        evidence = f.get("evidence")
        lines.append(f"**{title}**")
        if detail:
            lines.append(detail)
        if evidence:
            lines.append(f"*Evidence:* {evidence}")
        lines.append("")
    return "\n".join(lines)


def _visual_section(ctx: AgentContext, visualizer: dict) -> str:
    charts = visualizer.get("charts") or []
    if not charts:
        return "_No charts were generated for this dataset._\n"
    lines = ["The following charts were produced by the Visualization Agent."]
    for i, ch in enumerate(charts, 1):
        lines.append(f"\n### Chart {i}: {ch.get('title')}")
        rp = ch.get("relative_path", "")
        lines.append(f"\n![{ch.get('title')}]({rp})")
        lines.append("")
        if ch.get("x_column"):
            lines.append(f"- X: {ch.get('x_column')}")
        if ch.get("y_column"):
            lines.append(f"- Y: {ch.get('y_column')}")
        if ch.get("chart_type"):
            lines.append(f"- Type: {ch.get('chart_type')}")
    lines.append("")
    return "\n".join(lines)


def _patterns_section(analyst: dict, profile: dict) -> str:
    stats = analyst.get("statistics") or {}
    lines = []
    corr = stats.get("correlations") or []
    if corr:
        lines.append("**Notable correlations:**")
        for pair, val in corr[:5]:
            strength = "strong" if abs(val) >= 0.7 else ("moderate" if abs(val) >= 0.4 else "weak")
            lines.append(f"- {pair[0]} and {pair[1]}: {val} ({strength})")
    trends = stats.get("trends") or []
    if trends:
        lines.append("**Trends:**")
        for t in trends[:3]:
            pts = t.get("points", [])
            if len(pts) >= 2:
                lines.append(f"- {t['column']}: from {pts[0]['value']} to {pts[-1]['value']} across {len(pts)} periods")
    if not lines:
        lines.append("_See key insights for the main patterns found._")
    return "\n".join(lines) + "\n"


def _fallback_recommendations(cleaner: dict, analyst: dict) -> str:
    recs = []
    fixes = cleaner.get("recommended_fixes") or []
    safe_fixes = [f for f in fixes if f.get("safe_to_automate")][:3]
    for f in safe_fixes:
        recs.append(f"- Address data quality: {f.get('issue')} — {f.get('fix')}")
    findings = analyst.get("findings") or []
    top = findings[0] if findings else None
    if top:
        recs.append(f"- Focus on the strongest signal: {top.get('title')} ({top.get('evidence', '')})")
    if not recs:
        recs.append("- Explore the dataset further with targeted business questions.")
    return "\n".join(recs) + "\n"


def _limitations_section(profile: dict, cleaner: dict, ctx: AgentContext) -> str:
    lines = [
        "- Analysis is based only on the uploaded dataset; no external data was used.",
        "- Correlations describe relationships, not causation.",
        "- Outlier detection uses the IQR method and may flag legitimate extreme values.",
    ]
    q = profile.get("quality_score", {})
    if q.get("score", 100) < 70:
        lines.append(f"- Data quality score is {q.get('score')}/100 ({q.get('label')}); "
                     "results should be interpreted with caution.")
    return "\n".join(lines) + "\n"


def _conclusion_section(exec_summary: str) -> str:
    first = exec_summary.strip().split(".")[0] if exec_summary else ""
    return (f"{first}. The analysis covers the main trends, comparisons, and risks in the "
            "dataset and provides a basis for data-driven decisions.\n")


def _fallback_executive_summary(analyst: dict) -> str:
    findings = analyst.get("findings") or []
    if not findings:
        return "The analysis produced no structured insights."
    return "Key findings: " + " ".join(f.get("title") for f in findings[:4]) + "."