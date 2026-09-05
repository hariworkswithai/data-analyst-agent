"""DATA ANALYST AGENT - statistical and business analysis.

Chooses analyses relevant to the user's question, requests the matching
tools, and interprets the computed results. Never invents statistics.
"""

from __future__ import annotations

import json

from backend.agents.base import AgentContext, ToolPlanAgent, findings_validator
from backend.tools.registry import try_execute_tool

ANALYSIS_TOOLS = """- group_by_analysis(group_column, value_columns: list, agg: sum|mean|count|median|max|min)
  Aggregate numeric columns by a categorical group.
- calculate_correlation(method: pearson|spearman)
  Correlation matrix of numeric columns.
- calculate_statistics(columns: list)
  Descriptive statistics for numeric columns.
- analyze_trend(date_column, value_columns: list, period: D|W|M|Q)
  Time-series aggregation.
- detect_anomalies(value_columns: list, z_threshold: number)
  Z-score anomaly detection.
- get_sample(n)
  Small preview rows for inspection."""


class AnalystAgent(ToolPlanAgent):
    name = "analyst"
    label = "Data Analyst Agent"
    max_operations = 10
    interpret_max_tokens = 1500
    plan_max_tokens = 700

    def build_plan_prompt(self, ctx: AgentContext) -> str:
        cleaner_summary = ""
        prior = ctx.prior_results.get("cleaner")
        if prior:
            cleaner_summary = json.dumps(prior.output, default=str)[:1200]

        return f"""You are the DATA ANALYST agent. Plan a real analysis for the user's question using Python tools.

User question: {ctx.question}
Your assigned task: {ctx.task}

Dataset profile:
{ctx.profile_text}

Cleaner findings (context):
{cleaner_summary}

Available analysis tools:
{ANALYSIS_TOOLS}

Design an analysis plan. Prefer analyses that directly answer the user's
question (e.g. sales by region, top products, trends over time). Do NOT
run every possible analysis - pick the most relevant. Never invent numbers.

Return strict JSON:
{{"reasoning": "why these analyses",
 "operations": [
   {{"tool": "group_by_analysis", "arguments": {{"group_column":"Region","value_columns":["Sales"],"agg":"sum"}}, "reason": "why"}}
 ]}}"""

    def build_interpret_prompt(self, ctx: AgentContext, tool_outcomes: list[dict]) -> str:
        compact = json.dumps(tool_outcomes, default=str, ensure_ascii=False)
        return f"""You are the DATA ANALYST agent. Interpret these REAL computed results.
Derive business insights ONLY from the returned numbers. Never invent statistics.

Tool results (ground truth, computed by Python):
{compact}

User question: {ctx.question}

Return strict JSON:
{{"findings": [
   {{"title": "clear short insight", "detail": "explanation", "evidence": "exact numbers from results", "severity": "info|warning"}}
 ],
 "statistics": {{"key measures mentioned"}},
 "summary": "one-line summary of the analysis",
 "confidence": 0.0-1.0}}"""

    def validate_interpret(self, data):
        return findings_validator(data)

    def default_plan(self, ctx: AgentContext) -> dict:
        return {"operations": _heuristic_analysis_plan(ctx)}

    def deterministic_result(self, ctx: AgentContext) -> dict:
        findings = []
        stats = {}
        operations = _heuristic_analysis_plan(ctx)
        for op in operations:
            outcome = try_execute_tool(ctx.df, op["tool"], op["arguments"])
            if not outcome["ok"]:
                continue
            res = outcome["result"]
            if op["tool"] == "group_by_analysis":
                groups = res.get("groups", [])
                value_cols = res.get("value_columns", [])
                main_col = value_cols[0] if value_cols else None

                if groups:
                    if main_col:
                        total = sum((g.get(main_col) or 0) for g in groups) or 1
                        top = groups[0]
                        top_share = round((top.get(main_col) or 0) / total * 100, 1)
                        findings.append({
                            "title": f"Leading segment: {top.get('group')} in '{res.get('group_column')}'",
                            "detail": (
                                f"{top.get('group')} accounts for {top_share}% of total "
                                f"'{main_col}' across {len(groups)} groups."
                            ),
                            "evidence": f"{top.get(main_col)} / {round(total, 2)} total ({top_share}% share)",
                            "severity": "info",
                        })
                        second = groups[1] if len(groups) > 1 else None
                        if second and (second.get(main_col) or 0) > 0:
                            gap = round((top.get(main_col) or 0) / max((second.get(main_col) or 0), 0.001), 2)
                            findings.append({
                                "title": f"{res.get('group_column')} leader gap",
                                "detail": f"{top.get('group')} exceeds the next group ({second.get('group')}) by {gap}x in {main_col}.",
                                "evidence": f"{top.get(main_col)} vs {second.get(main_col)} (ratio {gap})",
                                "severity": "info",
                            })
                    else:
                        for row in groups[:3]:
                            findings.append({
                                "title": f"Group: {row.get('group')}",
                                "detail": _group_label(row, value_cols),
                                "evidence": _group_label(row, value_cols),
                                "severity": "info",
                            })
                stats[f"groups:{res.get('group_column')}"] = groups[:5]
            elif op["tool"] == "calculate_correlation":
                strong = _strong_correlations(res.get("correlation_matrix", []))
                for pair, corr in strong[:4]:
                    findings.append({
                        "title": f"Correlation {pair[0]}↔{pair[1]}",
                        "detail": f"Correlation coefficient {corr}.",
                        "evidence": f"r = {corr}",
                        "severity": "info",
                    })
                stats["correlations"] = strong[:4]
            elif op["tool"] == "analyze_trend":
                for trend in res.get("trends", [])[:2]:
                    points = trend.get("points", [])
                    if len(points) >= 2:
                        first, last = points[0]["value"], points[-1]["value"]
                        if first:
                            change = round((last - first) / abs(first) * 100, 1)
                            findings.append({
                                "title": f"Trend in '{trend['column']}'",
                                "detail": f"Changed {change}% from {first} to {last}.",
                                "evidence": f"{first} -> {last} ({change}%)",
                                "severity": "info",
                            })
                stats["trends"] = res.get("trends", [])
            elif op["tool"] == "detect_anomalies":
                for item in res.get("columns", [])[:3]:
                    if item.get("anomaly_count", 0) > 0:
                        findings.append({
                            "title": f"Anomalies in '{item['column']}'",
                            "detail": f"{item['anomaly_count']} unusual values flagged.",
                            "evidence": f"{item['anomaly_count']} z-scored outliers",
                            "severity": "warning",
                        })

        if not findings:
            findings.append({
                "title": "Basic descriptive analysis completed",
                "detail": "See statistics for a summary.",
                "evidence": "computed statistics",
                "severity": "info",
            })

        return {
            "findings": findings,
            "statistics": stats,
            "summary": f"Found {len(findings)} data-driven insights.",
            "confidence": 0.85,
            "notes": "Deterministic fallback analysis.",
        }


def _heuristic_analysis_plan(ctx: AgentContext) -> list[dict]:
    """Pick sensible default analyses from the dataset profile."""
    ops = []
    profile = ctx.profile or {}
    overview = profile.get("overview", {})
    columns = [str(c) for c in overview.get("column_names", [])]
    col_info = profile.get("columns", {})
    dtypes = overview.get("dtypes", {})

    numeric = [c for c in columns if "float" in str(dtypes.get(c)) or "int" in str(dtypes.get(c))]
    dates = [c for c in columns if "date" in str(dtypes.get(c)).lower() or "date" in c.lower()]
    categoricals = [
        c
        for c in columns
        if c not in numeric
        and "date" not in c.lower()
        and int(profile.get("unique_counts", {}).get("columns", []).count(c)) > -1
    ]
    # refine categoricals from unique counts
    unique_cols = {
        item["column"]: item.get("unique_count", 0)
        for item in profile.get("unique_counts", {}).get("columns", [])
    }
    categoricals = [c for c in categoricals if 2 <= unique_cols.get(c, 999) <= 60]

    value_col = numeric[0] if numeric else None

    # 1. group analysis on any suitable categorical
    for cat in categoricals[:2]:
        if value_col:
            ops.append({
                "tool": "group_by_analysis",
                "arguments": {"group_column": cat, "value_columns": numeric[:3], "agg": "sum"},
                "reason": f"Compare {cat} performance",
            })

    # 2. correlation
    if len(numeric) >= 2:
        ops.append({"tool": "calculate_correlation", "arguments": {}, "reason": "Relationships"})

    # 3. trend
    if dates and value_col:
        ops.append({
            "tool": "analyze_trend",
            "arguments": {"date_column": dates[0], "value_columns": [value_col], "period": "M"},
            "reason": "Trend over time",
        })

    # 4. anomalies
    if value_col:
        ops.append({
            "tool": "detect_anomalies",
            "arguments": {"value_columns": [value_col], "z_threshold": 3.0},
            "reason": "Find unusual values",
        })

    return ops


def _group_label(row: dict, value_cols: list[str]) -> str:
    parts = []
    for v in value_cols:
        if v in row:
            parts.append(f"{v}={row[v]}")
    return " ".join(parts)


def _strong_correlations(matrix: list[dict]) -> list[tuple[tuple[str, str], float]]:
    pairs = []
    if not matrix:
        return pairs
    cols = list(matrix[0].keys())[1:]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = None
            for row in matrix:
                if row.get("column") == a:
                    v = row.get(b)
                    break
            if v is None or isinstance(v, str):
                continue
            v = float(v)
            if abs(v) >= 0.4:
                pairs.append(((a, b), round(v, 3)))
    pairs.sort(key=lambda x: -abs(x[1]))
    return pairs