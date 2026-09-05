"""CLEANER AGENT - investigates data quality.

Checks missing values, duplicates, types, constant columns, outliers,
and formats. Uses the controlled dataset tools; never modifies the
original dataset.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.agents.base import AgentContext, ToolPlanAgent, findings_validator
from backend.tools.registry import try_execute_tool
from backend.utils.errors import LLMError

ALLOWED_OUTPUT = {
    "findings", "summary", "confidence", "issues", "recommended_fixes",
    "safe_to_automate", "notes",
}

QUALITY_PLAN = [
    {"tool": "get_missing_values", "arguments": {}, "reason": "Detect missing values."},
    {"tool": "get_duplicate_count", "arguments": {}, "reason": "Detect duplicate rows."},
    {"tool": "get_unique_counts", "arguments": {}, "reason": "Find constant or high-cardinality columns."},
    {"tool": "detect_outliers", "arguments": {}, "reason": "Detect outliers in numeric columns."},
    {"tool": "get_numeric_statistics", "arguments": {}, "reason": "Check numeric ranges and validity."},
    {"tool": "detect_category_inconsistencies", "arguments": {}, "reason": "Find near-duplicate category labels."},
]


class CleanerAgent(ToolPlanAgent):
    name = "cleaner"
    label = "Cleaner Agent"

    def build_plan_prompt(self, ctx: AgentContext) -> str:
        return f"""You are the Data Cleaner/Quality agent.
Dataset profile:
{ctx.profile_text}

User question: {ctx.question}

Decide which quality checks matter. Choose tools from the list below.
Return a JSON object:
{{"reasoning": "...", "operations": [{{"tool": "...", "arguments": {{...}}, "reason": "..."}}]}}

Available tools: inspect_dataset, get_column_info, get_missing_values,
get_duplicate_count, get_unique_counts, get_numeric_statistics,
detect_outliers, get_sample.

Tool arguments are JSON objects of parameter names. Never invent numbers.
Never modify any data - this is read-only.

Example:
{{"reasoning": "Checking core quality dimensions",
 "operations": [
   {{"tool": "get_missing_values", "arguments": {{}}, "reason": "Count missing values per column"}},
   {{"tool": "get_duplicate_count", "arguments": {{}}, "reason": "Count duplicate rows"}}
 ]}}"""

    def build_interpret_prompt(self, ctx: AgentContext, tool_outcomes: list[dict]) -> str:
        compact = json.dumps(tool_outcomes, default=str, ensure_ascii=False)
        return f"""You are the Data Cleaner. Interpret the following REAL tool results.
Only use the numbers returned. Never invent statistics.

Tool outcomes:
{compact}

Return strict JSON:
{{"findings": [{{"title": "...", "detail": "...", "evidence": "numbers from results", "severity": "error|warning|info"}}],
  "recommended_fixes": [{{"issue": "...", "fix": "...", "safe_to_automate": true}}],
  "summary": "one-line data-quality conclusion",
  "confidence": 0.0-1.0}}

Include findings ONLY supported by the tool results."""

    def validate_interpret(self, data):
        return findings_validator(data)

    def default_plan(self, ctx: AgentContext) -> dict:
        return {"operations": QUALITY_PLAN}

    def deterministic_result(self, ctx: AgentContext) -> dict:
        findings = []
        issues = []
        fixes = []
        safe = True

        for op in self.default_plan(ctx)["operations"]:
            outcome = try_execute_tool(ctx.df, op["tool"], op["arguments"])
            if not outcome["ok"]:
                continue
            res = outcome["result"]

        missing = try_execute_tool(ctx.df, "get_missing_values", {})["result"]
        if missing["total_missing"] > 0:
            for m in missing["columns_with_missing"][:5]:
                severity = "error" if m["count"] > ctx.df.shape[0] * 0.5 else "warning"
                findings.append({
                    "title": f"Missing values in '{m['column']}'",
                    "detail": f"{m['count']} missing values.",
                    "evidence": f"{m['count']} nulls",
                    "severity": severity,
                })
                issues.append(f"Possible missing values in '{m['column']}'")
                fixes.append({"issue": m["column"], "fix": "Impute or drop nulls", "safe_to_automate": True})

        dupes = try_execute_tool(ctx.df, "get_duplicate_count", {})["result"]
        if dupes["duplicate_row_count"] > 0:
            findings.append({
                "title": "Duplicate rows present",
                "detail": f"{dupes['duplicate_row_count']} duplicated rows found.",
                "evidence": f"{dupes['duplicate_row_count']} duplicate rows",
                "severity": "warning",
            })
            fixes.append({"issue": "duplicate rows", "fix": "Drop duplicates", "safe_to_automate": True})

        uniques = try_execute_tool(ctx.df, "get_unique_counts", {})["result"]
        numeric_cols = {
            c
            for c in ctx.df.columns
            if pd.api.types.is_numeric_dtype(ctx.df[c])
        }
        temporal_tokens = ("date", "time", "day", "month", "year", "timestamp", "period")
        for c in uniques.get("columns", []):
            if c.get("constant"):
                findings.append({
                    "title": f"Constant column '{c['column']}'",
                    "detail": "Single unique value; adds no information.",
                    "evidence": "1 unique value",
                    "severity": "info",
                })
                fixes.append({"issue": c["column"], "fix": "Consider dropping", "safe_to_automate": True})
            elif (
                c.get("high_cardinality")
                and c["column"] not in numeric_cols
                and not any(t in str(c["column"]).lower() for t in temporal_tokens)
            ):
                # High-cardinality continuous numeric columns are normal;
                # only flag object columns that look like IDs.
                findings.append({
                    "title": f"High-cardinality column '{c['column']}'",
                    "detail": "Mostly unique values; likely an ID field.",
                    "evidence": f"{c['unique_count']} unique values",
                    "severity": "info",
                })

        outliers = try_execute_tool(ctx.df, "detect_outliers", {})["result"]
        for o in outliers.get("columns", [])[:5]:
            if o.get("outlier_count", 0) > 0:
                sev = "warning" if o["outlier_percent"] > 5 else "info"
                findings.append({
                    "title": f"Outliers in '{o['column']}'",
                    "detail": f"{o['outlier_count']} outliers ({o['outlier_percent']}% of rows).",
                    "evidence": f"{o['outlier_count']} values outside IQR bounds",
                    "severity": sev,
                })
                issues.append(f"Suspicious outliers in '{o['column']}'")

        inconsistencies = try_execute_tool(ctx.df, "detect_category_inconsistencies", {})["result"]
        for inc in inconsistencies.get("columns", []):
            labels = " vs ".join(inc.get("labels", []))
            findings.append({
                "title": f"Inconsistent categories in '{inc['column']}'",
                "detail": f"'{labels}' group separately despite being near-duplicates.",
                "evidence": f"labels={inc.get('labels')} counts={inc.get('counts')}",
                "severity": "warning",
            })
            fixes.append({
                "issue": inc["column"],
                "fix": "Normalize category casing/spacing",
                "safe_to_automate": True,
            })
            issues.append(f"Formatting inconsistency in '{inc['column']}'")

        if not findings:
            findings.append({
                "title": "No major data quality issues",
                "detail": "Dataset looks clean on core checks.",
                "evidence": "quality checks",
                "severity": "info",
            })
            safe = False

        summary = f"{len(findings)} data quality observations."
        return {
            "findings": findings,
            "recommended_fixes": fixes,
            "safe_to_automate": safe,
            "issues": issues,
            "summary": summary,
            "confidence": 0.9,
            "notes": "Deterministic fallback analysis.",
        }