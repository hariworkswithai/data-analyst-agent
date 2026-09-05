"""Builds a compact, safe summary of a dataset for LLM consumption.

The full dataset is never sent to an LLM. Python computes everything;
the profile includes metadata, statistics, and small samples only.
"""

from __future__ import annotations

import pandas as pd

from backend.tools.dataset_tools import (
    detect_outliers,
    get_column_info,
    get_duplicate_count,
    get_missing_values,
    inspect_dataset,
    get_unique_counts,
)
from backend.tools.registry import (
    execute_tool,
    try_execute_tool,
)


def compute_dataset_profile(df: pd.DataFrame) -> dict:
    """Compute a complete profile used by agents and the report."""
    overview = inspect_dataset(df)
    missing = get_missing_values(df)
    duplicates = get_duplicate_count(df)
    unique = get_unique_counts(df)
    outliers = detect_outliers(df)

    column_info = {}
    for col in df.columns:
        try:
            info = get_column_info(df, str(col))
        except Exception:  # noqa: BLE001 - skip unproxyable columns
            info = {"column": str(col), "dtype": str(df[col].dtype), "error": True}
        column_info[str(col)] = info

    quality = _quality_score(df, missing, duplicates, outliers)
    head = df.head(5).where(pd.notna(df), None).to_dict(orient="records")

    return {
        "overview": overview,
        "columns": column_info,
        "missing_values": missing,
        "duplicates": duplicates,
        "unique_counts": unique,
        "outliers": outliers,
        "quality_score": quality,
        "sample_rows": head,
    }


def _quality_score(
    df: pd.DataFrame,
    missing: dict,
    duplicates: dict,
    outliers: dict,
) -> dict:
    """Compute a 0-100 data quality score with a short label."""
    total_cells = int(df.size) or 1
    missing_cells = int(missing["total_missing"])
    missing_ratio = missing_cells / total_cells

    dup_flag = duplicates["duplicate_row_count"] > 0
    outlier_ratio = 0.0
    for item in outliers.get("columns", []):
        outlier_ratio = max(outlier_ratio, item.get("outlier_percent", 0.0) / 100.0)

    score = 100.0
    score -= missing_ratio * 100
    if dup_flag:
        score -= 10
    if outlier_ratio > 0.02:
        score -= 10
    score = max(0, round(score))

    if score >= 90:
        label = "Good"
    elif score >= 70:
        label = "Fair"
    else:
        label = "Needs Cleaning"

    return {"score": score, "label": label}


def profile_to_llm_text(profile: dict, max_lines: int = 120) -> str:
    """Serialize a profile to compact text for LLM system/user messages."""
    import json

    def compact(o):
        return json.dumps(o, default=str, ensure_ascii=False)[:200]

    lines = []
    o = profile.get("overview", {})
    lines.append(f"[ROWS]={o.get('rows')} [COLUMNS]={o.get('columns')} "
                 f"[DTYPES]={compact(o.get('dtypes'))}")
    lines.append(f"[QUALITY]={profile.get('quality_score', {}).get('score')} "
                 f"({profile.get('quality_score', {}).get('label')})")
    lines.append(f"[MISSING]={compact(profile.get('missing_values'))}")
    lines.append(f"[DUPLICATES]={compact(profile.get('duplicates'))}")
    lines.append(f"[OUTLIERS]={compact(profile.get('outliers'))}")

    for col, info in profile.get("columns", {}).items():
        lines.append(f"[COL:{col}] {compact(info)}")

    lines.append(f"[SAMPLE]={compact(profile.get('sample_rows'))}")
    truncated = lines[:max_lines]
    if len(lines) > max_lines:
        truncated.append("... (truncated)")
    return "\n".join(truncated)


def dataset_tools_summary() -> str:
    """Plain-text list of tools for LLM prompts."""
    from backend.tools.registry import TOOL_DESCRIPTIONS

    lines = []
    for name, info in sorted(TOOL_DESCRIPTIONS.items()):
        params = ", ".join(sorted(info["parameters"]))
        lines.append(f"- {name}({params}): {info['description']}")
    return "\n".join(lines)