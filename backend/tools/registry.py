"""Registry of all safe tools available to agents.

Agents only call tools by name with JSON-safe arguments. The backend
resolves the function, validates arguments, executes it, and returns
structured results. No agent can execute arbitrary code or access
arbitrary filesystem paths.
"""

from __future__ import annotations

import inspect
import json
from typing import Callable

import pandas as pd

from backend.utils.errors import (
    InvalidToolArgumentsError,
    ToolError,
    UnknownToolError,
)

from backend.tools import dataset_tools, analysis_tools, visualization_tools


def _safe_row_limit(n) -> int:
    try:
        return max(0, min(int(n), dataset_tools.MAX_SAMPLE_ROWS))
    except (TypeError, ValueError):
        return dataset_tools.MAX_SAMPLE_ROWS


def _register() -> dict[str, Callable]:
    """Build the tool name -> function map."""
    d = {
        # Dataset tools
        "inspect_dataset": dataset_tools.inspect_dataset,
        "get_column_info": dataset_tools.get_column_info,
        "get_missing_values": dataset_tools.get_missing_values,
        "get_duplicate_count": dataset_tools.get_duplicate_count,
        "get_unique_counts": dataset_tools.get_unique_counts,
        "get_numeric_statistics": dataset_tools.get_numeric_statistics,
        "detect_outliers": dataset_tools.detect_outliers,
        "detect_category_inconsistencies": dataset_tools.detect_category_inconsistencies,
        "get_sample": dataset_tools.get_sample,
        # Analysis tools
        "group_by_analysis": analysis_tools.group_by_analysis,
        "calculate_correlation": analysis_tools.calculate_correlation,
        "calculate_statistics": analysis_tools.calculate_statistics,
        "analyze_trend": analysis_tools.analyze_trend,
        "detect_anomalies": analysis_tools.detect_anomalies,
        # Visualization tools
        "create_bar_chart": visualization_tools.create_bar_chart,
        "create_line_chart": visualization_tools.create_line_chart,
        "create_histogram": visualization_tools.create_histogram,
        "create_scatter_plot": visualization_tools.create_scatter_plot,
        "create_box_plot": visualization_tools.create_box_plot,
        "create_heatmap": visualization_tools.create_heatmap,
    }
    return d


TOOL_REGISTRY: dict[str, Callable] = _register()

# Describe each tool's signature for the LLM.
TOOL_DESCRIPTIONS: dict[str, dict] = {}


def _describe_tools() -> dict[str, dict]:
    descriptions = {}
    for name, fn in TOOL_REGISTRY.items():
        sig = inspect.signature(fn)
        params = {
            p: {
                "type": "string or list",
                "required": param.default is inspect.Parameter.empty,
            }
            for p, param in sig.parameters.items()
            if p != "df"
        }
        doc = (fn.__doc__ or "").strip().splitlines()
        descriptions[name] = {
            "description": doc[0] if doc else name.replace("_", " "),
            "parameters": params,
        }
    return descriptions


TOOL_DESCRIPTIONS = _describe_tools()


def list_tools() -> dict[str, dict]:
    """Public listing of available tools with signatures (no code)."""
    return TOOL_DESCRIPTIONS


def execute_tool(
    df: pd.DataFrame,
    tool_name: str,
    arguments: dict | None,
) -> dict:
    """Validate and execute a single tool call.

    Raises UnknownToolError for unknown tools and ToolError for invalid
    calls. The DataFrame is injected by the backend, never by an agent.
    """
    if tool_name not in TOOL_REGISTRY:
        raise UnknownToolError(
            f"Unknown tool '{tool_name}'. Available: {sorted(TOOL_REGISTRY)}"
        )

    fn = TOOL_REGISTRY[tool_name]
    args = _sanitize_and_validate(tool_name, arguments)
    try:
        result = fn(df, **args)
    except (ToolError, InvalidToolArgumentsError) as exc:
        raise exc
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Tool '{tool_name}' failed: {exc}") from exc

    if not isinstance(result, dict):
        raise ToolError(f"Tool '{tool_name}' returned a non-dict result.")
    return result


def _sanitize_and_validate(tool_name: str, arguments: dict | None) -> dict:
    """Coerce agent-supplied arguments to Python primitives.

    The DataFrame is NEVER accepted from an agent - it is injected by the
    backend only. Any attempt to pass 'df' (or similar internals) is rejected.
    """
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise InvalidToolArgumentsError("Tool arguments must be a JSON object.")

    cleaned: dict = {}
    for key, value in arguments.items():
        if key in {"df", "path", "filename"}:
            raise InvalidToolArgumentsError(
                f"Argument '{key}' is reserved for the backend and cannot be set by an agent."
            )
        t = type(value).__name__
        if value is None or isinstance(value, (str, int, float, bool, list)):
            cleaned[key] = value
        else:
            raise InvalidToolArgumentsError(
                f"Argument '{key}' has unsupported type '{t}'. "
                "Only JSON-safe primitives are allowed."
            )
    return cleaned


def try_execute_tool(
    df: pd.DataFrame,
    tool_name: str,
    arguments: dict | None,
    max_retries: int = 2,
) -> dict:
    """Execute a tool with retries for transient/validation failures."""
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            result = execute_tool(df, tool_name, arguments)
            return {
                "ok": True,
                "tool": tool_name,
                "result": result,
                "attempts": attempt + 1,
            }
        except (UnknownToolError, InvalidToolArgumentsError, ToolError) as exc:
            last_error = str(exc)
            if isinstance(exc, UnknownToolError):
                break
    return {
        "ok": False,
        "tool": tool_name,
        "error": last_error or "Unknown failure",
        "attempts": attempt + 1,
    }