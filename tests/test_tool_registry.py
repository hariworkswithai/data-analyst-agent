"""Tests for controlled tool execution via the registry."""

import pytest

from backend.tools.registry import (
    execute_tool,
    list_tools,
    try_execute_tool,
)
from backend.utils.errors import InvalidToolArgumentsError, UnknownToolError


def test_list_tools_contains_expected():
    tools = list_tools()
    for name in (
        "inspect_dataset",
        "get_missing_values",
        "get_duplicate_count",
        "group_by_analysis",
        "calculate_correlation",
        "analyze_trend",
        "create_bar_chart",
        "create_heatmap",
        "detect_outliers",
    ):
        assert name in tools


def test_execute_group(sales_df):
    res = execute_tool(
        sales_df,
        "group_by_analysis",
        {"group_column": "Region", "value_columns": ["Sales"], "agg": "sum"},
    )
    assert len(res["groups"]) == 4
    totals = {g["group"]: g["Sales"] for g in res["groups"]}
    assert totals["North"] == pytest.approx(sum(sales_df[sales_df.Region == "North"].Sales))


def test_unknown_tool_rejected(sales_df):
    with pytest.raises(UnknownToolError):
        execute_tool(sales_df, "system.run", {})


def test_invalid_arguments_rejected(sales_df):
    with pytest.raises(InvalidToolArgumentsError):
        execute_tool(sales_df, "get_missing_values", {"columns": "not-a-list"})


def test_invalid_agg_rejected(sales_df):
    with pytest.raises(InvalidToolArgumentsError):
        execute_tool(sales_df, "group_by_analysis", {"group_column": "Region", "agg": "SYSTEM()"})


def test_try_execute_reports_failure_not_raises(sales_df):
    outcome = try_execute_tool(sales_df, "not_a_tool", {})
    assert outcome["ok"] is False
    assert "not_a_tool" in outcome["error"]


def test_execute_tool_injects_df_only(sales_df):
    # df is never accepted from an agent's arguments
    with pytest.raises(InvalidToolArgumentsError):
        execute_tool(sales_df, "inspect_dataset", {"df": "<payload>"})