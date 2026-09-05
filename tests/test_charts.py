"""Tests for URL-safe chart generation."""

from backend.config import BASE_DIR
from backend.services.jobs import job_exists  # noqa: F401 (module import check)
from backend.tools.registry import execute_tool, try_execute_tool


def _chart_variant(df, tool, kwargs):
    res = execute_tool(df, tool, kwargs)
    assert res["relative_path"].startswith("charts/")
    p = BASE_DIR / "generated" / res["relative_path"]
    assert p.exists()
    assert p.stat().st_size > 500
    return res


def test_bar_chart(sales_df):
    res = _chart_variant(sales_df, "create_bar_chart",
                         {"x_column": "Region", "y_column": "Sales", "agg": "sum"})
    assert res["chart_type"] == "bar"


def test_line_chart(sales_df):
    res = _chart_variant(sales_df, "create_line_chart",
                         {"x_column": "Date", "y_column": "Sales"})
    assert res["chart_type"] == "line"


def test_histogram(sales_df):
    res = _chart_variant(sales_df, "create_histogram", {"column": "Sales", "bins": 10})
    assert res["chart_type"] == "histogram"


def test_scatter(sales_df):
    res = _chart_variant(sales_df, "create_scatter_plot",
                         {"x_column": "Sales", "y_column": "Profit"})
    assert res["chart_type"] == "scatter"


def test_box(sales_df):
    res = _chart_variant(sales_df, "create_box_plot",
                         {"y_column": "Sales", "x_column": "Region"})
    assert res["chart_type"] == "box"


def test_heatmap(sales_df):
    res = _chart_variant(sales_df, "create_heatmap", {})
    assert res["chart_type"] == "heatmap"


def test_chart_invalid_column(sales_df):
    outcome = try_execute_tool(sales_df, "create_bar_chart",
                               {"x_column": "Nope", "y_column": "Sales"})
    assert outcome["ok"] is False


def test_histogram_on_text_rejected(sales_df):
    outcome = try_execute_tool(sales_df, "create_histogram", {"column": "Region"})
    assert outcome["ok"] is False