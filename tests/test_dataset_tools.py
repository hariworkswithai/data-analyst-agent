"""Tests for the controlled dataset tools."""

import pytest

from backend.tools.dataset_tools import (
    detect_category_inconsistencies,
    detect_outliers,
    get_duplicate_count,
    get_missing_values,
    get_numeric_statistics,
    get_unique_counts,
    inspect_dataset,
)
from backend.utils.errors import InvalidToolArgumentsError


def test_inspect_dataset(sales_df):
    meta = inspect_dataset(sales_df)
    assert meta["rows"] == 40
    assert meta["columns"] == 6
    assert "Sales" in meta["column_names"]


def test_missing_values(messy_df):
    res = get_missing_values(messy_df)
    assert res["total_missing"] >= 1
    assert any(m["column"] == "value" and m["count"] >= 1 for m in res["columns_with_missing"])


def test_duplicates(messy_df):
    res = get_duplicate_count(messy_df)
    assert res["duplicate_row_count"] >= 2  # the copy of row 3 gives keep=False pair


def test_unique_counts_constant(messy_df):
    res = get_unique_counts(messy_df)
    const = [c for c in res["columns"] if c["constant"]]
    assert any(c["column"] == "const" for c in const)


def test_category_inconsistencies(messy_df):
    res = detect_category_inconsistencies(messy_df)
    found = [c for c in res["columns"] if c["column"] == "category"]
    assert found, "expected a category inconsistency to be flagged"
    labels = [l.lower() for l in found[0]["labels"]]
    assert "north" in labels


def test_category_inconsistencies_ignores_numeric(messy_df):
    from backend.tools.dataset_tools import detect_category_inconsistencies

    res = detect_category_inconsistencies(messy_df)
    numeric_cols = {c["column"] for c in res["columns"]}
    assert "value" not in numeric_cols


def test_numeric_statistics(sales_df):
    res = get_numeric_statistics(sales_df)
    cols = {c["column"]: c for c in res["columns"]}
    assert cols["Sales"]["mean"] == pytest.approx(2950.0, rel=1e-2)


def test_outliers(messy_df):
    res = detect_outliers(messy_df)
    value = [c for c in res["columns"] if c["column"] == "value"]
    assert value and value[0]["outlier_count"] >= 1  # 999 should be flagged


def test_unknown_column_rejected(sales_df):
    with pytest.raises(InvalidToolArgumentsError):
        get_missing_values(sales_df, columns=["nope"])