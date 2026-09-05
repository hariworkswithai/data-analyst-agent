"""Unit tests for dataset loading and validation."""

import io

import pandas as pd
import pytest

from backend.utils.errors import DatasetValidationError
from backend.utils.validation import validate_and_load

from tests.conftest import sys  # noqa: F401


def _write_tmp_csv(tmp_path, df: pd.DataFrame, name: str = "test.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_load_valid_csv(tmp_path, sales_df):
    path = _write_tmp_csv(tmp_path, sales_df)
    loaded = validate_and_load(path, "sales.csv")
    assert loaded.rows == 40
    assert loaded.columns == 6


def test_reject_non_csv(tmp_path, sales_df):
    path = _write_tmp_csv(tmp_path, sales_df, "data.txt")
    with pytest.raises(DatasetValidationError, match="Unsupported file type"):
        validate_and_load(path, "data.txt")


def test_reject_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("col1,col2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_and_load(str(p), "empty.csv")


def test_reject_duplicate_columns(tmp_path):
    p = tmp_path / "dupes.csv"
    p.write_text("A,A,B,C,D,E\n1,2,3,4,5,6\n7,8,9,10,11,12\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate column"):
        validate_and_load(str(p), "dupes.csv")


def test_reject_too_many_rows(tmp_path):
    big = pd.DataFrame({"x": range(300000), "y": range(300000)})
    big.to_csv(tmp_path / "big.csv", index=False)
    with pytest.raises(DatasetValidationError, match="max"):
        validate_and_load(str(tmp_path / "big.csv"), "big.csv")


def test_bad_csv_lines_skipped(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n4,5\n6,7,8\n", encoding="utf-8")
    loaded = validate_and_load(str(p), "bad.csv")
    assert loaded.rows <= 3  # malformed line is skipped safely