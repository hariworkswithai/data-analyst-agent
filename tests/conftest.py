"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import BASE_DIR  # noqa: E402


@pytest.fixture
def sales_df():
    rng = pd.date_range("2024-01-01", periods=40, freq="W")
    return pd.DataFrame({
        "Date": rng,
        "Region": ["North", "South", "East", "West"] * 10,
        "Product": ["A", "B", "C", "D"] * 10,
        "Sales": [round(1000 + i * 100, 2) for i in range(40)],
        "Profit": [round(100 + i * 10, 2) for i in range(40)],
        "Quantity": [i % 20 + 1 for i in range(40)],
    })


@pytest.fixture
def messy_df():
    """Dataset with known quality issues."""
    df = pd.DataFrame({
        "id": list(range(12)),
        "name": ["a", "b", "c", "a", "e", "f", "g", "h", "i", "j", "k", "l"],
        "value": [1.0, 2.0, 3.0, None, 5.0, 999.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "const": ["x"] * 12,
        "category": ["north", "South", "East", "North", "West", "north", "South", "West", "North", "East", "West", "North"],
    })
    # one duplicate row
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    return df


@pytest.fixture
def sample_csv_path():
    p = BASE_DIR / "sample_data" / "sales.csv"
    if not p.exists():
        pytest.skip("sample_data/sales.csv not present")
    return p