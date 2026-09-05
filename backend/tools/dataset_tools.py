"""Dataset inspection and profiling tools.

All numerical values are computed with pandas/NumPy. The LLM never
invents statistics; it only interprets what these functions return.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.utils.errors import InvalidToolArgumentsError, ToolError

MAX_SAMPLE_ROWS = 5
UNIQUE_LIST_LIMIT = 20


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return high-level dataset metadata."""
    if df is None or df.empty:
        raise ToolError("Cannot inspect an empty dataset.")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
    }


def get_column_info(df: pd.DataFrame, column: str) -> dict:
    """Return type, non-null count, unique count, and range for a column."""
    _require_column(df, column)
    col = df[column]
    nunique = int(col.nunique(dropna=True))
    na_count = int(col.isna().sum())
    dtype = str(col.dtype)

    if pd.api.types.is_numeric_dtype(col):
        stats = _numeric_summary(col)
        kind = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(col):
        stats = {
            "min": _safe_str(col.min()),
            "max": _safe_str(col.max()),
        }
        kind = "datetime"
    else:
        stats = {
            "top": _safe_str(col.mode().iloc[0]) if not col.mode().empty else None,
            "top_frequency": int(col.value_counts(dropna=True).iloc[0])
            if col.notna().any()
            else 0,
        }
        kind = "categorical/object"

    return {
        "column": column,
        "dtype": dtype,
        "kind": kind,
        "non_null": int(col.notna().sum()),
        "null_count": na_count,
        "unique_count": nunique,
        **stats,
    }


def get_missing_values(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Return missing-value counts for each column (optionally filtered)."""
    if df is None:
        raise InvalidToolArgumentsError("DataFrame is required.")
    cols = list(df.columns) if columns is None else _require_columns(df, columns)
    missing = df[cols].isna().sum().to_dict()
    return {
        "total_missing": int(sum(missing.values())),
        "columns_with_missing": [
            {"column": str(c), "count": int(v)}
            for c, v in missing.items()
            if v > 0
        ],
    }


def get_duplicate_count(df: pd.DataFrame) -> dict:
    """Return the number of fully duplicated rows."""
    if df is None:
        raise InvalidToolArgumentsError("DataFrame is required.")
    dupes = int(df.duplicated(keep=False).sum())
    return {"duplicate_row_count": dupes, "duplicate_groups": int(df.duplicated(keep=False).sum() > 0)}


def get_unique_counts(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Return unique value counts, flag constant and high-cardinality columns."""
    cols = list(df.columns) if columns is None else _require_columns(df, columns)
    rows = int(df.shape[0])
    result = []
    for c in cols:
        nunique = int(df[c].nunique(dropna=True))
        result.append(
            {
                "column": str(c),
                "unique_count": nunique,
                "constant": nunique <= 1,
                "high_cardinality": rows > 0 and nunique / rows > 0.95 and rows > 10,
            }
        )
    return {"columns": result}


def get_numeric_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Descriptive statistics for numeric columns."""
    cols = _numeric_columns(df, columns)
    if not cols:
        return {"columns": [], "note": "No numeric columns found."}
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    result = []
    for c in cols:
        row = stats.loc[c]
        result.append(
            {
                "column": str(c),
                "count": int(row["count"]),
                "mean": _safe_float(row["mean"]),
                "std": _safe_float(row["std"]),
                "min": _safe_float(row["min"]),
                "q25": _safe_float(row["25%"]),
                "median": _safe_float(row["50%"]),
                "q75": _safe_float(row["75%"]),
                "max": _safe_float(row["max"]),
                "iqr": _safe_float(row["75%"] - row["25%"]),
            }
        )
    return {"columns": result}


def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Detect outliers using the IQR method (1.5x rule)."""
    cols = _numeric_columns(df, columns)
    result = []
    for c in cols:
        col = df[c].dropna()
        if len(col) < 4 or col.nunique() < 3:
            continue
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (col < lower) | (col > upper)
        count = int(mask.sum())
        pct = round(count / len(col) * 100, 2) if len(col) else 0.0
        result.append(
            {
                "column": str(c),
                "outlier_count": count,
                "outlier_percent": pct,
                "lower_bound": _safe_float(lower),
                "upper_bound": _safe_float(upper),
                "min": _safe_float(col.min()),
                "max": _safe_float(col.max()),
            }
        )
    return {"columns": result}


def detect_category_inconsistencies(df: pd.DataFrame) -> dict:
    """Find categorical columns with near-duplicate category labels.

    Flags formatting inconsistencies like 'North' vs 'north' which would
    silently split a category during grouping.
    """
    inconsistencies = []
    for c in df.columns:
        col = df[c].dropna()
        if not len(col):
            continue
        if pd.api.types.is_numeric_dtype(col):
            continue
        if col.nunique() > 500:
            continue
        labels = [str(v).strip() for v in col.unique()]
        seen: dict[str, str] = {}
        for label in labels:
            folded = label.casefold()
            if folded in seen and seen[folded] != label:
                inconsistencies.append({
                    "column": str(c),
                    "labels": sorted({seen[folded], label}),
                    "counts": {
                        seen[folded]: int((col.astype(str).str.strip() == seen[folded]).sum()),
                        label: int((col.astype(str).str.strip() == label).sum()),
                    },
                    "recommendation": "Normalize case/spacing so categories group correctly.",
                })
            else:
                seen[folded] = label
    return {"columns": inconsistencies}


def get_sample(df: pd.DataFrame, n: int = MAX_SAMPLE_ROWS) -> list[dict]:
    """Return a small, safe sample of rows for the LLM to inspect."""
    n = max(1, min(int(n), MAX_SAMPLE_ROWS))
    sample = df.head(n).where(pd.notna(df), None)
    return sample.to_dict(orient="records")


def _require_column(df: pd.DataFrame, column: str) -> None:
    if df is None or column not in df.columns:
        raise InvalidToolArgumentsError(f"Column '{column}' does not exist in the dataset.")
    if column in df.columns and df[column].dtype == object:
        # coerce string-only columns
        pass


def _require_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df is None or not columns:
        raise InvalidToolArgumentsError("A non-empty column list is required.")
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise InvalidToolArgumentsError(
            f"Unknown columns: {unknown}. Available: {list(df.columns)}"
        )
    return columns


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if df is None:
        raise InvalidToolArgumentsError("DataFrame is required.")
    if columns:
        _require_columns(df, columns)
        return [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _numeric_summary(col: pd.Series) -> dict:
    try:
        s = col.dropna()
        return {
            "min": _safe_float(s.min()),
            "max": _safe_float(s.max()),
            "mean": _safe_float(s.mean()),
            "median": _safe_float(s.median()),
            "std": _safe_float(s.std()),
        }
    except Exception:
        return {}


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_str(v) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _json_safe(obj):
    """Best-effort JSON-safe conversion."""
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj