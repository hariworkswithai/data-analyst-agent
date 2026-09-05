"""Analysis tools: group comparisons, correlations, trends, anomalies.

These functions compute real values with pandas/NumPy. The LLM only
interprets the returned numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.utils.errors import InvalidToolArgumentsError, ToolError


def group_by_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_columns: list[str] | None = None,
    agg: str = "sum",
) -> dict:
    """Aggregate numeric columns grouped by a categorical column."""
    _require_column(df, group_column)
    values = _numeric_columns(df, value_columns) if value_columns else _numeric_columns(df, None)
    values = [v for v in values if v != group_column]

    if not values:
        raise ToolError("No numeric value columns available for grouping.")

    agg = _normalize_agg(agg)
    grouped = df.groupby(group_column, dropna=False)[values].agg(agg)
    # Sort by the first value column for deterministic ordering.
    grouped = grouped.sort_values(by=values[0], ascending=False)

    rows = []
    for idx, group_vals in grouped.iterrows():
        row = {"group": _safe_str(idx), **{k: _safe_float(group_vals[k]) for k in values}}
        rows.append(row)

    return {
        "group_column": group_column,
        "value_columns": values,
        "agg": agg,
        "groups": rows,
        "group_count": len(rows),
        "total": {k: _safe_float(float(df[values[0]].sum())) for k in [values[0]]},
    }


def calculate_correlation(df: pd.DataFrame, method: str = "pearson") -> dict:
    """Correlation matrix for numeric columns."""
    cols = _numeric_columns(df, None)
    if len(cols) < 2:
        return {"columns": cols, "correlation_matrix": [], "note": "Need at least 2 numeric columns."}
    method = method if method in {"pearson", "spearman"} else "pearson"
    corr = df[cols].corr(method=method, numeric_only=True)
    matrix = []
    for c in cols:
        row = {"column": str(c)}
        for o in cols:
            v = corr.loc[c, o]
            row[str(o)] = _safe_float(v)
        matrix.append(row)
    return {"columns": cols, "method": method, "correlation_matrix": matrix}


def calculate_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Compat alias for get_numeric_statistics with a fuller dict."""
    from backend.tools.dataset_tools import get_numeric_statistics

    base = get_numeric_statistics(df, columns)
    return {
        "statistics": base.get("columns", []),
        "note": base.get("note", ""),
    }


def analyze_trend(
    df: pd.DataFrame,
    date_column: str,
    value_columns: list[str],
    period: str = "D",
) -> dict:
    """Aggregate value columns over time (daily/weekly/monthly)."""
    _require_column(df, date_column)
    if not value_columns:
        raise InvalidToolArgumentsError("value_columns is required.")

    pd_col = df[date_column]
    if not pd.api.types.is_datetime64_any_dtype(pd_col):
        converted = pd.to_datetime(df[date_column], errors="coerce")
        if converted.isna().all():
            raise ToolError(
                f"Column '{date_column}' could not be parsed as a date."
            )
        df = df.copy()
        df["_date"] = converted
    else:
        df = df.copy()
        df["_date"] = pd_col

    period = period.upper()
    key = {"D": "D", "W": "W", "M": "M", "Q": "Q"}.get(period, "D")
    df["_period"] = df["_date"].dt.to_period(key).astype(str)

    trend_rows = []
    for value in value_columns:
        if not pd.api.types.is_numeric_dtype(df[value]):
            continue
        series = df.groupby("_period")[value].sum().sort_index()
        trend_rows.append(
            {
                "column": value,
                "points": [
                    {"period": str(k), "value": _safe_float(v)}
                    for k, v in series.items()
                ],
            }
        )
    return {
        "date_column": date_column,
        "period": key,
        "trends": trend_rows,
    }


def detect_anomalies(
    df: pd.DataFrame,
    value_columns: list[str] | None = None,
    z_threshold: float = 3.0,
) -> dict:
    """Flag anomalies using z-scores relative to each column's distribution."""
    cols = _numeric_columns(df, value_columns)
    z_threshold = float(z_threshold)
    results = []
    for c in cols:
        col = df[c].dropna()
        if col.std() == 0 or len(col) < 5:
            continue
        z = (col - col.mean()) / col.std()
        suspicious = z.abs() > z_threshold
        count = int(suspicious.sum())
        results.append(
            {
                "column": str(c),
                "z_threshold": z_threshold,
                "anomaly_count": count,
                "anomaly_percent": round(count / len(col) * 100, 2),
                "examples": [
                    {"value": _safe_float(v), "z_score": round(float(zz), 2)}
                    for v, zz in zip(col[suspicious].head(5), z[suspicious].head(5))
                ],
            }
        )
    return {"columns": results}


def _require_column(df: pd.DataFrame, column: str) -> None:
    if df is None or column not in df.columns:
        raise InvalidToolArgumentsError(f"Column '{column}' does not exist in the dataset.")


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if df is None:
        raise InvalidToolArgumentsError("DataFrame is required.")
    if columns:
        unknown = [c for c in columns if c not in df.columns]
        if unknown:
            raise InvalidToolArgumentsError(f"Unknown columns: {unknown}")
        return [str(c) for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _normalize_agg(agg: str) -> str:
    aliases = {
        "sum": "sum",
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "count": "count",
        "median": "median",
        "max": "max",
        "min": "min",
        "std": "std",
    }
    if agg not in aliases:
        raise InvalidToolArgumentsError(
            f"Invalid aggregation '{agg}'. Use one of: sum, mean, count, median, max, min, std."
        )
    return aliases[agg]


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