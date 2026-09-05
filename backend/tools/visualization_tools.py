"""Controlled chart generation tools.

These functions produce PNG charts into the configured chart directory.
The LLM only decides *which* chart to create and *what* it means;
the Python layer owns the actual drawing and all numerical handling.
"""

from __future__ import annotations

import uuid

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.config import CHART_DIR
from backend.utils.errors import InvalidToolArgumentsError, ToolError

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

# Keep colormaps stable for categoricals.
_COLOR_CYCLE = plt.get_cmap("Set2").colors


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None = None,
    group_column: str | None = None,
    agg: str = "sum",
    title: str = "Bar Chart",
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """Bar chart comparing categories. Optionally grouped."""
    _require_column(df, x_column)
    agg = _normalize_agg(agg)
    if y_column is None:
        data = df[x_column].value_counts(dropna=False).sort_values(ascending=False)
        labels = [str(k) for k in data.index]
        values = [float(v) for v in data.values]
        hue = None
    else:
        _require_column(df, y_column)
        if group_column:
            _require_column(df, group_column)
            pivot = (
                df.groupby([group_column, x_column])[y_column]
                .agg(agg)
                .unstack(fill_value=0)
            )
            labels = [str(c) for c in pivot.columns]
            values = pivot.values.T
            hue = [str(c) for c in pivot.index]
        else:
            grp = df.groupby(x_column, dropna=False)[y_column].agg(agg).sort_values(ascending=False)
            labels = [str(k) for k in grp.index]
            values = [float(v) for v in grp.values]
            hue = None

    _limit_labels(labels, values, hue)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels) + 3), 5))
    x = np.arange(len(labels))
    total_width = 0.8
    if hue:
        n = len(hue)
        width = total_width / max(n, 1)
        for i, h in enumerate(hue):
            offset = (i - (n - 1) / 2) * width
            row = values[i] if values.ndim > 1 else values
            ax.bar(x + offset, [float(v) for v in row], width, label=str(h))
        ax.legend(title=group_column, fontsize=8)
    else:
        ax.bar(x, values, color=_COLOR_CYCLE[: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or (y_column or "count"))
    fig.tight_layout()
    return _save_figure(fig, {"type": "bar", "x": x_column, "y": y_column, "title": title})


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str | None = None,
    agg: str = "sum",
    title: str = "Line Chart",
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """Time/ordered line chart for one or more series."""
    _require_column(df, x_column)
    _require_column(df, y_column)
    agg = _normalize_agg(agg)

    x_dates = _try_to_datetime(df[x_column])
    dfc = df.copy()
    dfc["_x"] = x_dates if x_dates is not None else df[x_column]

    fig, ax = plt.subplots(figsize=(8, 5))
    if group_column:
        _require_column(df, group_column)
        for i, (g, sub) in enumerate(dfc.groupby(group_column, dropna=False)):
            series = sub.groupby(pd.Grouper(key="_x", freq="D") if x_dates is not None else sub["_x"])[y_column].agg(agg)
            if x_dates is not None:
                series = sub.set_index("_x")[y_column].resample("D").agg(agg)
            ax.plot(series.index, series.values, marker="o", ms=3, label=str(g), color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)])
        ax.legend(fontsize=8)
    else:
        series = dfc.set_index("_x")[y_column].sort_index()
        if x_dates is not None:
            series = series.resample("D").agg(agg)
        ax.plot(series.index, series.values, marker="o", ms=3, color=_COLOR_CYCLE[0])

    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or y_column)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_figure(fig, {"type": "line", "x": x_column, "y": y_column, "title": title})


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str = "Histogram",
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """Distribution of a numeric column."""
    _require_column(df, column)
    col = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        raise ToolError(f"Column '{column}' is not numeric; cannot plot a histogram.")
    bins = max(5, min(int(bins), 100))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(col, bins=bins, color=_COLOR_CYCLE[0], edgecolor="white", alpha=0.9)
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label or column)
    ax.set_ylabel(y_label or "Frequency")
    fig.tight_layout()
    return _save_figure(fig, {"type": "histogram", "column": column, "title": title})


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
    title: str = "Scatter Plot",
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """Scatter plot of two numeric columns, optionally colored."""
    _require_column(df, x_column)
    _require_column(df, y_column)
    for c in (x_column, y_column):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ToolError(f"Column '{c}' is not numeric; cannot plot a scatter.")

    fig, ax = plt.subplots(figsize=(7, 5))
    if color_column:
        _require_column(df, color_column)
        if pd.api.types.is_numeric_dtype(df[color_column]):
            sc = ax.scatter(df[x_column], df[y_column], c=df[color_column], s=40, alpha=0.7, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=color_column)
        else:
            for i, (g, sub) in enumerate(df.groupby(color_column, dropna=False)):
                ax.scatter(sub[x_column], sub[y_column], s=40, alpha=0.7, label=str(g), color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)])
            ax.legend(fontsize=8)
    else:
        ax.scatter(df[x_column], df[y_column], s=40, alpha=0.7, color=_COLOR_CYCLE[0])

    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or y_column)
    fig.tight_layout()
    return _save_figure(fig, {"type": "scatter", "x": x_column, "y": y_column, "title": title})


def create_box_plot(
    df: pd.DataFrame,
    y_column: str,
    x_column: str | None = None,
    title: str = "Box Plot",
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """Box plot for a numeric column, optionally split by groups."""
    _require_column(df, y_column)
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ToolError(f"Column '{y_column}' is not numeric; cannot plot a box plot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    if x_column:
        _require_column(df, x_column)
        groups = [sub[y_column].dropna().values for _, sub in df.groupby(x_column, dropna=False)]
        labels = [str(g) for g in df[x_column].dropna().unique()]
        ax.boxplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(x_label or x_column)
    else:
        ax.boxplot(df[y_column].dropna(), showmeans=True)
    ax.set_title(title, pad=12)
    ax.set_ylabel(y_label or y_column)
    fig.tight_layout()
    return _save_figure(fig, {"type": "box", "y": y_column, "x": x_column, "title": title})


def create_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Correlation Heatmap",
    method: str = "pearson",
) -> dict:
    """Correlation heatmap of numeric columns."""
    cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if columns:
        _require_columns(df, columns)
        cols = [c for c in columns if c in cols]
    if len(cols) < 2:
        raise ToolError("Need at least 2 numeric columns for a heatmap.")
    corr = df[cols].corr(method=method, numeric_only=True)

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(cols) + 3), 0.7 * len(cols) + 3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title(title, pad=12)
    fig.tight_layout()
    return _save_figure(fig, {"type": "heatmap", "columns": cols, "title": title})


def _save_figure(fig: plt.Figure, meta: dict) -> dict:
    """Save a figure to the chart directory and return its metadata."""
    filename = f"{uuid.uuid4().hex[:12]}.png"
    path = CHART_DIR / filename
    try:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        plt.close(fig)
        raise ToolError(f"Chart generation failed: {exc}") from exc
    return {
        "chart_type": meta.get("type"),
        "filename": filename,
        "path": str(path),
        "relative_path": f"charts/{filename}",
        "title": meta.get("title", ""),
        "x_column": meta.get("x"),
        "y_column": meta.get("y"),
        "columns": meta.get("columns", []),
    }


def _require_column(df: pd.DataFrame, column: str) -> None:
    if df is None or column not in df.columns:
        raise InvalidToolArgumentsError(f"Column '{column}' does not exist in the dataset.")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise InvalidToolArgumentsError(f"Unknown columns: {unknown}")


def _normalize_agg(agg: str) -> str:
    aliases = {"sum": "sum", "mean": "mean", "average": "mean", "avg": "mean",
               "count": "count", "median": "median", "max": "max", "min": "min"}
    norm = aliases.get(agg, agg)
    if norm not in aliases.values():
        raise InvalidToolArgumentsError(
            f"Invalid aggregation '{agg}'. Use one of: sum, mean, count, median, max, min."
        )
    return norm


def _limit_labels(labels: list, values, hue) -> None:
    """Trim extremely long category lists to keep charts readable."""
    MAX = 25
    if len(labels) > MAX:
        keep = labels[:MAX]
        if isinstance(values, np.ndarray) and values.ndim > 1:
            values = values[:, :MAX]
        elif isinstance(values, list):
            values = values[:MAX]
        labels.clear()
        labels.extend(keep)


def _try_to_datetime(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype == object:
        converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().mean() > 0.6:
            return converted
    return None