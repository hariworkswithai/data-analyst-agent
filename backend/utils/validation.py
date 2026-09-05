"""Validation for uploaded files. Treats uploads as untrusted input."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.config import config
from backend.utils.errors import DatasetValidationError


@dataclass
class LoadedDataset:
    """A validated, loaded dataset with safe metadata."""

    df: pd.DataFrame
    filename: str
    path: str
    rows: int
    columns: int


def validate_and_load(path: str, filename: str) -> LoadedDataset:
    """Validate a file and load it into a DataFrame.

    Enforces extension, size, and dimension limits.
    Raises DatasetValidationError for anything unsafe.
    """
    ext = Path(filename).suffix.lower() if isinstance(filename, str) else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

    if path.upper().endswith(".CSV"):
        try:
            _check_unique_headers(path)
            df = _load_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse error safely
            raise DatasetValidationError(
                f"Could not parse file as CSV: {exc}"
            ) from exc
    else:
        raise DatasetValidationError("Only CSV files are currently supported.")

    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    rows, cols = df.shape
    if rows > config.MAX_ROWS:
        raise DatasetValidationError(
            f"Dataset has {rows} rows (max {config.MAX_ROWS})."
        )
    if cols > config.MAX_COLUMNS:
        raise DatasetValidationError(
            f"Dataset has {cols} columns (max {config.MAX_COLUMNS})."
        )
    if df.columns.duplicated().any():
        raise DatasetValidationError(
            f"Duplicate column names found: {list(df.columns[df.columns.duplicated()])}."
        )

    return LoadedDataset(df=df, filename=filename, path=str(path), rows=rows, columns=cols)


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV with lenient-but-safe parsing."""
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )
    return df


def _check_unique_headers(path: str) -> None:
    """Reject files whose raw header line has duplicate column names.

    pandas mangles duplicate headers silently; we check the raw first line
    so dangerous/incorrect schemas are rejected before parsing.
    """
    import csv as _csv

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.readline()
    except OSError:
        return
    if not raw.strip():
        return
    try:
        header = next(_csv.reader([raw]))
    except _csv.Error:
        return
    seen = set()
    for col in header:
        c = col.strip()
        if c in seen:
            raise DatasetValidationError(
                f"Duplicate column names found: duplicates in header {header}."
            )
        seen.add(c)