"""Lightweight data access helpers used by the merged engine and legacy SCR9."""
from __future__ import annotations
import pandas as pd
from quant_excel_loader import load_results_excel, _normalize_columns, _parse_number, SLOT_NAMES


def load_results_dataframe(path: str = "number prediction learn.xlsx") -> pd.DataFrame:
    """Return a wide dataframe similar to the Excel layout.

    Columns: DATE, FRBD, GZBD, GALI, DSWR. The function preserves the raw
    cell values except that "XX" is kept for downstream checks. Dates are
    parsed to datetime for consistent handling.
    """
    df_raw = pd.read_excel(path, engine="openpyxl")
    df_raw = _normalize_columns(df_raw)
    if "DATE" not in df_raw.columns and "date" in df_raw.columns:
        df_raw = df_raw.rename(columns={"date": "DATE"})
    if "DATE" not in df_raw.columns:
        raise ValueError("DATE column missing in Excel file")

    df_raw["DATE"] = pd.to_datetime(df_raw["DATE"], errors="coerce")
    for slot in SLOT_NAMES:
        if slot not in df_raw.columns:
            df_raw[slot] = None
    return df_raw[["DATE", *SLOT_NAMES]]


def load_results_long(path: str = "number prediction learn.xlsx") -> pd.DataFrame:
    """Compat wrapper around load_results_excel for callers that expect long form."""
    return load_results_excel(path)
