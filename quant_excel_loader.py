import pandas as pd
from typing import Optional

SLOT_NAMES = ["FRBD", "GZBD", "GALI", "DSWR"]
SLOT_MAP = {name: idx + 1 for idx, name in enumerate(SLOT_NAMES)}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.upper() == "XX":
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits) % 100
    except Exception:
        return None


def load_results_excel(path: str) -> pd.DataFrame:
    """Load the Excel file and return a long-form dataframe.

    The loader mirrors the expectations from the legacy scripts while adding
    strict handling for "XX" (closed days) and malformed values. Returned
    frame columns: date (datetime64), slot (1-4), number (int 0-99).
    """
    df_raw = pd.read_excel(path, engine="openpyxl")
    df_raw = _normalize_columns(df_raw)

    # Identify date column
    date_col = "date" if "date" in df_raw.columns else "DATE"
    if date_col not in df_raw.columns:
        raise ValueError("DATE column missing in Excel file")

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
    long_rows = []
    for _, row in df_raw.iterrows():
        date_val = row.get(date_col)
        if pd.isna(date_val):
            continue
        for slot_name in SLOT_NAMES:
            if slot_name not in df_raw.columns:
                continue
            num = _parse_number(row.get(slot_name))
            if num is None:
                continue
            long_rows.append(
                {
                    "date": date_val,
                    "slot": SLOT_MAP[slot_name],
                    "number": num,
                }
            )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df

    long_df = long_df.dropna(subset=["date", "slot", "number"])
    long_df = long_df.astype({"slot": int, "number": int})
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["date", "slot"]).reset_index(drop=True)
    return long_df
