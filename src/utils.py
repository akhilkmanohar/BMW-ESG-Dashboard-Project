from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pdfplumber

FOOTNOTE_SUPERSCRIPTS = str.maketrans(
    {
        "¹": "",
        "²": "",
        "³": "",
        "⁴": "",
        "⁵": "",
        "⁶": "",
        "⁷": "",
        "⁸": "",
        "⁹": "",
        "⁰": "",
        "†": "",
        "‡": "",
    }
)
FOOTNOTE_REGEX = re.compile(r"\((?:[a-z]|[ivx]+)\)", flags=re.IGNORECASE)
MULTISPACE_REGEX = re.compile(r"\s+")
NON_BREAKING_SPACES = {"\u00A0", "\u202F", "\u2007"}


def strip_footnote_markers(text: str) -> str:
    """Remove common superscript and parenthetical footnote markers."""
    if text is None:
        return ""
    cleaned = str(text).translate(FOOTNOTE_SUPERSCRIPTS)
    cleaned = FOOTNOTE_REGEX.sub("", cleaned)
    return cleaned


def sanitize_text(text: str) -> str:
    """Collapse whitespace, drop footnote markers, and trim."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    working = strip_footnote_markers(str(text))
    for char in NON_BREAKING_SPACES:
        working = working.replace(char, " ")
    working = working.replace("−", "-")  # minus sign variants
    working = working.replace("\n", " ").replace("\r", " ")
    working = MULTISPACE_REGEX.sub(" ", working)
    return working.strip()


def is_probably_number(text: str) -> bool:
    """Heuristic number check after sanitisation."""
    candidate = sanitize_text(text)
    if not candidate:
        return False
    candidate = candidate.replace(",", ".") if candidate.count(",") == 1 and candidate.count(".") == 0 else candidate
    candidate = candidate.replace(" ", "")
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+", candidate))


def expand_page_spec(page_spec: str) -> List[int]:
    """Expand strings like '4-6;8' into explicit page numbers."""
    pages: List[int] = []
    for chunk in re.split(r"[;,]", page_spec):
        chunk = chunk.strip()
        if not chunk or chunk == "?":
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            if start_str.isdigit() and end_str.isdigit():
                start = int(start_str)
                end = int(end_str)
                step = 1 if end >= start else -1
                pages.extend(list(range(start, end + step, step)))
        elif chunk.isdigit():
            pages.append(int(chunk))
    return sorted(set(pages))


def detect_header_rows(df: pd.DataFrame) -> int:
    """Return the number of rows that belong to the header block."""
    header_rows = 1
    max_rows_to_inspect = min(5, len(df))
    for idx in range(max_rows_to_inspect):
        row = df.iloc[idx]
        values = [sanitize_text(val) for val in row.tolist()]
        if not any(values):
            continue
        numeric_count = sum(1 for val in values if is_probably_number(val))
        text_count = sum(1 for val in values if val and not is_probably_number(val))
        if numeric_count >= max(2, text_count):
            header_rows = max(1, idx)
            break
        header_rows = idx + 1
    return min(header_rows, len(df))


def collapse_multirow_header(df: pd.DataFrame) -> pd.DataFrame:
    """Combine leading rows into a single header and return the remaining body."""
    if df.empty:
        return df
    working = df.copy()
    working = working.applymap(sanitize_text)
    working = working.replace("", np.nan)
    working = working.dropna(axis=1, how="all")
    working = working.dropna(axis=0, how="all")
    working = working.fillna("")
    working.reset_index(drop=True, inplace=True)

    if working.empty:
        return working

    header_rows = detect_header_rows(working)
    header_block = working.iloc[:header_rows].to_numpy()
    data_block = working.iloc[header_rows:].copy()

    column_labels: List[str] = []
    for col_idx in range(working.shape[1]):
        pieces: List[str] = []
        for row in header_block:
            value = sanitize_text(row[col_idx])
            if value and value not in pieces:
                pieces.append(value)
        label = " ".join(pieces).strip()
        label = sanitize_text(label)
        if not label:
            label = f"column_{col_idx + 1}"
        column_labels.append(label)

    data_block.columns = column_labels
    data_block = data_block.replace("", np.nan).dropna(how="all").fillna("")
    data_block.reset_index(drop=True, inplace=True)
    return data_block


def standardize_numeric_string(value: str) -> str:
    """Normalise decimal/thousand separators."""
    text = sanitize_text(value)
    if not text:
        return text
    text = text.replace(" ", "")
    if text.count(",") > 0 and text.count(".") > 0:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") > 0:
        parts = text.split(",")
        if len(parts[-1]) in {1, 2}:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    return text


def extract_numeric_and_unit(cell_value: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Split a cell into numeric value, unit, and residual note."""
    text = sanitize_text(cell_value)
    if not text:
        return None, None, None
    if text.lower() in {"n/a", "na", "-", "–"}:
        return None, None, text

    match = re.search(r"([-+]?\d[\d\s.,]*)", text)
    if not match:
        return None, None, text

    number_str = match.group(1)
    numeric_candidate = standardize_numeric_string(number_str)
    try:
        numeric_value = float(numeric_candidate)
    except ValueError:
        numeric_value = None

    residual = text.replace(number_str, "").strip(" ;,")
    unit = residual if residual and residual not in {"market-based", "location-based"} else None
    note = None if residual == unit else residual if residual else None

    return numeric_value, unit, note


def pdfplumber_extract_table(pdf_path: Path, page_number: int) -> pd.DataFrame:
    """Fallback text-based extraction using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(pdf.pages):
            return pd.DataFrame()
        page = pdf.pages[page_index]
        text = page.extract_text(x_tolerance=1, y_tolerance=3)
        if not text:
            return pd.DataFrame()
        rows: List[List[str]] = []
        for raw_line in text.splitlines():
            line = sanitize_text(raw_line)
            if not line:
                continue
            segments = re.split(r"\s{2,}", line)
            segments = [sanitize_text(segment) for segment in segments if sanitize_text(segment)]
            if segments:
                rows.append(segments)
        if not rows:
            return pd.DataFrame()
        max_len = max(len(row) for row in rows)
        normalised_rows = [row + [""] * (max_len - len(row)) for row in rows]
        df = pd.DataFrame(normalised_rows)
        df = collapse_multirow_header(df)
        return df


def ensure_dataframe(df_like: Iterable[Sequence[str]]) -> pd.DataFrame:
    """Convert any iterable of row sequences to a DataFrame."""
    rows = [list(row) for row in df_like]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
