"""OAM (Object–Attribute Matrix) CSV ingestion utilities.

This project assumes a "wide" OAM CSV similar to the one you provided:

Row 0:  Direction ID, d1, d2, ... dM, Y
Row 1:  Type,         X,  X,  ... X,   (blank)
Row 2:  Attribute ID, A1, A2, ... AM,  (blank)
Row 3:  Attribute,    name1, ...
Row 4:  Attribute Unit, ...
Row 5+: <Object name>, x1, x2, ... xM, y

Where Direction ID uses:
- 0 = higher raw value is better (rank descending)
- 1 = lower raw value is better (rank ascending)

The COCO Y0 engine expects ranked integer inputs, so we rank BEFORE submitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import io
import re

import numpy as np
import pandas as pd


HEADER_LABELS = {
    "direction id": "direction",
    "type": "type",
    "attribute id": "attr_id",
    "attribute": "attr_name",
    "attribute unit": "attr_unit",
}


@dataclass
class OAM:
    directions: List[int]          # length = M
    attr_ids: List[str]            # length = M
    attr_names: List[str]          # length = M
    attr_units: List[str]          # length = M
    y_unit: str                    # optional unit metadata for Y
    objects: List[str]             # length = N
    x_raw: pd.DataFrame            # shape (N, M) numeric
    y: pd.Series                   # length = N numeric
    harmonized_attr_ids: List[str] = field(default_factory=list)
    harmonized_y: bool = False

def _make_unique_labels(labels: List[object], default_prefix: str = "Col") -> List[str]:
    """Return unique, non-empty labels while preserving order."""
    out: List[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(labels, start=1):
        base = str(raw).strip()
        if not base:
            base = f"{default_prefix}{i}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        out.append(base if count == 1 else f"{base}_{count}")
    return out


def _parse_numeric(cell: object) -> float:
    """Parse a number from messy CSV cells (e.g., '91.20%', ' 85 ', '0')."""
    if cell is None:
        return np.nan
    if isinstance(cell, float) and np.isnan(cell):
        return np.nan
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    s = s.replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]

    # European decimal comma
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # Remove thousand separators like 1,234
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    try:
        return float(s)
    except ValueError:
        return np.nan


def _norm_token(value: object) -> str:
    """Normalize labels for robust matching (e.g., 'Y0', 'y_0', ' y ')."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-_:/]+", "", text)
    return text


def _is_y_token(value: object) -> bool:
    """Accept Y markers like Y, Y0, y_0, coco:y0."""
    token = _norm_token(value)
    return bool(re.fullmatch(r"(?:coco)?y0*", token))


def _harmonize_column_scale(series: pd.Series, unit_hint: str = "") -> Tuple[pd.Series, bool]:
    """Harmonize mixed ratio/percent encodings within one numeric column.

    Example fixed case: [0.54, 0.41, 97, 100] -> [54, 41, 97, 100]
    """
    s = series.copy()
    vals = s.dropna().astype(float)
    if vals.empty:
        return s, False

    ge_zero = vals.ge(0).all()
    le_hundred = vals.le(100).all()
    has_ratio = (vals.ge(0) & vals.lt(1)).any()
    has_percent_like = vals.gt(1).any()

    # Harmonize only when a column mixes ratio and percent encodings.
    # If all values are in 0..1, keep them as-is even when unit text says "percent".
    if ge_zero and le_hundred and has_ratio and has_percent_like:
        mask = s.notna() & s.astype(float).between(0, 1, inclusive="both")
        if mask.any():
            s.loc[mask] = s.loc[mask].astype(float) * 100.0
            return s, True

    return s, False


def _harmonize_mixed_scales(
    x_raw: pd.DataFrame,
    attr_ids: List[str],
    attr_units: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply per-attribute harmonization for mixed ratio/percent values."""
    out = x_raw.copy()
    changed: List[str] = []
    for i, col in enumerate(out.columns):
        unit_hint = attr_units[i] if i < len(attr_units) else ""
        fixed, did_change = _harmonize_column_scale(out[col], unit_hint=unit_hint)
        out[col] = fixed
        if did_change:
            changed.append(attr_ids[i] if i < len(attr_ids) else str(col))
    return out, changed


def _decode_csv_bytes(file_bytes: bytes) -> str:
    """Decode uploaded CSV bytes robustly and normalize all newline variants."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = file_bytes.decode("utf-8", errors="replace")

    # Canonicalize line endings to avoid platform-specific parsing differences.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_oam_csv(file_bytes: bytes) -> OAM:
    """Read an OAM CSV from uploaded bytes."""
    csv_text = _decode_csv_bytes(file_bytes)
    df = pd.read_csv(io.StringIO(csv_text), header=None, dtype=str)

    # Find header rows
    row_index = {}
    for i, v in enumerate(df[0].fillna("")):
        key = str(v).strip().lower()
        if key in HEADER_LABELS:
            row_index[HEADER_LABELS[key]] = i

    missing = {"direction", "attr_id", "attr_name"} - set(row_index)
    if missing:
        raise ValueError(f"Missing required header row(s): {sorted(missing)}. Found: {row_index}")

    rid_dir = row_index["direction"]
    rid_attr_id = row_index["attr_id"]
    rid_attr_name = row_index["attr_name"]
    rid_attr_unit = row_index.get("attr_unit")

    # Determine the Y column robustly.
    # Some CSVs contain trailing separators, which create an extra blank last column.
    ncols = df.shape[1]
    dir_row = df.iloc[rid_dir].fillna("").astype(str)
    y_candidates = [idx for idx in range(1, ncols) if _is_y_token(dir_row.iloc[idx])]
    if y_candidates:
        y_col = y_candidates[-1]
    else:
        # Fallback to the right-most non-empty column among key header rows.
        y_col = ncols - 1
        for idx in range(ncols - 1, 0, -1):
            cells = [
                str(df.iloc[rid_dir, idx]).strip(),
                str(df.iloc[rid_attr_id, idx]).strip(),
                str(df.iloc[rid_attr_name, idx]).strip(),
            ]
            if any(c and c.lower() not in {"nan", "none"} for c in cells):
                y_col = idx
                break

    # Attribute columns are 1..(y_col-1)
    attr_cols = list(range(1, y_col))

    # Directions and attribute metadata
    dir_vals = df.iloc[rid_dir, attr_cols].tolist()
    directions: List[int] = []
    for d in dir_vals:
        try:
            directions.append(int(str(d).strip()))
        except Exception:
            directions.append(0)

    attr_ids = [str(x).strip() for x in df.iloc[rid_attr_id, attr_cols].tolist()]
    attr_ids = _make_unique_labels(attr_ids, default_prefix="A")
    attr_names = [str(x).strip() for x in df.iloc[rid_attr_name, attr_cols].tolist()]
    if rid_attr_unit is not None:
        attr_units = [str(x).strip() for x in df.iloc[rid_attr_unit, attr_cols].tolist()]
        y_unit = str(df.iloc[rid_attr_unit, y_col]).strip()
    else:
        attr_units = [""] * len(attr_cols)
        y_unit = ""

    # Object rows: everything not a recognized header label
    header_rows = set(row_index.values())
    obj_df = df[~df.index.isin(header_rows)].copy()
    obj_df = obj_df[obj_df[0].notna() & (obj_df[0].astype(str).str.strip() != "")]

    raw_objects = [str(x).strip() for x in obj_df[0].tolist()]
    objects = _make_unique_labels(raw_objects, default_prefix="Object")

    x_raw = obj_df.loc[:, attr_cols].apply(lambda col: col.map(_parse_numeric))
    x_raw.columns = attr_ids

    y = obj_df.iloc[:, y_col].apply(_parse_numeric)
    y.name = "Y"

    # Keep numeric values exactly as parsed; do not auto-convert decimal/percent scales.
    harmonized_attr_ids: List[str] = []

    # Basic sanity checks
    if x_raw.shape[1] != len(directions):
        raise ValueError("Attribute column count mismatch while parsing.")

    return OAM(
        directions=directions,
        attr_ids=attr_ids,
        attr_names=attr_names,
        attr_units=attr_units,
        y_unit=y_unit,
        objects=objects,
        x_raw=x_raw.reset_index(drop=True),
        y=y.reset_index(drop=True),
        harmonized_attr_ids=harmonized_attr_ids,
        harmonized_y=False,
    )
