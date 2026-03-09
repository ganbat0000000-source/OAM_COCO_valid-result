from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import streamlit as st


def inject_app_theme(title: str) -> None:
    st.markdown(
        """
<style>
:root {
    --app-bg: #000000;
    --panel-bg: #101010;
    --text: #b3b3b3;
    --muted: #8e8e8e;
    --border: #2a2a2a;
    --df-header-bg: #2b6fff;
    --df-header-text: #ffffff;
}
@media (prefers-color-scheme: light) {
    :root {
        --df-header-bg: #ffd84d;
        --df-header-text: #000000;
    }
}
@media (prefers-color-scheme: dark) {
    :root {
        --df-header-bg: #2b6fff;
        --df-header-text: #ffffff;
    }
}
html, body, [data-testid="stAppViewContainer"], .stApp, .main {
    background: var(--app-bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background: #050505 !important;
    color: var(--text) !important;
}
.sticky-app-title {
    position: sticky;
    top: 0;
    z-index: 1200;
    background: var(--app-bg);
    color: #c9c9c9;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
    padding: 0.65rem 0 0.55rem;
    border-bottom: 1px solid var(--border);
}
#page-nav-anchor {
    height: 0;
}
#page-nav-anchor + div[data-testid="stHorizontalBlock"] {
    position: sticky;
    top: 3.25rem;
    z-index: 1190;
    background: var(--app-bg);
    border-bottom: 1px solid var(--border);
    padding-top: 0.45rem;
    padding-bottom: 0.55rem;
    margin-bottom: 0.65rem;
}
div.stButton > button,
div.stDownloadButton > button {
    margin: 0.2rem 0.35rem 0.5rem;
    padding: 0.58rem 0.8rem;
    border-radius: 10px;
    transition: transform 0.16s ease, box-shadow 0.2s ease, background-color 0.2s ease, color 0.2s ease;
}
div.stButton > button[kind="secondary"],
div.stDownloadButton > button[kind="secondary"] {
    background: #111111;
    border: 1px solid var(--border);
    color: var(--muted);
}
div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: #1f6bff;
    border: 1px solid #2a7bff;
    color: #f8fbff;
}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.35);
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.35);
    background: #cb1f34 !important;
    border-color: #df3c50 !important;
    color: #fff6f7 !important;
}
[data-testid="stDataFrame"] [role="gridcell"] {
    color: var(--text) !important;
    background: var(--panel-bg) !important;
    text-align: right !important;
    justify-content: flex-end !important;
}
[data-testid="stDataFrame"] [role="gridcell"] * {
    text-align: right !important;
    justify-content: flex-end !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    text-align: center !important;
    justify-content: center !important;
    padding-right: 0.55rem !important;
    background: var(--df-header-bg) !important;
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.01em !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
    text-shadow: none !important;
}
[data-testid="stDataFrame"] [role="columnheader"] * {
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
}
[data-testid="stDataFrame"] [role="columnheader"] span,
[data-testid="stDataFrame"] [role="columnheader"] div,
[data-testid="stDataFrame"] [role="columnheader"] p,
[data-testid="stDataFrame"] [role="columnheader"] [data-testid="stMarkdownContainer"] p {
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
}
[data-testid="stDataFrame"] [role="grid"] {
    background: var(--panel-bg) !important;
}
[data-testid="stDataFrame"] table td,
[data-testid="stDataFrame"] table th {
    color: var(--text) !important;
    background: var(--panel-bg) !important;
}
[data-testid="stDataFrame"] table td {
    text-align: right !important;
}
[data-testid="stDataFrame"] table th {
    text-align: center !important;
    padding-right: 0.55rem !important;
    background: var(--df-header-bg) !important;
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.01em !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
    text-shadow: none !important;
}
[data-testid="stDataFrame"] table th * {
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
}
[data-testid="stDataFrame"] thead th,
[data-testid="stDataFrame"] thead th div,
[data-testid="stDataFrame"] thead th span {
    background: var(--df-header-bg) !important;
    color: var(--df-header-text) !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.01em !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
    text-shadow: none !important;
}
[data-testid="stMetric"] {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.95rem 0.8rem;
    margin: 0.5rem 0.25rem 1rem;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    text-align: right !important;
    width: 100%;
}
[data-testid="stMetricLabel"] {
    font-size: 1.02rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
}
[data-testid="stMetricDelta"] {
    justify-content: flex-end !important;
}
[data-testid="stDataFrame"] {
    margin-top: 0.55rem;
    margin-bottom: 1.05rem;
}
.app-table-wrap {
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 0.55rem;
    margin-bottom: 1.05rem;
    background: var(--panel-bg);
}
.app-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel-bg);
}
.app-table-wrap thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--df-header-bg) !important;
    color: var(--df-header-text) !important;
    font-weight: 800 !important;
    font-size: 1.03rem !important;
    letter-spacing: 0.01em;
    text-align: left !important;
    border: 1px solid var(--border) !important;
    -webkit-text-fill-color: var(--df-header-text) !important;
    opacity: 1 !important;
}
.app-table-wrap tbody td {
    color: #f2f7ff !important;
    border: 1px solid var(--border) !important;
}
.app-table-wrap tbody tr:nth-child(even) td {
    background: #0d1322 !important;
}
.app-table-wrap tbody tr:nth-child(odd) td {
    background: #0b111f !important;
}
.app-note {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    background: var(--panel-bg);
    color: var(--text);
    margin-top: 0.65rem;
    text-align: center;
}
.section-title {
    color: #f3f7ff;
    font-size: 1.72rem;
    font-weight: 760;
    margin: 0.25rem 0 0.85rem;
}
.block-title {
    color: #eef4ff;
    font-size: 1.34rem;
    font-weight: 760;
    letter-spacing: 0.01em;
    text-align: center;
    margin: 0.4rem 0 0.42rem;
}
[data-testid="stCaptionContainer"] {
    background: rgba(31, 107, 255, 0.12);
    border: 1px solid #244a8f;
    border-radius: 10px;
    padding: 0.36rem 0.62rem;
    margin: 0.35rem 0 0.75rem;
}
[data-testid="stCaptionContainer"] p,
p.stCaption {
    color: #d6e6ff !important;
    opacity: 1 !important;
    font-size: 1rem !important;
    font-weight: 640 !important;
    line-height: 1.38 !important;
}
.intake-summary {
    background: #141414;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin: 0.35rem 0 0.95rem;
    text-align: center;
    font-size: 1.42rem;
    font-weight: 520;
    color: #d2d2d2;
    letter-spacing: 0.01em;
}
.intake-summary strong {
    color: #f0f0f0;
    font-weight: 760;
}
.intake-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin: 0.35rem 0 0.95rem;
}
.intake-item {
    background: #141414;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.95rem 0.75rem;
    text-align: center;
}
.intake-label {
    color: #dce7ff;
    font-size: 1.12rem;
    font-weight: 650;
    margin-bottom: 0.2rem;
    text-align: center;
}
.intake-value {
    color: #ffffff;
    font-size: 1.62rem;
    font-weight: 800;
    text-align: center;
}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="sticky-app-title">{title}</div>', unsafe_allow_html=True)


def compact_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        try:
            text = format(float(value), ".15g")
            if "e" in text.lower():
                dec = Decimal(text)
                text = format(dec, "f")
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return "0" if text in {"-0", ""} else text
        except (InvalidOperation, ValueError):
            return str(value)
    return str(value)


def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    cols: list[str] = []
    for i, raw in enumerate(out.columns, start=1):
        base = str(raw).strip() or f"Col{i}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        cols.append(base if count == 1 else f"{base}_{count}")
    out.columns = cols
    return out


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = _dedupe_column_names(df)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        else:
            text_col = out[col].fillna("").map(str).str.strip()
            numeric_candidate = pd.to_numeric(text_col.str.replace(",", ".", regex=False), errors="coerce")
            non_empty = text_col != ""
            if bool(non_empty.any()) and bool(numeric_candidate[non_empty].notna().all()):
                out[col] = numeric_candidate
            else:
                out[col] = text_col
    return out


def style_dataframe_for_display(df: pd.DataFrame):
    def _is_fixed_two_decimal_col(col_name: object) -> bool:
        token = str(col_name or "").lower()
        return (
            "0-1" in token
            or "(0,1)" in token
            or "index score" in token
            or "delta/fact (%)" in token
        )

    def _fixed_two(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return ""
        if pd.isna(value):
            return ""
        try:
            fv = float(value)
            if abs(fv - round(fv)) < 1e-12:
                return str(int(round(fv)))
            return f"{fv:.2f}"
        except (TypeError, ValueError):
            return str(value)

    out = format_dataframe_for_display(df)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    styler = out.style
    if numeric_cols:
        fmt_map = {c: (_fixed_two if _is_fixed_two_decimal_col(c) else compact_number) for c in numeric_cols}
        styler = styler.format(fmt_map)
    return (
        styler
        .set_properties(**{"text-align": "right"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("background-color", "var(--df-header-bg)"),
                        ("color", "var(--df-header-text)"),
                        ("font-weight", "700"),
                        ("font-size", "1.02rem"),
                        ("opacity", "1"),
                    ],
                },
                {"selector": "td", "props": [("text-align", "right")]},
            ]
        )
    )
