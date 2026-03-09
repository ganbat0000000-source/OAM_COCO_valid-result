import hashlib
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

from src.coco_client import run_coco_y0
from src.coco_parse import (
    extract_coco_overview,
    extract_coco_section_tables,
    extract_coco_totals,
    extract_coco_y0_table,
)
from src.oam_io import read_oam_csv
from src.ranking import rank_oam_columns
from src.ui_display import compact_number, format_dataframe_for_display, inject_app_theme, style_dataframe_for_display


st.set_page_config(page_title="OAM -> COCO Y0 Automation", layout="wide")
inject_app_theme("OAM -> COCO Y0 Automation")


def _direction_label(direction: int) -> str:
    return "Lower is better" if int(direction) == 1 else "Higher is better"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    code = hex_color.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _blend_color(start_hex: str, end_hex: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    s = _hex_to_rgb(start_hex)
    e = _hex_to_rgb(end_hex)
    out = tuple(int(round(s[i] + (e[i] - s[i]) * t)) for i in range(3))
    return _rgb_to_hex(out)


def _corr_bg_color(value: float, diagonal: bool = False) -> str:
    if diagonal:
        return "#63be7b"
    if value >= 0:
        # Positive values: yellow -> green (Excel-like).
        return _blend_color("#ffe699", "#63be7b", value)
    # Negative values: light red -> red.
    return _blend_color("#f4cccc", "#f8696b", abs(value))


def _attribute_display_labels(oam) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for i, (attr_id, attr_name) in enumerate(zip(oam.attr_ids, oam.attr_names)):
        name_text = str(attr_name).strip()
        id_text = str(attr_id).strip()
        base = name_text or id_text
        count = seen.get(base, 0) + 1
        seen[base] = count
        labels.append(base if count == 1 else f"{base}_{count}")
    return labels


def _y_display_label(oam) -> str:
    return "Y"


def _build_attribute_sheet(oam) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Attribute ID": oam.attr_ids,
            "Attribute Name": oam.attr_names,
            "Attribute Unit": list(getattr(oam, "attr_units", [""] * len(oam.attr_ids))),
            "Direction ID": oam.directions,
            "Direction Rule": [_direction_label(v) for v in oam.directions],
        }
    )


def _build_object_sheet(oam) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Object": oam.objects,
            _y_display_label(oam): oam.y.values,
        }
    )


def _build_input_sheet(oam) -> pd.DataFrame:
    df = oam.x_raw.copy()
    df.columns = _attribute_display_labels(oam)
    df.insert(0, "Object", oam.objects)
    df[_y_display_label(oam)] = oam.y.values
    return df


def _build_ranked_matrix(oam) -> pd.DataFrame:
    return rank_oam_columns(oam.x_raw, oam.directions)


def _build_inverse_ranked_matrix(oam) -> pd.DataFrame:
    ranked = _build_ranked_matrix(oam)
    n_objects = len(oam.objects)
    return (n_objects + 1) - ranked


def _normalized_y(y: pd.Series) -> pd.Series:
    return y.copy()


def _build_ranked_sheet(oam) -> pd.DataFrame:
    ranked = _build_ranked_matrix(oam).copy()
    ranked.columns = _attribute_display_labels(oam)
    ranked.insert(0, "Object", oam.objects)
    ranked[_y_display_label(oam)] = _normalized_y(oam.y).values
    return ranked


def _build_inverse_ranked_sheet(oam) -> pd.DataFrame:
    ranked = _build_inverse_ranked_matrix(oam).copy()
    ranked.columns = _attribute_display_labels(oam)
    ranked.insert(0, "Object", oam.objects)
    ranked[_y_display_label(oam)] = _normalized_y(oam.y).values
    return ranked


def _build_correlation_matrix(oam) -> pd.DataFrame:
    data = oam.x_raw.copy()
    data.columns = _attribute_display_labels(oam)
    return data.corr(method="pearson")


def _style_correlation_matrix(corr_df: pd.DataFrame):
    out = corr_df.copy()
    n = len(out.index)
    for i in range(n):
        for j in range(n):
            if i < j:
                out.iat[i, j] = pd.NA

    def _fmt_corr(v: object) -> str:
        if pd.isna(v):
            return ""
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if abs(fv - round(fv)) < 1e-12:
            return str(int(round(fv)))
        return f"{fv:.2f}"

    def _style_cells(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        size = len(df.index)
        for i in range(size):
            for j in range(size):
                if i < j:
                    styles.iat[i, j] = "background-color: #d9d9d9; color: #d9d9d9;"
                    continue
                cell_value = df.iat[i, j]
                if pd.isna(cell_value):
                    styles.iat[i, j] = "background-color: #d9d9d9;"
                    continue
                bg = _corr_bg_color(float(cell_value), diagonal=(i == j))
                styles.iat[i, j] = f"background-color: {bg}; color: #000000;"
        return styles

    return (
        out.style
        .format(_fmt_corr)
        .apply(_style_cells, axis=None)
        .set_properties(**{"text-align": "right"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "right")]},
            ]
        )
    )


def _to_float_loose(value) -> float | None:
    txt = str(value).strip()
    if not txt:
        return None
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", txt)
    if not nums:
        return None
    token = nums[-1].replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _numeric_values_equal(left: object, right: object, tol: float = 1e-12) -> bool:
    left_num = _to_float_loose(left)
    right_num = _to_float_loose(right)
    if left_num is not None and right_num is not None:
        return abs(float(left_num) - float(right_num)) <= float(tol)
    return str(left).strip() == str(right).strip()


def _build_direction_qa_sheet(oam, corr_threshold: float = 0.35) -> pd.DataFrame:
    attr_ids = list(oam.attr_ids)
    attr_names = list(oam.attr_names)
    attr_labels = _attribute_display_labels(oam)
    user_dirs = [1 if int(v) == 1 else 0 for v in oam.directions]

    data = oam.x_raw.copy()
    data.columns = attr_ids
    corr = data.corr(method="pearson")
    y_corr = data.apply(lambda c: c.corr(oam.y))

    rows = []
    for i, attr_id in enumerate(attr_ids):
        user_dir = user_dirs[i]
        higher_score = 0.0
        lower_score = 0.0
        peer_count = 0

        for j, peer_id in enumerate(attr_ids):
            if i == j:
                continue
            corr_ij = corr.at[attr_id, peer_id] if attr_id in corr.index and peer_id in corr.columns else pd.NA
            if pd.isna(corr_ij):
                continue

            weight = abs(float(corr_ij))
            if weight < float(corr_threshold):
                continue

            peer_count += 1
            peer_dir = user_dirs[j]
            implied_dir = peer_dir if float(corr_ij) >= 0 else 1 - peer_dir
            if implied_dir == 1:
                lower_score += weight
            else:
                higher_score += weight

        corr_with_y = y_corr.get(attr_id, pd.NA)
        if not pd.isna(corr_with_y):
            y_weight = abs(float(corr_with_y)) * 1.15
            if float(corr_with_y) >= 0:
                higher_score += y_weight
            else:
                lower_score += y_weight

        total_score = higher_score + lower_score
        score_gap = abs(higher_score - lower_score)
        confidence = (score_gap / total_score) if total_score > 0 else 0.0
        recommended_dir = 0 if higher_score >= lower_score else 1
        has_signal = total_score > 0
        mismatch = has_signal and (recommended_dir != user_dir)

        if not has_signal:
            warning = "No signal"
            reason = "Not enough correlation evidence to infer a recommendation."
        elif mismatch and confidence >= 0.35:
            warning = "High"
            reason = "User rule conflicts with a strong correlation-based recommendation."
        elif mismatch:
            warning = "Medium"
            reason = "User rule conflicts with the correlation-based recommendation."
        elif confidence < 0.12:
            warning = "Low confidence"
            reason = "User rule aligns, but evidence is weak."
        else:
            warning = "Aligned"
            reason = "User rule is aligned with the correlation-based recommendation."

        rows.append(
            {
                "Attribute ID": attr_id,
                "Attribute Name": attr_names[i],
                "Attribute": attr_labels[i],
                "User Direction ID": user_dir,
                "User Direction Rule": _direction_label(user_dir),
                "Recommended Direction ID": recommended_dir,
                "Recommended Direction Rule": _direction_label(recommended_dir),
                "Early Warning": warning,
                "Confidence": confidence,
                "Peers Used (|corr|>=0.35)": peer_count,
                "Corr with Y": corr_with_y,
                "Evidence (Higher score)": higher_score,
                "Evidence (Lower score)": lower_score,
                "QA Reason": reason,
            }
        )

    return pd.DataFrame(rows)


def _norm_label_token(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _translate_header_label(label: object) -> str:
    raw = str(label or "").strip()
    token = _norm_label_token(raw)
    if not token:
        return raw
    if "objekt" in token or token == "object":
        return "Object"
    if "becsl" in token or "estim" in token:
        return "Estimation"
    if "teny" in token or "fact" in token:
        suffix_match = re.search(r"([+-]\d+)$", token)
        suffix = suffix_match.group(1) if suffix_match else ""
        return f"Fact{suffix}"
    if "delta" in token or "elteres" in token:
        return "Delta"
    if "rangsor" in token or token == "rank":
        return "Rank"
    if "lepcs" in token or "stairs" in token:
        nums = re.findall(r"\d+", token)
        return f"Stairs({nums[0]})" if nums else "Stairs"
    return raw


def _translate_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    seen: dict[str, int] = {}
    cols: list[str] = []
    for col in out.columns:
        base = _translate_header_label(col).strip() or str(col)
        count = seen.get(base, 0) + 1
        seen[base] = count
        cols.append(base if count == 1 else f"{base}_{count}")
    out.columns = cols
    return out


def _section_display_label(label: object) -> str:
    token = _norm_label_token(label)
    if "rangsor" in token or "rank" in token:
        return "Ranking"
    if "lepcs" in token or "stairs" in token:
        nums = re.findall(r"\d+", token)
        return f"Stairs({nums[0]})" if nums else "Stairs"
    return str(label)


def _section_sort_key(label: object) -> tuple[int, str]:
    display = _section_display_label(label)
    if display == "Ranking":
        return (0, display)
    if display == "Stairs(1)":
        return (1, display)
    if display == "Stairs(2)":
        return (2, display)
    return (3, display)


def _to_coco_matrix_tsv(ranked_x: pd.DataFrame, y: pd.Series, col_sep: str = "\t", row_sep: str = "\r\n") -> str:
    df = ranked_x.copy()
    df["Y"] = y.values
    lines = []
    for row in df.to_numpy():
        vals = []
        for value in row:
            fv = float(value)
            vals.append(str(int(fv)) if fv.is_integer() else str(value))
        lines.append(col_sep.join(vals))
    # Canonical CRLF with trailing terminator avoids legacy parser edge-cases on last row.
    return row_sep.join(lines) + row_sep


def _run_coco_with_ranked_matrix(oam, ranked_matrix: pd.DataFrame, steps: int, identifier: str) -> tuple[pd.DataFrame, str]:
    y_for_coco = _normalized_y(oam.y)
    attr_with_y = _attribute_display_labels(oam) + [_y_display_label(oam)]

    object_block = "\n".join(oam.objects)
    attribute_block = "\t".join(attr_with_y)

    auto_steps = len(oam.objects)
    steps_val = auto_steps if steps == 0 else int(steps)

    max_rank = int(ranked_matrix.max().max()) if not ranked_matrix.empty else 1
    if steps_val < max_rank:
        steps_val = max_rank

    last_html = ""

    matrix_tsv = _to_coco_matrix_tsv(ranked_matrix, y_for_coco, col_sep="\t", row_sep="\r\n")

    html, _ = run_coco_y0(
        matrix_tsv=matrix_tsv,
        object_names=object_block,
        attribute_names=attribute_block,
        steps=steps_val,
        identifier=identifier,
    )
    last_html = html

    output_df = extract_coco_y0_table(html, oam.objects)
    if not output_df.empty:
        return output_df, html

    return pd.DataFrame(), last_html


def _run_coco_estimation(oam, steps: int, identifier: str) -> tuple[pd.DataFrame, str]:
    ranked_matrix = _build_ranked_matrix(oam)
    return _run_coco_with_ranked_matrix(oam, ranked_matrix, steps, identifier)


def _run_coco_inverse_estimation(oam, steps: int, identifier: str) -> tuple[pd.DataFrame, str]:
    inverse_ranked_matrix = _build_inverse_ranked_matrix(oam)
    inverse_identifier = f"{identifier}_inverse"
    return _run_coco_with_ranked_matrix(oam, inverse_ranked_matrix, steps, inverse_identifier)


def _extract_coco_metrics(oam, coco_y0_df: pd.DataFrame) -> pd.DataFrame:
    empty_cols = ["Object", "Estimation", "Fact", "Delta", "Delta/Fact"]
    if coco_y0_df is None or coco_y0_df.empty:
        return pd.DataFrame(columns=empty_cols)

    obj_col = str(coco_y0_df.columns[0])

    def _pick_col_by_keywords(df: pd.DataFrame, keywords: list[str]) -> str | None:
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in keywords):
                return str(c)
        return None

    est_col = None
    for c in coco_y0_df.columns:
        cl = str(c).lower()
        if "becsl" in cl or "estim" in cl:
            est_col = str(c)
            break
    fact_col = _pick_col_by_keywords(coco_y0_df, ["tény", "tÃ©ny", "teny", "fact"])
    delta_col = _pick_col_by_keywords(coco_y0_df, ["delta"])
    delta_fact_col = _pick_col_by_keywords(coco_y0_df, ["delta/fact", "delta / fact", "/fact"])

    if est_col is None:
        candidates = []
        for c in coco_y0_df.columns[1:]:
            vals = [_to_float_loose(v) for v in coco_y0_df[c].tolist()]
            score = sum(v is not None for v in vals)
            candidates.append((score, str(c)))
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            est_col = candidates[0][1]

    if est_col is None:
        return pd.DataFrame(columns=empty_cols)

    obj_set = set(oam.objects)
    rows = []
    for _, r in coco_y0_df.iterrows():
        obj = str(r.get(obj_col, "")).strip()
        est = _to_float_loose(r.get(est_col, ""))
        fact = _to_float_loose(r.get(fact_col, "")) if fact_col else None
        delta = _to_float_loose(r.get(delta_col, "")) if delta_col else None
        delta_fact = _to_float_loose(r.get(delta_fact_col, "")) if delta_fact_col else None
        if not obj or est is None or obj not in obj_set:
            continue
        if delta is None and fact is not None:
            delta = float(fact) - float(est)
        if delta_fact is None and fact not in (None, 0) and delta is not None:
            delta_fact = float(delta) / float(fact)
        rows.append((obj, float(est), fact, delta, delta_fact))

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    return pd.DataFrame(rows, columns=empty_cols)


def _build_result_ranking_legacy(oam, coco_y0_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = _extract_coco_metrics(oam, coco_y0_df)
    if metrics_df.empty:
        return pd.DataFrame(columns=["Object", "Estimation", "Fact", "Delta", "Delta/Fact (%)", "Rank"])
    fact_col = _pick_col_by_keywords(coco_y0_df, ["tény", "teny", "fact"])
    delta_col = _pick_col_by_keywords(coco_y0_df, ["delta"])

    if est_col is None:
        candidates = []
        for c in coco_y0_df.columns[1:]:
            vals = [_to_float_loose(v) for v in coco_y0_df[c].tolist()]
            score = sum(v is not None for v in vals)
            candidates.append((score, str(c)))
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            est_col = candidates[0][1]

    if est_col is None:
        return pd.DataFrame(columns=["Object", "Estimation", "Fact", "Delta", "Delta/Fact (%)", "Rank"])

    obj_set = set(oam.objects)
    rows = []
    for _, r in coco_y0_df.iterrows():
        obj = str(r.get(obj_col, "")).strip()
        est = _to_float_loose(r.get(est_col, ""))
        fact = _to_float_loose(r.get(fact_col, "")) if fact_col else None
        delta = _to_float_loose(r.get(delta_col, "")) if delta_col else None
        if not obj or est is None:
            continue
        if obj not in obj_set:
            continue
        if delta is None and fact is not None:
            delta = float(fact) - float(est)
        delta_pct = None
        if fact not in (None, 0):
            delta_pct = (float(delta) / float(fact)) * 100.0 if delta is not None else None
        rows.append((obj, float(est), fact, delta, delta_pct))

    if not rows:
        return pd.DataFrame(columns=["Object", "Estimation", "Fact", "Delta", "Delta/Fact (%)", "Rank"])

    out = pd.DataFrame(
        rows,
        columns=["Object", "Estimation", "Fact", "Delta", "Delta/Fact (%)"],
    )
    out = out.sort_values("Estimation", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out


def _build_result_ranking(oam, coco_y0_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = _extract_coco_metrics(oam, coco_y0_df)
    if metrics_df.empty:
        return pd.DataFrame(columns=["Object", "Estimation", "Fact", "Delta", "Delta/Fact (%)", "Rank"])

    out = metrics_df.copy()
    out["Delta/Fact (%)"] = out["Delta/Fact"].apply(lambda v: None if pd.isna(v) else float(v) * 100.0)
    out = out.drop(columns=["Delta/Fact"])
    out = out.sort_values("Estimation", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out


def _build_validation_table(oam, baseline_coco_df: pd.DataFrame, inverse_coco_df: pd.DataFrame) -> pd.DataFrame:
    base_df = _extract_coco_metrics(oam, baseline_coco_df).rename(
        columns={
            "Estimation": "Baseline Estimation",
            "Fact": "Baseline Fact",
            "Delta": "Baseline Delta",
            "Delta/Fact": "Baseline Delta/Fact",
        }
    )
    inverse_df = _extract_coco_metrics(oam, inverse_coco_df).rename(
        columns={
            "Estimation": "Inverse Estimation",
            "Fact": "Inverse Fact",
            "Delta": "Inverse Delta",
            "Delta/Fact": "Inverse Delta/Fact",
        }
    )

    merged = pd.DataFrame({"Object": oam.objects})
    merged = merged.merge(base_df, on="Object", how="left")
    merged = merged.merge(inverse_df, on="Object", how="left")

    def _valid_flag(row: pd.Series) -> int:
        base_ratio = row.get("Baseline Delta/Fact")
        inverse_ratio = row.get("Inverse Delta/Fact")
        if pd.isna(base_ratio) or pd.isna(inverse_ratio):
            return 0
        return 1 if float(base_ratio) * float(inverse_ratio) <= 0 else 0

    merged["Validation"] = merged.apply(_valid_flag, axis=1)
    merged["Status"] = merged["Validation"].map({1: "Valid", 0: "Invalid"})
    merged["Baseline Delta/Fact (%)"] = merged["Baseline Delta/Fact"].apply(
        lambda v: None if pd.isna(v) else float(v) * 100.0
    )
    merged["Inverse Delta/Fact (%)"] = merged["Inverse Delta/Fact"].apply(
        lambda v: None if pd.isna(v) else float(v) * 100.0
    )
    return merged


def _build_validated_result(oam, baseline_coco_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    result_df = _build_result_ranking(oam, baseline_coco_df)
    if result_df.empty or validation_df is None or validation_df.empty:
        return result_df

    validation_cols = validation_df[
        ["Object", "Validation", "Status", "Baseline Delta/Fact (%)", "Inverse Delta/Fact (%)"]
    ].copy()
    return result_df.merge(validation_cols, on="Object", how="left")


def _result_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Result")
    return out.getvalue()


def _to_standard_csv_bytes(df: pd.DataFrame) -> bytes:
    csv_text = df.to_csv(index=False, float_format="%.15g", lineterminator="\r\n")
    return csv_text.encode("utf-8-sig")


def _render_input_table(df: pd.DataFrame, height: int) -> None:
    styled = style_dataframe_for_display(df)
    try:
        styled = styled.hide(axis="index")
    except Exception:
        pass
    st.markdown(
        f'<div class="app-table-wrap" style="max-height:{int(height)}px;">{styled.to_html()}</div>',
        unsafe_allow_html=True,
    )


def _csv_template_text() -> str:
    # Minimal valid OAM shape users can copy/paste into a CSV editor.
    return (
        "Direction ID,0,1,Y\r\n"
        "Type,X,X,\r\n"
        "Attribute ID,A1,A2,\r\n"
        "Attribute,Attribute 1,Attribute 2,\r\n"
        "Attribute Unit,decimal,decimal,\r\n"
        "Object 1,0.54,0.41,1000\r\n"
        "Object 2,0.53,0.07,900\r\n"
    )


def _example_csv_bytes() -> bytes:
    example_path = Path(__file__).resolve().parent / "assets" / "Example.csv"
    if example_path.exists():
        return example_path.read_bytes()
    return _csv_template_text().encode("utf-8-sig")


def _build_result_chart(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig_w = max(8, min(24, 0.5 * max(len(df), 1)))
    has_delta = "Delta" in df.columns and df["Delta"].notna().any()
    fig_h = 8 if has_delta else 5
    fig, axes = plt.subplots(2 if has_delta else 1, 1, figsize=(fig_w, fig_h), sharex=True)
    if has_delta:
        ax = axes[0]
        ax_delta = axes[1]
    else:
        ax = axes
        ax_delta = None

    ax.bar(df["Object"], df["Estimation"], color="#2f7cff", label="Estimation")
    if "Fact" in df.columns and df["Fact"].notna().any():
        ax.plot(df["Object"], df["Fact"], color="#8f8f8f", marker="o", linewidth=1.4, label="Official benchmark (Fact)")

    est_min = float(df["Estimation"].min())
    est_max = float(df["Estimation"].max())
    if est_min == est_max:
        span = max(abs(est_min) * 0.1, 1.0)
        ax.set_ylim(est_min - span, est_max + span)
    else:
        ax.set_ylim(est_min * 0.9, est_max * 1.1)

    ax.set_ylabel("COCO score (points)")
    ax.set_title("Object Estimation Ranking (Fine-tuned Y Axis)", fontsize=16, fontweight="bold", pad=12)
    if "Fact" in df.columns and df["Fact"].notna().any():
        ax.legend(loc="best")

    ax.set_facecolor("#101010")
    if ax_delta is not None:
        colors = ["#2f7cff" if float(v) >= 0 else "#cb1f34" for v in df["Delta"].fillna(0)]
        ax_delta.set_facecolor("#101010")
        ax_delta.bar(df["Object"], df["Delta"].fillna(0), color=colors)
        ax_delta.axhline(0, color="#6f6f6f", linewidth=0.8)
        ax_delta.set_ylabel("Delta (points)")
        ax_delta.set_title("Delta Sensitivity", fontsize=14, fontweight="bold", pad=10)
        ax_delta.tick_params(axis="x", rotation=75, labelsize=8)
    else:
        ax.tick_params(axis="x", rotation=75, labelsize=8)

    ax.set_xlabel("Object")
    for axis in ([ax_delta] if ax_delta is not None else []) + [ax]:
        axis.tick_params(colors="#b3b3b3")
        axis.xaxis.label.set_color("#b3b3b3")
        axis.yaxis.label.set_color("#b3b3b3")
        axis.title.set_color("#c9c9c9")
        for spine in axis.spines.values():
            spine.set_color("#2a2a2a")
    fig.patch.set_facecolor("#000000")
    fig.tight_layout()
    return fig


def _result_chart_png(df: pd.DataFrame) -> bytes | None:
    fig = _build_result_chart(df)
    if fig is None:
        return None

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _build_unit_qa_summary(qa_df: pd.DataFrame) -> pd.DataFrame:
    if qa_df is None or qa_df.empty or "Early Warning" not in qa_df.columns:
        return pd.DataFrame(columns=["Status", "Count"])
    order = ["High", "Medium", "Low confidence", "Aligned", "No signal"]
    counts = qa_df["Early Warning"].astype(str).value_counts()
    rows = [{"Status": name, "Count": int(counts.get(name, 0))} for name in order]
    return pd.DataFrame(rows)


def _get_stairs2_table(section_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for key, table in section_tables.items():
        key_l = str(key).lower()
        if "(2" in key_l or "stairs(2" in key_l:
            return table
    return pd.DataFrame()


def _attributes_with_stair_range(oam, stairs2_df: pd.DataFrame) -> pd.DataFrame:
    if stairs2_df is None or stairs2_df.empty:
        return pd.DataFrame(columns=["Attribute ID", "Attribute Name", "Pattern"])

    n = len(oam.objects)
    required = set(range(0, n))
    out_rows = []
    display_labels = _attribute_display_labels(oam)
    for attr_id, attr_name, display_label in zip(oam.attr_ids, oam.attr_names, display_labels):
        source_col = None
        if display_label in stairs2_df.columns:
            source_col = display_label
        elif attr_id in stairs2_df.columns:
            source_col = attr_id
        if source_col is None:
            continue
        vals = [_to_float_loose(v) for v in stairs2_df[source_col].tolist()]
        ints = [int(round(v)) for v in vals if v is not None]
        if len(ints) < n:
            continue
        if set(ints[:n]) == required:
            out_rows.append((attr_id, attr_name, f"{n-1}..0"))

    return pd.DataFrame(out_rows, columns=["Attribute ID", "Attribute Name", "Pattern"])


# Session state
if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None
if "parsed_oam" not in st.session_state:
    st.session_state.parsed_oam = None
if "parsed_error" not in st.session_state:
    st.session_state.parsed_error = None
if "access_ranked" not in st.session_state:
    st.session_state.access_ranked = False
if "access_correlation" not in st.session_state:
    st.session_state.access_correlation = False
if "access_estimation" not in st.session_state:
    st.session_state.access_estimation = False
if "access_validation" not in st.session_state:
    st.session_state.access_validation = False
if "access_result" not in st.session_state:
    st.session_state.access_result = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "input"
if "coco_estimation_df" not in st.session_state:
    st.session_state.coco_estimation_df = None
if "coco_estimation_html" not in st.session_state:
    st.session_state.coco_estimation_html = None
if "inverse_coco_estimation_df" not in st.session_state:
    st.session_state.inverse_coco_estimation_df = None
if "inverse_coco_estimation_html" not in st.session_state:
    st.session_state.inverse_coco_estimation_html = None
if "validation_df" not in st.session_state:
    st.session_state.validation_df = None
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "upload_filename" not in st.session_state:
    st.session_state.upload_filename = None


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload OAM CSV", type=["csv"])
    with st.expander("CSV format guide", expanded=False):
        st.caption("Required row labels in the first column: Direction ID, Attribute ID, Attribute.")
        st.caption("Direction ID rule: 0 = higher is better, 1 = lower is better.")
        st.code(_csv_template_text(), language="csv")
        st.download_button(
            "Download CSV Template",
            data=_csv_template_text().encode("utf-8-sig"),
            file_name="oam_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download Example CSV",
            data=_example_csv_bytes(),
            file_name="Example.csv",
            mime="text/csv",
            use_container_width=True,
        )

steps = 0
identifier = "Teszt"


if uploaded is not None:
    uploaded_bytes = uploaded.getvalue()
    upload_sig = hashlib.md5(uploaded_bytes).hexdigest()
    if upload_sig != st.session_state.upload_sig:
        st.session_state.upload_sig = upload_sig
        st.session_state.upload_filename = uploaded.name
        st.session_state.access_ranked = False
        st.session_state.access_correlation = False
        st.session_state.access_estimation = False
        st.session_state.access_validation = False
        st.session_state.access_result = False
        st.session_state.current_page = "input"
        st.session_state.coco_estimation_df = None
        st.session_state.coco_estimation_html = None
        st.session_state.inverse_coco_estimation_df = None
        st.session_state.inverse_coco_estimation_html = None
        st.session_state.validation_df = None
        st.session_state.result_df = None
        try:
            st.session_state.parsed_oam = read_oam_csv(uploaded_bytes)
            st.session_state.parsed_error = None
            st.session_state.access_correlation = True
        except Exception as exc:
            st.session_state.parsed_oam = None
            st.session_state.parsed_error = str(exc)
else:
    st.session_state.upload_sig = None
    st.session_state.upload_filename = None
    st.session_state.parsed_oam = None
    st.session_state.parsed_error = None
    st.session_state.access_ranked = False
    st.session_state.access_correlation = False
    st.session_state.access_estimation = False
    st.session_state.access_validation = False
    st.session_state.access_result = False
    st.session_state.current_page = "input"
    st.session_state.coco_estimation_df = None
    st.session_state.coco_estimation_html = None
    st.session_state.inverse_coco_estimation_df = None
    st.session_state.inverse_coco_estimation_html = None
    st.session_state.validation_df = None
    st.session_state.result_df = None


available_pages = {
    "input": True,
    "correlation": st.session_state.access_correlation,
    "ranked": st.session_state.access_ranked,
    "estimation": st.session_state.access_estimation,
    "validation": st.session_state.access_validation,
    "result": st.session_state.access_result,
}
if not available_pages.get(st.session_state.current_page, False):
    st.session_state.current_page = "input"

nav_items = [
    ("input", "1) Input Data"),
    ("correlation", "2) Correlation Matrix"),
    ("ranked", "3) Ranked Data"),
    ("estimation", "4) COCO Y0 Estimation"),
    ("validation", "5) Validation"),
    ("result", "6) Result"),
]
st.markdown('<div id="page-nav-anchor"></div>', unsafe_allow_html=True)
nav_cols = st.columns(len(nav_items), gap="small")
for col, (page_id, label) in zip(nav_cols, nav_items):
    with col:
        nav_label = f"> {label}" if st.session_state.current_page == page_id else label
        if st.button(
            nav_label,
            key=f"nav_{page_id}",
            use_container_width=True,
            type="secondary",
            disabled=not available_pages[page_id],
        ):
            st.session_state.current_page = page_id
            st.rerun()


if st.session_state.current_page == "input":
    st.markdown('<div class="section-title">CSV Intake</div>', unsafe_allow_html=True)
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to view objects, attributes, and input sheets.")
    elif st.session_state.parsed_error:
        st.error(f"CSV parsing failed: {st.session_state.parsed_error}")
    else:
        oam = st.session_state.parsed_oam
        if getattr(oam, "harmonized_attr_ids", None):
            harmonized = ", ".join(str(v) for v in oam.harmonized_attr_ids)
            st.warning(
                "Detected mixed ratio/percent values in these attributes and auto-harmonized them to a consistent percent scale: "
                f"{harmonized}."
            )
        object_count = len(oam.objects)
        attr_count = len(oam.attr_ids)
        y_value = oam.y.dropna().iloc[0] if oam.y.notna().any() else None
        y_text = compact_number(y_value) if y_value is not None else "-"

        st.markdown(
            (
                '<div class="intake-summary-grid">'
                '<div class="intake-item">'
                '<div class="intake-label">Objects</div>'
                f'<div class="intake-value">{object_count}</div>'
                "</div>"
                '<div class="intake-item">'
                '<div class="intake-label">Attributes</div>'
                f'<div class="intake-value">{attr_count}</div>'
                "</div>"
                '<div class="intake-item">'
                f'<div class="intake-label">{_y_display_label(oam)}</div>'
                f'<div class="intake-value">{y_text}</div>'
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="block-title">Objects + Y</div>', unsafe_allow_html=True)
        _render_input_table(_build_object_sheet(oam), height=280)
        st.caption("Y is shown as numeric target values (no unit text in headers).")

        st.markdown('<div class="block-title">Attributes</div>', unsafe_allow_html=True)
        _render_input_table(_build_attribute_sheet(oam), height=300)

        st.markdown('<div class="block-title">Input Sheet (Objects + Attributes + Y)</div>', unsafe_allow_html=True)
        _render_input_table(_build_input_sheet(oam), height=420)
        st.caption("Numeric comparisons are normalized, so equivalent values like 0.4 and 0.40 are treated as equal.")

        if st.button("Correlation Matrix", type="primary", use_container_width=True, key="corr_btn"):
            st.session_state.current_page = "correlation"
            st.rerun()

elif st.session_state.current_page == "ranked":
    oam = st.session_state.parsed_oam
    st.markdown('<div class="section-title">Ranked Data (Excel RANK.EQ style)</div>', unsafe_allow_html=True)
    ranked_df = _build_ranked_sheet(oam)
    st.dataframe(
        style_dataframe_for_display(ranked_df),
        use_container_width=True,
        height=620,
        hide_index=True,
    )
    st.caption("Ranked Data is unitless rank scale (1..n) for each attribute.")
    st.download_button(
        "Download Ranked Data (CSV)",
        data=_to_standard_csv_bytes(ranked_df),
        file_name="ranked_oam_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Run COCO Y0", type="primary", use_container_width=True, key="run_coco_btn"):
        with st.spinner("Submitting ranked data to COCO Y0 and loading output table..."):
            try:
                estimation_df, estimation_html = _run_coco_estimation(oam, int(steps), identifier)
            except Exception as exc:
                st.error("COCO Y0 run failed. Common causes: no internet access or COCO form changes.")
                st.exception(exc)
                st.stop()

        st.session_state.coco_estimation_df = estimation_df
        st.session_state.coco_estimation_html = estimation_html
        st.session_state.inverse_coco_estimation_df = None
        st.session_state.inverse_coco_estimation_html = None
        st.session_state.validation_df = None
        st.session_state.result_df = None
        st.session_state.access_estimation = True
        st.session_state.access_validation = False
        st.session_state.access_result = False
        st.session_state.current_page = "estimation"
        st.rerun()

elif st.session_state.current_page == "estimation":
    st.markdown('<div class="section-title">COCO Y0 Estimation</div>', unsafe_allow_html=True)
    totals_df = None

    if st.session_state.coco_estimation_html:
        overview, sections = extract_coco_overview(st.session_state.coco_estimation_html)
        keys = ["identifier", "objects", "attributes", "steps", "offset", "description"]
        labels = {
            "identifier": "Identifier",
            "objects": "Objects",
            "attributes": "Attributes",
            "steps": "Steps",
            "offset": "Offset",
            "description": "Description",
        }
        if overview:
            overview_df = pd.DataFrame([{labels[k]: overview.get(k, "-") for k in keys}])
            st.dataframe(
                style_dataframe_for_display(overview_df),
                use_container_width=True,
                hide_index=True,
                height=88,
            )

        if sections:
            st.markdown(" | ".join(f"**{_section_display_label(label)}**" for label in sections))

        section_tables = extract_coco_section_tables(
            st.session_state.coco_estimation_html,
            st.session_state.parsed_oam.objects if st.session_state.parsed_oam is not None else [],
        )
        for section_name in sorted(section_tables.keys(), key=_section_sort_key):
            section_df = section_tables.get(section_name, pd.DataFrame())
            if section_df.empty:
                continue
            section_df = _translate_dataframe_headers(section_df)
            section_label = _section_display_label(section_name)
            st.markdown(f'<div class="block-title">{section_label}</div>', unsafe_allow_html=True)
            st.dataframe(
                style_dataframe_for_display(section_df),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            if section_label == "Ranking":
                st.caption("Ranking values are COCO internal scores (points), not raw physical units.")
            elif section_label.startswith("Stairs"):
                st.caption("Stair values are COCO internal rank-interval calculations (points), not raw physical units.")

        totals = extract_coco_totals(st.session_state.coco_estimation_html)
        if totals:
            totals_rows = [
                ("S1 Sum", totals.get("s1_sum", "")),
                ("S20 Sum", totals.get("s20_sum", "")),
                ("Estimation Sum", totals.get("estimation_sum", "")),
                ("Actual Sum", totals.get("actual_sum", "")),
                ("Actual - Estimation Delta", totals.get("actual_estimation_delta", "")),
            ]
            totals_df = pd.DataFrame(totals_rows, columns=["Metric", "Value"])

    if st.session_state.coco_estimation_df is None:
        st.info("No estimation found yet. Use Run COCO Y0 on page 2.")
    elif st.session_state.coco_estimation_df.empty:
        st.warning("COCO response was received, but COCO:Y0 output table could not be detected.")
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
    else:
        estimation_df_for_display = _translate_dataframe_headers(st.session_state.coco_estimation_df)
        st.dataframe(
            style_dataframe_for_display(estimation_df_for_display),
            use_container_width=True,
            height=680,
            hide_index=True,
        )
        st.caption("Columns like Estimation / Fact / Delta are on COCO's internal score scale (points).")

        if totals_df is not None and not totals_df.empty:
            st.markdown('<div class="block-title">COCO Totals</div>', unsafe_allow_html=True)
            st.dataframe(
                style_dataframe_for_display(totals_df),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

        if st.button("Run Validation", type="primary", use_container_width=True, key="run_validation_btn"):
            try:
                oam = st.session_state.parsed_oam
                with st.spinner("Running inverse ranked table through COCO Y0 and computing validation..."):
                    inverse_df, inverse_html = _run_coco_inverse_estimation(oam, int(steps), identifier)
                    validation_df = _build_validation_table(oam, st.session_state.coco_estimation_df, inverse_df)
                    result_df = _build_validated_result(oam, st.session_state.coco_estimation_df, validation_df)
                if result_df.empty or validation_df.empty:
                    st.warning("Could not build validation output from COCO Y0 runs.")
                    st.stop()

                st.session_state.inverse_coco_estimation_df = inverse_df
                st.session_state.inverse_coco_estimation_html = inverse_html
                st.session_state.validation_df = validation_df
                st.session_state.result_df = result_df
                st.session_state.access_validation = True
                st.session_state.access_result = True
                st.session_state.current_page = "validation"
                st.rerun()
            except Exception as exc:
                st.error("Failed to run inverse validation.")
                st.exception(exc)

elif st.session_state.current_page == "validation":
    st.markdown('<div class="section-title">Validation</div>', unsafe_allow_html=True)
    oam = st.session_state.parsed_oam
    validation_df = st.session_state.validation_df
    inverse_coco_df = st.session_state.inverse_coco_estimation_df

    if validation_df is None or validation_df.empty or inverse_coco_df is None:
        st.info("No validation yet. Use Run Validation on the COCO Y0 Estimation page.")
    else:
        inverse_ranked_df = _build_inverse_ranked_sheet(oam)
        st.markdown('<div class="block-title">Inverse Ranked Data</div>', unsafe_allow_html=True)
        st.dataframe(
            style_dataframe_for_display(inverse_ranked_df),
            use_container_width=True,
            height=320,
            hide_index=True,
        )
        st.caption("Inverse rank formula: n + 1 - rank, where n is the number of objects.")

        st.markdown('<div class="block-title">Inverse COCO Y0 Output</div>', unsafe_allow_html=True)
        st.dataframe(
            style_dataframe_for_display(_translate_dataframe_headers(inverse_coco_df)),
            use_container_width=True,
            height=360,
            hide_index=True,
        )

        st.markdown('<div class="block-title">Binary Validation Check</div>', unsafe_allow_html=True)
        st.dataframe(
            style_dataframe_for_display(
                validation_df[
                    [
                        "Object",
                        "Baseline Delta/Fact (%)",
                        "Inverse Delta/Fact (%)",
                        "Validation",
                        "Status",
                    ]
                ]
            ),
            use_container_width=True,
            height=620,
            hide_index=True,
        )
        st.caption("Validation rule: 1 if (Delta/Fact) * (Inverse Delta/Fact) <= 0, else 0.")

        valid_count = int((validation_df["Validation"] == 1).sum())
        invalid_count = int((validation_df["Validation"] == 0).sum())
        total_count = int(len(validation_df))
        m1, m2, m3 = st.columns(3, gap="small")
        with m1:
            st.metric("Valid", valid_count)
        with m2:
            st.metric("Invalid", invalid_count)
        with m3:
            pct = (valid_count / total_count * 100.0) if total_count else 0.0
            st.metric("Valid %", f"{pct:.1f}%")

        if st.button("Open Result", type="primary", use_container_width=True, key="open_result_btn"):
            st.session_state.current_page = "result"
            st.rerun()

elif st.session_state.current_page == "result":
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
    result_df = st.session_state.result_df
    base_name = (st.session_state.upload_filename or "result").rsplit(".", 1)[0]

    if result_df is None or result_df.empty:
        st.info("No result yet. Use Run Validation on the COCO Y0 Estimation page.")
    else:
        st.markdown('<div class="block-title">Ranked Objects (Best to Least)</div>', unsafe_allow_html=True)
        st.dataframe(
            style_dataframe_for_display(result_df),
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        st.caption("Result columns Estimation, Fact, and Delta are COCO score points. Validation=1 means valid, 0 means invalid.")

        oam = st.session_state.parsed_oam
        coco_html = st.session_state.coco_estimation_html or ""
        section_tables = extract_coco_section_tables(coco_html, oam.objects if oam is not None else [])
        stairs2_df = _get_stairs2_table(section_tables)
        direct_attrs_df = _attributes_with_stair_range(oam, stairs2_df) if oam is not None else pd.DataFrame()

        est_sum = float(result_df["Estimation"].sum()) if "Estimation" in result_df.columns else None
        fact_sum = float(result_df["Fact"].dropna().sum()) if "Fact" in result_df.columns and result_df["Fact"].notna().any() else None
        sums_match = None
        if est_sum is not None and fact_sum is not None:
            sums_match = abs(est_sum - fact_sum) < 1e-9

        direction_text = ""
        if oam is not None:
            higher_count = sum(1 for d in oam.directions if int(d) == 0)
            lower_count = sum(1 for d in oam.directions if int(d) == 1)
            direction_text = (
                f"Direction rules used in ranking: {higher_count} attributes are higher-is-better, "
                f"{lower_count} attributes are lower-is-better."
            )

        best_obj = str(result_df.iloc[0]["Object"]) if len(result_df) > 0 else "-"
        least_obj = str(result_df.iloc[-1]["Object"]) if len(result_df) > 0 else "-"

        st.markdown('<div class="block-title">Interpretation</div>', unsafe_allow_html=True)
        interp_lines = [
            f"Best object by COCO estimation: {best_obj}.",
            f"Least object by COCO estimation: {least_obj}.",
            direction_text,
        ]
        if "Validation" in result_df.columns:
            valid_count = int((result_df["Validation"] == 1).sum())
            interp_lines.append(f"Validation summary: {valid_count}/{len(result_df)} objects passed the inverse-run check.")
        if sums_match is not None:
            interp_lines.append(
                f"Model check sum(Facts)=sum(Estimations): {'Yes' if sums_match else 'No'} "
                f"(Facts={compact_number(fact_sum)}, Estimations={compact_number(est_sum)})."
            )
        interp_lines = [ln for ln in interp_lines if ln.strip()]
        st.markdown("\n".join(f"- {ln}" for ln in interp_lines))

        if not direct_attrs_df.empty:
            st.markdown(
                '<div class="block-title">Attributes with stair values only from n-1 to 0</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                style_dataframe_for_display(direct_attrs_df),
                use_container_width=True,
                hide_index=True,
                height=min(320, 70 + 35 * len(direct_attrs_df)),
            )
        else:
            st.markdown("- No attributes detected with full n-1..0 stair-only pattern in Stairs(2).")

        csv_bytes = _to_standard_csv_bytes(result_df)
        xlsx_bytes = _result_to_excel_bytes(result_df)
        chart_png = _result_chart_png(result_df)

        dl_cols = st.columns(3, gap="small")
        with dl_cols[0]:
            st.download_button(
                "Download Result (CSV)",
                data=csv_bytes,
                file_name=f"{base_name}_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "Download Result (Excel)",
                data=xlsx_bytes,
                file_name=f"{base_name}_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl_cols[2]:
            if chart_png is not None:
                st.download_button(
                    "Download Graph (PNG)",
                    data=chart_png,
                    file_name=f"{base_name}_result_graph.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.write("")

        st.markdown('<div class="block-title">Estimation Graph</div>', unsafe_allow_html=True)
        has_fact_benchmark = "Fact" in result_df.columns and result_df["Fact"].notna().any()
        if has_fact_benchmark:
            st.caption("Benchmark line shown: Official (naive) benchmark based on Fact values.")
        else:
            st.caption("No official (naive) benchmark available.")
        fig = _build_result_chart(result_df)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)

elif st.session_state.current_page == "correlation":
    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    oam = st.session_state.parsed_oam
    if oam is None:
        st.info("Upload a CSV in the sidebar to calculate correlations.")
    else:
        corr_df = _build_correlation_matrix(oam)
        st.markdown("Pearson correlation across attribute columns.")
        st.dataframe(
            _style_correlation_matrix(corr_df),
            use_container_width=True,
            hide_index=False,
            height=min(900, 120 + 38 * (len(corr_df.index) + 1)),
        )

        st.markdown('<div class="block-title">Quality Assurance: User Rules vs Recommended Rules</div>', unsafe_allow_html=True)
        qa_df = _build_direction_qa_sheet(oam, corr_threshold=0.35)
        st.dataframe(
            style_dataframe_for_display(qa_df),
            use_container_width=True,
            hide_index=True,
            height=min(900, 120 + 38 * (len(qa_df.index) + 1)),
        )
        qa_summary_df = _build_unit_qa_summary(qa_df)
        if not qa_summary_df.empty:
            st.markdown('<div class="block-title">Conflict Summary</div>', unsafe_allow_html=True)
            s_cols = st.columns(len(qa_summary_df), gap="small")
            for idx, row in qa_summary_df.reset_index(drop=True).iterrows():
                with s_cols[idx]:
                    st.metric(str(row["Status"]), int(row["Count"]))

        warning_df = qa_df[qa_df["Early Warning"].isin(["High", "Medium"])]
        if warning_df.empty:
            st.success("No high/medium direction conflicts were detected.")
        else:
            st.warning(
                f"Detected {len(warning_df)} potential direction-rule conflicts. Review these before ranking."
            )
            st.dataframe(
                style_dataframe_for_display(
                    warning_df[
                        [
                            "Attribute ID",
                            "User Direction Rule",
                            "Recommended Direction Rule",
                            "Early Warning",
                            "QA Reason",
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
                height=min(520, 120 + 38 * (len(warning_df.index) + 1)),
            )

        if st.button("Rank", type="primary", use_container_width=True, key="rank_from_corr_btn"):
            st.session_state.access_ranked = True
            st.session_state.current_page = "ranked"
            st.rerun()

