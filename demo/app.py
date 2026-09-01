"""Guided Streamlit demo for the SleepTCN graduation thesis."""

from __future__ import annotations

import hashlib
import html
import sys
import tempfile
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from sleeptcn.demo import (  # noqa: E402
    DEMO_EXPERIMENTS,
    EXPERIMENT_VARIANTS,
    STAGE_NAMES,
    DemoPrediction,
    DemoRecord,
    available_fold_records,
    inspect_edf_demo,
    load_demo_models,
    load_edf_demo_records,
    load_locked_prediction,
    load_processed_demo_record,
    n3_to_n2_mask,
    predict_record,
    stage_transition_mask,
    validate_asset_manifest,
)
from sleeptcn.metrics import compute_metrics  # noqa: E402


ASSET_ROOT = WORKSPACE / "demo/assets"
PROCESSED_ROOT = WORKSPACE / "data/processed"
REFERENCE_VARIANT_ROOT = PROCESSED_ROOT / "filtered_v2"
PAGES = ("Tổng quan", "Phân tích ca", "Bằng chứng khóa luận")
MODEL_COLORS = {"Chuyên gia": "#263238", "E0": "#6D5BD0", "E3": "#0F8B8D", "E6": "#E07A5F"}
MODEL_LABELS = {
    "E0": "E0 · 15CNN + BiLSTM · tín hiệu thô",
    "E3": "E3 · ResNet-1D + TCN · lọc + scale",
    "E6": "E6 · ResNet-1D + TCN · z-score",
}
STAGE_COLORS = {"W": "#EAB308", "N1": "#F97316", "N2": "#38BDF8", "N3": "#1D4ED8", "REM": "#9333EA"}
STAGE_DESCRIPTIONS = {
    "W": "Thức",
    "N1": "Ngủ nông",
    "N2": "Ngủ ổn định",
    "N3": "Ngủ sâu",
    "REM": "Giấc ngủ REM",
}
STAGE_ORDER = ["W", "REM", "N1", "N2", "N3"]
STAGE_LEVELS = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1}
HYPNOGRAM_POSITION = np.array([0, 2, 3, 4, 1], dtype=np.int8)
CURATED_RECORDS = {
    "SC4412E": "N3→N2 tập trung gần chuyển pha",
    "SC4612E": "E3 cải thiện rõ trên bản ghi",
    "SC4601E": "trường hợp E3 chưa phải tốt nhất",
    "SC4331F": "bản ghi không có N3",
}
ERROR_FILTERS = (
    "N3 → N2 của E3",
    "Ba mô hình bất đồng",
    "E3 đúng · E0 sai",
    "E3 sai · E0 đúng",
    "N1 bị E3 phân loại sai",
)
FILTER_QUESTIONS = {
    "N3 → N2 của E3": "E3 có bỏ sót N3 bằng cách kéo nhãn về N2 ở đâu không?",
    "Ba mô hình bất đồng": "Ba pipeline có đưa ra quyết định khác nhau ở những epoch nào?",
    "E3 đúng · E0 sai": "E3 có sửa được lỗi mà pipeline E0 mắc phải ở đâu không?",
    "E3 sai · E0 đúng": "E3 có làm mất một dự đoán đúng của E0 ở đâu không?",
    "N1 bị E3 phân loại sai": "E3 gặp khó khăn với N1 ở những đoạn nào?",
}


st.set_page_config(
    page_title="SleepTCN · Sleep Record Explorer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1220px; padding-top: 1.5rem; padding-bottom: 4rem; }
    .brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.55rem; color:#18313D; }
    .brand-mark { width:35px; height:35px; border-radius:11px; display:grid; place-items:center;
      background:#0F766E; color:white; font-size:1.05rem; box-shadow:0 7px 18px rgba(15,118,110,.22); }
    .brand-name { font-size:1.08rem; font-weight:800; }
    .brand-sub { color:#718096; font-size:.78rem; }
    .hero { padding:2.5rem 2.6rem; border-radius:26px; color:white; overflow:hidden;
      background:radial-gradient(circle at 92% 18%,rgba(94,234,212,.34),transparent 30%),
                 linear-gradient(125deg,#102B3F 0%,#155E75 56%,#0F766E 100%);
      box-shadow:0 22px 60px rgba(15,52,70,.18); margin:1rem 0 1.4rem; }
    .hero-kicker { color:#99F6E4; font-size:.76rem; font-weight:850; letter-spacing:.13em;
      text-transform:uppercase; margin-bottom:.72rem; }
    .hero h1 { color:white; max-width:770px; font-size:2.55rem; line-height:1.1; margin:0 0 .85rem; }
    .hero p { color:#D9F3F0; max-width:760px; font-size:1.03rem; line-height:1.65; margin:0; }
    .hero-tags { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }
    .hero-tag { border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.1);
      color:white; padding:.32rem .62rem; border-radius:999px; font-size:.76rem; }
    .section-kicker { color:#0F766E; font-size:.73rem; font-weight:850; letter-spacing:.11em;
      text-transform:uppercase; margin:1.8rem 0 .25rem; }
    .section-title { color:#17212B; font-size:1.45rem; font-weight:780; margin:0 0 .2rem; }
    .section-copy { color:#657586; margin:0 0 1rem; line-height:1.55; }
    .feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:.9rem 0; }
    .feature-card { background:white; border:1px solid #E5EAF0; border-radius:18px; padding:1.2rem;
      box-shadow:0 7px 22px rgba(15,23,42,.045); min-height:155px; }
    .feature-icon { width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
      background:#E7F8F5; color:#0F766E; font-weight:850; margin-bottom:.8rem; }
    .feature-card h3 { color:#1B2A34; font-size:1rem; margin:0 0 .45rem; }
    .feature-card p { color:#657586; font-size:.88rem; line-height:1.5; margin:0; }
    .pipeline { display:flex; align-items:stretch; gap:.6rem; margin:1rem 0; }
    .pipe-step { flex:1; background:white; border:1px solid #E3E9EF; border-radius:15px; padding:1rem;
      text-align:center; color:#1F3440; font-weight:720; box-shadow:0 5px 16px rgba(15,23,42,.035); }
    .pipe-step small { display:block; color:#748391; font-weight:450; margin-top:.3rem; line-height:1.35; }
    .pipe-arrow { align-self:center; color:#0F766E; font-weight:900; }
    .note { border-radius:15px; padding:.92rem 1.05rem; line-height:1.55; margin:.7rem 0; }
    .note-green { background:#ECFDF5; color:#174E45; border:1px solid #BFE9DC; }
    .note-amber { background:#FFF7ED; color:#7C3F13; border:1px solid #FED7AA; }
    .note-blue { background:#EFF6FF; color:#244767; border:1px solid #BFDBFE; }
    .stat-card { background:white; border:1px solid #E5EAF0; border-radius:16px; padding:1rem 1.05rem;
      box-shadow:0 7px 22px rgba(15,23,42,.04); min-height:112px; }
    .stat-label { color:#718096; font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; font-weight:800; }
    .stat-value { color:#17212B; font-size:1.6rem; font-weight:800; margin:.28rem 0; }
    .stat-detail { color:#657586; font-size:.82rem; line-height:1.4; }
    .insight-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin:.8rem 0 1.2rem; }
    .insight-card { background:white; border:1px solid #E5EAF0; border-top:3px solid #0F8B8D; border-radius:15px;
      padding:1rem 1.05rem; box-shadow:0 7px 22px rgba(15,23,42,.035); min-height:132px; }
    .insight-card:nth-child(2) { border-top-color:#F59E0B; }
    .insight-card:nth-child(3) { border-top-color:#6D5BD0; }
    .insight-label { color:#718096; font-size:.71rem; text-transform:uppercase; letter-spacing:.08em; font-weight:850; }
    .insight-card p { color:#34495A; font-size:.88rem; line-height:1.5; margin:.45rem 0 0; }
    .decision-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin:.8rem 0; }
    .decision-cell { background:white; border:1px solid #E5EAF0; border-radius:14px; padding:.75rem .85rem; }
    .decision-label { color:#718096; font-size:.73rem; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
    .decision-stage { color:#17212B; font-size:1.32rem; font-weight:850; margin:.22rem 0; }
    .decision-status { font-size:.76rem; font-weight:750; }
    .decision-ok { color:#047857; } .decision-error { color:#B91C1C; }
    .confidence-title { color:#263746; font-weight:800; font-size:.87rem; margin-bottom:.15rem; }
    .confidence-caption { color:#718096; font-size:.76rem; line-height:1.4; margin-bottom:.35rem; }
    .table-heading { color:#263746; font-weight:800; margin:.75rem 0 .25rem; }
    .explorer-hero { margin:1rem 0 1.35rem; padding:1.6rem 1.8rem; border-radius:20px;
      background:#E7F8F5; border:1px solid #BFE9DC; color:#153C3B; }
    .explorer-hero h1 { color:#173B46; font-size:2rem; line-height:1.15; margin:0 0 .45rem; }
    .explorer-hero p { color:#416360; max-width:780px; margin:0; line-height:1.55; }
    .epoch-stage { background:white; border:1px solid #E5EAF0; border-left:6px solid #0F8B8D;
      border-radius:15px; padding:1rem 1.1rem; min-height:132px; }
    .epoch-stage-label { color:#718096; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:850; }
    .epoch-stage-value { color:#17212B; font-size:1.7rem; font-weight:850; margin:.3rem 0 .15rem; }
    .epoch-stage-detail { color:#657586; font-size:.86rem; line-height:1.45; }
    .stage-legend { display:flex; gap:.85rem; flex-wrap:wrap; margin:.35rem 0 .8rem; color:#586878; font-size:.82rem; }
    .stage-dot { width:9px; height:9px; display:inline-block; border-radius:50%; margin-right:.28rem; }
    .confidence-panel { background:white; border:1px solid #E5EAF0; border-radius:16px; padding:.8rem 1rem; }
    .confidence-row { display:grid; grid-template-columns:44px 42px 1fr 48px; gap:.55rem;
      align-items:center; padding:.55rem 0; border-bottom:1px solid #EEF2F5; }
    .confidence-row:last-child { border-bottom:0; }
    .model-chip { font-weight:850; font-size:.83rem; }
    .stage-chip { font-weight:800; font-size:.82rem; }
    .bar-track { height:9px; background:#EDF1F4; border-radius:999px; overflow:hidden; }
    .bar-fill { height:100%; border-radius:999px; }
    .bar-value { color:#536474; font-size:.8rem; text-align:right; }
    .qc-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.8rem 0 1.1rem; }
    .qc-card { background:white; border:1px solid #E5EAF0; border-radius:14px; padding:.9rem; }
    .qc-title { color:#718096; font-size:.72rem; font-weight:800; text-transform:uppercase; }
    .qc-value { color:#17212B; font-size:1rem; font-weight:780; margin-top:.32rem; }
    .qc-ok { color:#047857; } .qc-bad { color:#B91C1C; }
    div[data-testid="stMetric"] { background:white; border:1px solid #E5EAF0; border-radius:16px;
      padding:.82rem 1rem; box-shadow:0 6px 18px rgba(15,23,42,.04); }
    div[data-testid="stDataFrame"] { border:1px solid #E5EAF0; border-radius:14px; overflow:hidden; }
    button[kind="primary"], button[kind="secondary"] { border-radius:12px; font-weight:720; }
    [data-baseweb="tab-list"] { gap:.5rem; }
    @media (max-width:800px) {
      .block-container { padding:1rem; } .hero { padding:1.55rem; } .hero h1 { font-size:1.85rem; }
      .feature-grid,.qc-grid,.insight-grid { grid-template-columns:1fr; } .pipeline { flex-direction:column; }
      .decision-grid { grid-template-columns:repeat(2,1fr); }
      .pipe-arrow { transform:rotate(90deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_processed_record(path: str) -> DemoRecord:
    return load_processed_demo_record(Path(path), "filtered_v2")


@st.cache_data(show_spinner=False)
def cached_processed_record_variant(path: str, variant: str) -> DemoRecord:
    return load_processed_demo_record(Path(path), variant)


@st.cache_resource(show_spinner="Đang xác minh gói checkpoint và prediction…")
def cached_asset_manifest() -> dict:
    return validate_asset_manifest(ASSET_ROOT)


@st.cache_data(show_spinner=False)
def cached_locked_prediction(experiment_id: str, record: DemoRecord) -> DemoPrediction:
    return load_locked_prediction(
        ASSET_ROOT,
        experiment_id,
        record,
        validated_manifest=cached_asset_manifest(),
    )


def _with_temporary_edf(data: bytes, filename: str, function):
    suffix = Path(filename).suffix or ".edf"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
        return function(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def cached_edf_inspection(data: bytes, filename: str) -> dict:
    return _with_temporary_edf(data, filename, inspect_edf_demo)


@st.cache_data(show_spinner=False)
def cached_uploaded_records(data: bytes, filename: str) -> dict[str, DemoRecord]:
    loaded = _with_temporary_edf(data, filename, load_edf_demo_records)
    return {
        experiment_id: DemoRecord(
            record_key=Path(filename).stem,
            x=record.x,
            labels=None,
            original_epoch_index=record.original_epoch_index,
            source=record.source,
            note=record.note,
            data_variant=record.data_variant,
        )
        for experiment_id, record in loaded.items()
    }


@st.cache_resource(show_spinner=False)
def cached_models(experiment_id: str, device: str):
    return load_demo_models(
        ASSET_ROOT,
        experiment_id,
        device,
        validated_manifest=cached_asset_manifest(),
    )


@st.cache_data(show_spinner=False)
def load_locked_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(WORKSPACE / "runs/v2/publication/gate8" / filename)


def brand() -> None:
    st.markdown(
        """
        <div class="brand"><div class="brand-mark">◒</div><div>
          <div class="brand-name">SleepTCN Explorer</div>
          <div class="brand-sub">Single-channel EEG · Graduation thesis demo</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div><div class="section-copy">{copy}</div>',
        unsafe_allow_html=True,
    )


def navigate(page: str, source: str | None = None, record_key: str | None = None) -> None:
    st.session_state["main_page"] = page
    if source is not None:
        st.session_state["analysis_source"] = source
    if record_key is not None:
        st.session_state["selected_record_key"] = record_key


def hero() -> None:
    st.markdown(
        """
        <div class="hero"><div class="hero-kicker">Interactive evidence · Sleep staging</div>
          <h1>Nhìn thấy nơi mô hình đúng, sai và không chắc chắn</h1>
          <p>So sánh ba pipeline trên cùng một đêm ngủ, đi từ hypnogram toàn đêm đến đúng epoch EEG gây nhầm lẫn—không cần đọc hàng chục bảng kết quả.</p>
          <div class="hero-tags"><span class="hero-tag">EEG Fpz–Cz · 100 Hz</span>
            <span class="hero-tag">W / N1 / N2 / N3 / REM</span><span class="hero-tag">E0 · E3 · E6</span>
            <span class="hero-tag">Artifact đã khóa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_legend() -> None:
    content = "".join(
        f'<span><i class="stage-dot" style="background:{STAGE_COLORS[stage]}"></i>{stage}</span>'
        for stage in ("W", "N1", "N2", "N3", "REM")
    )
    st.markdown(f'<div class="stage-legend">{content}</div>', unsafe_allow_html=True)


def record_stage_summary(stages: np.ndarray) -> pd.DataFrame:
    valid_labels = stages[stages >= 0]
    counts = np.bincount(valid_labels, minlength=len(STAGE_NAMES))
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "Giai đoạn": list(STAGE_NAMES),
            "Epoch": counts.astype(int),
            "Thời lượng (phút)": counts.astype(float) * .5,
            "Tỷ lệ": counts / total if total else np.zeros(len(STAGE_NAMES)),
        }
    )


def stage_distribution_chart(summary: pd.DataFrame) -> alt.Chart:
    bars = alt.Chart(summary).mark_bar(cornerRadiusEnd=5, size=29).encode(
        y=alt.Y("Giai đoạn:N", sort=list(STAGE_NAMES), title=None),
        x=alt.X("Epoch:Q", title="Số epoch (mỗi epoch = 30 giây)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "Giai đoạn:N",
            scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Giai đoạn:N"),
            alt.Tooltip("Epoch:Q", format=".0f"),
            alt.Tooltip("Thời lượng (phút):Q", format=".1f"),
            alt.Tooltip("Tỷ lệ:Q", format=".1%"),
        ],
    )
    labels = bars.mark_text(align="left", baseline="middle", dx=5, color="#27343D").encode(
        text=alt.Text("Epoch:Q", format=".0f")
    )
    return (bars + labels).properties(height=235).configure_view(stroke=None)


def sleep_timeline_chart(
    record: DemoRecord, stages: np.ndarray, label_source: str
) -> alt.Chart:
    positions = np.flatnonzero(stages >= 0)
    frame = pd.DataFrame(
        {
            "minute": record.original_epoch_index[positions].astype(float) * .5,
            "minute_end": record.original_epoch_index[positions].astype(float) * .5 + .5,
            "stage": [stage_name(int(value)) for value in stages[positions]],
            "level": [STAGE_LEVELS[stage_name(int(value))] for value in stages[positions]],
            "band_low": [STAGE_LEVELS[stage_name(int(value))] - .36 for value in stages[positions]],
            "band_high": [STAGE_LEVELS[stage_name(int(value))] + .36 for value in stages[positions]],
            "epoch": record.original_epoch_index[positions].astype(int),
            "lane": label_source,
        }
    )
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    base = alt.Chart(frame)
    segments = base.mark_rect()
    segments = segments.encode(
        x=alt.X("minute:Q", title="Thời gian từ đầu bản ghi (phút)"),
        x2="minute_end:Q",
        y=alt.Y(
            "band_high:Q",
            title="Giai đoạn ngủ",
            scale=alt.Scale(domain=[0, 5.25]),
            axis=alt.Axis(
                values=[1, 2, 3, 4, 5],
                labelExpr="datum.value === 5 ? 'W' : datum.value === 4 ? 'REM' : datum.value === 3 ? 'N1' : datum.value === 2 ? 'N2' : 'N3'",
                labelFontWeight="bold",
                labelPadding=9,
            ),
        ),
        y2="band_low:Q",
        color=alt.Color(
            "stage:N",
            scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("minute:Q", title="Phút", format=".1f"),
            alt.Tooltip("epoch:Q", title="Epoch gốc", format=".0f"),
            alt.Tooltip("stage:N", title="Giai đoạn"),
            alt.Tooltip("lane:N", title="Nguồn"),
        ],
    )
    return (
        alt.layer(segments)
        .add_params(zoom)
        .properties(height=270)
        .configure_view(stroke="#E5EAF0", strokeWidth=1)
        .configure_axis(gridColor="#EDF1F4", labelColor="#536474", titleColor="#536474")
    )


def epoch_stage_html(
    record: DemoRecord,
    stages: np.ndarray,
    position: int,
    label_source: str,
    confidence: float | None = None,
) -> str:
    stage = stage_name(int(stages[position]))
    if stage == "Bỏ qua":
        return (
            '<div class="epoch-stage" style="border-left-color:#94A3B8">'
            f'<div class="epoch-stage-label">{html.escape(label_source)} của epoch đang xem</div>'
            '<div class="epoch-stage-value">Không có nhãn</div>'
            '<div class="epoch-stage-detail">Epoch này được giữ trong trục thời gian nhưng không dùng để tính thống kê.</div></div>'
        )
    description = STAGE_DESCRIPTIONS[stage]
    minute = record.original_epoch_index[position] * .5
    confidence_text = (
        f" · confidence {confidence:.0%}" if confidence is not None else ""
    )
    return (
        f'<div class="epoch-stage" style="border-left-color:{STAGE_COLORS[stage]}">'
        f'<div class="epoch-stage-label">{html.escape(label_source)} của epoch đang xem</div>'
        f'<div class="epoch-stage-value">{stage} · {html.escape(description)}</div>'
        f'<div class="epoch-stage-detail">Epoch gốc #{int(record.original_epoch_index[position])} · phút {minute:g}–{minute + .5:g} · 30 giây EEG{confidence_text}</div></div>'
    )


def render_stage_record(
    record: DemoRecord,
    stages: np.ndarray,
    label_source: str,
    *,
    slider_key: str,
    display_record: DemoRecord | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    if len(stages) != len(record.x):
        st.error("Số dự đoán không khớp số epoch của bản ghi.")
        return
    if probabilities is not None and probabilities.shape != (len(record.x), len(STAGE_NAMES)):
        st.error("Confidence không khớp số epoch hoặc số lớp của prediction.")
        return
    display_record = display_record or record
    if (
        len(display_record.x) != len(record.x)
        or not np.array_equal(display_record.original_epoch_index, record.original_epoch_index)
    ):
        st.error("Tín hiệu EEG hiển thị không căn chỉnh với prediction.")
        return
    summary = record_stage_summary(stages)
    valid_epochs = int((stages >= 0).sum())
    epoch_label = "Epoch có nhãn" if label_source == "Nhãn chuyên gia" else "Epoch có dự đoán"
    top_left, top_middle, top_right = st.columns(3)
    top_left.metric("Bản ghi", record.record_key)
    top_middle.metric("Thời lượng phân tích", format_duration(valid_epochs))
    top_right.metric(epoch_label, f"{valid_epochs:,}")

    section(
        "01 · Phân bố giai đoạn ngủ",
        "Đêm ngủ này gồm bao nhiêu epoch ở từng giai đoạn?",
        f"Mỗi epoch dài 30 giây. Dữ liệu giai đoạn bên dưới là: {label_source}. Biểu đồ cho biết số epoch; bảng bên cạnh cho biết thêm thời lượng và tỷ lệ.",
    )
    chart_column, table_column = st.columns([1.05, .95])
    with chart_column:
        st.altair_chart(stage_distribution_chart(summary), width="stretch")
    with table_column:
        st.dataframe(
            summary.style.format({"Thời lượng (phút)": "{:.1f}", "Tỷ lệ": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )
    stage_legend()

    section(
        "02 · Nhìn toàn bộ đêm ngủ",
        "Giai đoạn nào xuất hiện ở thời điểm nào?",
        f"Đường bậc thang đặt W/REM/N1/N2/N3 ở trục trái; mỗi điểm là một epoch 30 giây theo {label_source}. Rê chuột để xem thời điểm và kéo ngang để phóng to một vùng.",
    )
    st.altair_chart(sleep_timeline_chart(record, stages, label_source), width="stretch")

    section(
        "03 · Xem sóng EEG của một epoch",
        "Lọc theo giai đoạn, rồi đối chiếu sóng EEG với kết quả phân giai đoạn của epoch đó",
        "Đây là công cụ trực quan hóa dữ liệu nghiên cứu, không phải chẩn đoán y khoa. Một epoch 30 giây không nên được diễn giải tách rời toàn bộ bản ghi.",
    )
    valid_positions = np.flatnonzero(stages >= 0)
    if len(valid_positions) == 0:
        st.info("Bản ghi không có epoch phù hợp để hiển thị.")
        return
    stage_filter = st.selectbox(
        "Chỉ xem epoch thuộc giai đoạn",
        ["Tất cả", *STAGE_NAMES],
        format_func=lambda value: (
            "Tất cả giai đoạn" if value == "Tất cả" else f"{value} · {STAGE_DESCRIPTIONS[value]}"
        ),
        key=f"{slider_key}_stage_filter",
    )
    positions = (
        valid_positions
        if stage_filter == "Tất cả"
        else np.flatnonzero(stages == STAGE_NAMES.index(stage_filter))
    )
    if len(positions) == 0:
        st.info(f"Không có epoch {stage_filter} trong bản ghi này.")
        return
    selection = st.slider(
        "Chọn epoch trong danh sách đã lọc",
        min_value=1,
        max_value=len(positions),
        value=(len(positions) + 1) // 2,
        step=1,
        key=f"{slider_key}_epoch",
    )
    position = int(positions[selection - 1])
    confidence = (
        float(probabilities[position, int(stages[position])])
        if probabilities is not None
        else None
    )
    st.caption(f"Đang xem epoch {selection}/{len(positions)} trong bộ lọc · epoch gốc #{int(record.original_epoch_index[position])}.")
    signal_column, stage_column = st.columns([1.3, .7])
    with signal_column:
        st.altair_chart(eeg_chart(display_record, position), width="stretch")
        st.caption(
            "Trục ngang: 30 giây trong epoch. Trục dọc: biên độ EEG gần đơn vị µV. "
            "Sóng này luôn là EEG gốc để E0 và E3 có thể được đối chiếu trên cùng tín hiệu."
        )
    with stage_column:
        st.markdown(
            epoch_stage_html(record, stages, position, label_source, confidence),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="note note-blue"><b>Cách dùng khi demo.</b> Chọn một vùng trên timeline, kéo thanh đến thời điểm tương ứng, sau đó đọc sóng EEG cùng {html.escape(label_source)}. Confidence là softmax của model, không phải xác suất đúng đã hiệu chuẩn.</div>',
            unsafe_allow_html=True,
        )


def render_uploaded_record_explorer() -> None:
    model = st.segmented_control(
        "Mô hình phân giai đoạn",
        ["E3", "E0"],
        default="E3",
        key="uploaded_model",
        width="stretch",
    )
    uploaded = st.file_uploader("Tải một file EDF", type=["edf", "rec"])
    if uploaded is None:
        st.info(f"Tải EDF có kênh EEG Fpz-Cz, 100 Hz và đơn vị µV để chạy dự đoán {model}.")
        return
    payload = uploaded.getvalue()
    inspection = cached_edf_inspection(payload, uploaded.name)
    if not inspection["ready"]:
        st.error("EDF chưa phù hợp để chạy demo. Cần EEG Fpz-Cz, 100 Hz, đơn vị µV và ít nhất một epoch 30 giây.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(
        f"Đầu vào hợp lệ: {inspection['complete_epochs']:,} epoch hoàn chỉnh. Demo chỉ chạy {model} trên {device.upper()} và không so sánh thêm mô hình khác."
    )
    digest = hashlib.sha256(payload).hexdigest()
    run_key = f"{digest}:{model}:{device}"
    if st.button("Phân tích bản ghi", type="primary", width="stretch"):
        try:
            with st.spinner(f"Đang tiền xử lý và chạy {model}…"):
                records = cached_uploaded_records(payload, uploaded.name)
                record = records[model]
                display_record = records["E0"]
                prediction = predict_record(record, cached_models(model, device))
            st.session_state["simple_edf_result"] = (
                run_key,
                record,
                display_record,
                prediction,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Không thể hoàn tất suy luận {model}. Hãy kiểm tra lại gói checkpoint demo và file EDF.")
            st.caption(str(error))
    stored = st.session_state.get("simple_edf_result")
    if not stored or len(stored) != 4 or stored[0] != run_key:
        return
    _, record, display_record, prediction = stored
    st.markdown(
        f'<div class="note note-amber"><b>Lưu ý.</b> Các giai đoạn dưới đây là dự đoán của {model}, không phải nhãn chuyên gia. Chúng chỉ phục vụ minh họa pipeline.</div>',
        unsafe_allow_html=True,
    )
    render_stage_record(
        record,
        prediction.predicted,
        f"Dự đoán {model}",
        slider_key=f"uploaded_{model}_{digest[:12]}",
        display_record=display_record,
        probabilities=prediction.probabilities,
    )


def render_sleep_record_explorer() -> None:
    st.markdown(
        """
        <div class="explorer-hero">
          <h1>Khám phá một đêm ngủ từ EEG</h1>
          <p>Chọn một bản ghi mẫu hoặc tải EDF để chạy E3/E0. Demo chỉ làm ba việc: đếm giai đoạn ngủ, hiển thị chúng theo thời gian và cho xem sóng EEG của từng epoch.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source = st.segmented_control(
        "Nguồn bản ghi",
        ["Bản ghi mẫu", "Tải EDF"],
        default="Bản ghi mẫu",
        width="stretch",
    )
    if source == "Tải EDF":
        render_uploaded_record_explorer()
        return
    paths = {path.stem: path for path in sorted(REFERENCE_VARIANT_ROOT.glob("*.npz"))}
    if not paths:
        st.error("Không tìm thấy bản ghi đã xử lý trong data/processed/filtered_v2.")
        return
    suggested = [key for key in ("SC4412E", "SC4612E", "SC4601E", "SC4331F") if key in paths]
    choices = suggested or list(paths)
    selected_key = st.selectbox("Chọn bản ghi Sleep-EDF", choices)
    model = st.segmented_control(
        "Mô hình phân giai đoạn",
        ["E3", "E0"],
        default="E3",
        key="sample_model",
        width="stretch",
    )
    variant = EXPERIMENT_VARIANTS[model]
    model_path = PROCESSED_ROOT / variant / f"{selected_key}.npz"
    if not model_path.is_file():
        st.error(f"Không tìm thấy dữ liệu {variant} cho {selected_key}.")
        return
    record = cached_processed_record_variant(str(model_path), variant)
    raw_path = PROCESSED_ROOT / "paper_raw_v1" / f"{selected_key}.npz"
    if not raw_path.is_file():
        st.error(f"Không tìm thấy EEG gốc cho {selected_key}.")
        return
    display_record = cached_processed_record_variant(str(raw_path), "paper_raw_v1")
    try:
        prediction = cached_locked_prediction(model, record)
    except (FileNotFoundError, ValueError) as error:
        st.error("Không thể nạp prediction artifact đã khóa cho bản ghi mẫu.")
        st.code(
            "python scripts/prepare_demo_assets.py --ref run-in-docker --fold 0 --seed 123",
            language="powershell",
        )
        st.caption(str(error))
        return
    st.caption(f"Đang xem prediction artifact đã khóa của {model} trên {selected_key}; không chạy lại mô hình trong lúc trình diễn.")
    render_stage_record(
        record,
        prediction.predicted,
        f"Dự đoán {model}",
        slider_key=f"sample_{model}_{record.record_key}",
        display_record=display_record,
        probabilities=prediction.probabilities,
    )


def render_overview() -> None:
    hero()
    action_left, action_right, _ = st.columns([1, 1, 1.2])
    action_left.button(
        "Khám phá một ca điển hình",
        type="primary",
        width="stretch",
        on_click=navigate,
        args=("Phân tích ca", "Bản ghi kiểm thử", "SC4412E"),
    )
    action_right.button(
        "Phân tích EDF của tôi",
        width="stretch",
        on_click=navigate,
        args=("Phân tích ca", "Tải EDF mới", None),
    )
    section(
        "Hệ thống làm gì?",
        "Một luồng từ tín hiệu đến bằng chứng",
        "Mỗi bước đều có đầu ra quan sát được; người xem không phải tin vào một con số cuối cùng.",
    )
    st.markdown(
        """
        <div class="pipeline"><div class="pipe-step">EDF / Sleep-EDF<small>Một kênh EEG Fpz–Cz</small></div>
          <div class="pipe-arrow">→</div><div class="pipe-step">Kiểm tra tín hiệu<small>Kênh · tần số · đơn vị · thời lượng</small></div>
          <div class="pipe-arrow">→</div><div class="pipe-step">E0 / E3 / E6<small>Ba lựa chọn kiến trúc và tiền xử lý</small></div>
          <div class="pipe-arrow">→</div><div class="pipe-step">Hypnogram<small>So sánh, phóng đại và xuất kết quả</small></div></div>
        """,
        unsafe_allow_html=True,
    )
    section(
        "Ba câu hỏi",
        "Demo chỉ giữ lại những gì giúp trả lời",
        "Không đưa toàn bộ Gate và cấu hình kỹ thuật lên màn hình chính.",
    )
    st.markdown(
        """
        <div class="feature-grid">
          <div class="feature-card"><div class="feature-icon">1</div><h3>Ba mô hình khác nhau ở đâu?</h3><p>Hypnogram được căn theo cùng epoch để sai khác có thể nhìn thấy ngay.</p></div>
          <div class="feature-card"><div class="feature-icon">2</div><h3>Lỗi quan trọng nằm chỗ nào?</h3><p>Bộ điều hướng đưa thẳng đến N3→N2, lỗi N1 hoặc vùng các mô hình bất đồng.</p></div>
          <div class="feature-card"><div class="feature-icon">3</div><h3>Phát hiện có ý nghĩa gì?</h3><p>EEG, confidence và câu giải thích thay đổi theo đúng epoch đang chọn.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section("Kết quả chính", "Ba con số cần nhớ", "Số liệu tổng hợp đã khóa, không lấy từ ca minh họa.")
    columns = st.columns(3)
    columns[0].metric("E3 Macro-F1 · 10 fold", "0,7904")
    columns[1].metric("Suy luận nhanh hơn E0", "3,76×")
    columns[2].metric("Huấn luyện E3 · 10 fold", "3 giờ 16 phút")
    st.markdown(
        '<div class="note note-blue"><b>Thông điệp.</b> Giá trị của project không nằm ở một kiến trúc mới vượt SOTA, mà ở đánh giá có kiểm soát, lợi ích vận hành đo được và việc xác định rõ phần lỗi còn lại.</div>',
        unsafe_allow_html=True,
    )


def record_predictions(record: DemoRecord) -> dict[str, DemoPrediction]:
    return {
        experiment_id: cached_locked_prediction(experiment_id, record)
        for experiment_id in DEMO_EXPERIMENTS
    }


def stage_name(value: int) -> str:
    return STAGE_NAMES[value] if 0 <= value <= 4 else "Bỏ qua"


def event_mask(
    record: DemoRecord, predictions: dict[str, DemoPrediction], filter_name: str
) -> np.ndarray:
    assert record.labels is not None
    labels = record.labels
    e0 = predictions["E0"].predicted
    e3 = predictions["E3"].predicted
    e6 = predictions["E6"].predicted
    valid = labels >= 0
    if filter_name == "N3 → N2 của E3":
        return n3_to_n2_mask(labels, e3)
    if filter_name == "Ba mô hình bất đồng":
        return valid & ((e0 != e3) | (e0 != e6) | (e3 != e6))
    if filter_name == "E3 đúng · E0 sai":
        return valid & (e3 == labels) & (e0 != labels)
    if filter_name == "E3 sai · E0 đúng":
        return valid & (e3 != labels) & (e0 == labels)
    if filter_name == "N1 bị E3 phân loại sai":
        return (labels == 1) & (e3 != 1)
    raise ValueError(filter_name)


def event_label(
    record: DemoRecord, predictions: dict[str, DemoPrediction], index: int
) -> str:
    assert record.labels is not None
    true_stage = stage_name(int(record.labels[index]))
    e3_stage = stage_name(int(predictions["E3"].predicted[index]))
    return f"Phút {record.original_epoch_index[index] * .5:g} · {true_stage} → {e3_stage}"


def hypnogram_frame(
    record: DemoRecord,
    predictions: dict[str, DemoPrediction],
    start: int = 0,
    stop: int | None = None,
) -> pd.DataFrame:
    stop = len(record.x) if stop is None else stop
    indices = np.arange(start, stop)
    minutes = record.original_epoch_index[indices].astype(float) * .5
    rows: list[dict] = []
    arrays: list[tuple[str, np.ndarray]] = []
    if record.labels is not None:
        arrays.append(("Chuyên gia", record.labels))
    arrays.extend(
        (experiment_id, predictions[experiment_id].predicted)
        for experiment_id in DEMO_EXPERIMENTS
        if experiment_id in predictions
    )
    disagreement = np.zeros(len(record.x), dtype=bool)
    if len(predictions) > 1:
        stack = np.stack([prediction.predicted for prediction in predictions.values()])
        disagreement = np.any(stack != stack[0], axis=0)
    e3_error = (
        n3_to_n2_mask(record.labels, predictions["E3"].predicted)
        if record.labels is not None and "E3" in predictions
        else np.zeros(len(record.x), dtype=bool)
    )
    for source, values in arrays:
        for offset, position in enumerate(indices):
            value = int(values[position])
            region = ""
            if source == "E3" and e3_error[position]:
                region = "N3 → N2"
            elif source == "E3" and disagreement[position]:
                region = "Mô hình bất đồng"
            rows.append(
                {
                    "minute": float(minutes[offset]),
                    "minute_end": float(minutes[offset] + .5),
                    "source": source,
                    "stage": stage_name(value) if 0 <= value <= 4 else None,
                    "true_stage": stage_name(int(record.labels[position]))
                    if record.labels is not None
                    else "Không có nhãn",
                    "E0": stage_name(int(predictions["E0"].predicted[position]))
                    if "E0" in predictions
                    else "—",
                    "E3": stage_name(int(predictions["E3"].predicted[position]))
                    if "E3" in predictions
                    else "—",
                    "E6": stage_name(int(predictions["E6"].predicted[position]))
                    if "E6" in predictions
                    else "—",
                    "confidence": float(predictions[source].probabilities[position].max())
                    if source in predictions
                    else None,
                    "region": region,
                }
            )
    return pd.DataFrame(rows)


def hypnogram_chart(frame: pd.DataFrame, source_order: list[str]) -> alt.Chart:
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    base = alt.Chart(frame)
    tiles = (
        base.transform_filter("datum.stage != null")
        .mark_rect(stroke="#FFFFFF", strokeWidth=.35)
        .encode(
            x=alt.X("minute:Q", title="Thời gian từ đầu bản ghi (phút)"),
            x2="minute_end:Q",
            y=alt.Y(
                "source:N",
                sort=source_order,
                title=None,
                scale=alt.Scale(paddingInner=.22, paddingOuter=.18),
                axis=alt.Axis(labelFontWeight="bold", labelPadding=9),
            ),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(
                    domain=list(STAGE_COLORS),
                    range=list(STAGE_COLORS.values()),
                ),
                legend=alt.Legend(title="Màu = giai đoạn ngủ", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("minute:Q", title="Phút", format=".1f"),
                alt.Tooltip("source:N", title="Nguồn"),
                alt.Tooltip("stage:N", title="Giai đoạn"),
                alt.Tooltip("true_stage:N", title="Nhãn chuyên gia"),
                alt.Tooltip("E0:N"),
                alt.Tooltip("E3:N"),
                alt.Tooltip("E6:N"),
                alt.Tooltip("confidence:Q", title="Confidence", format=".1%"),
                alt.Tooltip("region:N", title="Đánh dấu"),
            ],
        )
    )
    error_outline = (
        base.transform_filter(alt.datum.region != "")
        .mark_rect(fillOpacity=0, strokeWidth=2.2)
        .encode(
            x="minute:Q",
            x2="minute_end:Q",
            y=alt.Y("source:N", sort=source_order),
            stroke=alt.Color(
                "region:N",
                scale=alt.Scale(
                    domain=["N3 → N2", "Mô hình bất đồng"],
                    range=["#DC2626", "#F59E0B"],
                ),
                legend=alt.Legend(title="Viền = cần chú ý", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("minute:Q", title="Phút", format=".1f"),
                alt.Tooltip("region:N", title="Đánh dấu"),
                alt.Tooltip("true_stage:N", title="Nhãn chuyên gia"),
            ],
        )
    )
    return (
        alt.layer(tiles, error_outline)
        .add_params(zoom)
        .properties(height=max(150, 52 * len(source_order)))
        .configure_view(stroke="#E5EAF0", strokeWidth=1)
        .configure_axis(gridColor="#EDF1F4", labelColor="#536474", titleColor="#536474")
    )


def eeg_chart(record: DemoRecord, position: int) -> alt.Chart:
    scale = 1.0 if record.data_variant == "paper_raw_v1" else 100.0
    signal = record.x[position] * scale
    frame = pd.DataFrame({"second": np.arange(3000) / 100.0, "amplitude_uv": signal})
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return (
        alt.Chart(frame)
        .mark_line(color="#0F766E", strokeWidth=1)
        .encode(
            x=alt.X("second:Q", title="Giây trong epoch"),
            y=alt.Y("amplitude_uv:Q", title="Biên độ (µV)"),
            tooltip=[
                alt.Tooltip("second:Q", title="Giây", format=".2f"),
                alt.Tooltip("amplitude_uv:Q", title="µV", format=".1f"),
            ],
        )
        .add_params(zoom)
        .properties(height=260)
        .configure_view(stroke=None)
    )


def confidence_html(
    record: DemoRecord, predictions: dict[str, DemoPrediction], position: int
) -> str:
    rows = []
    for experiment_id in DEMO_EXPERIMENTS:
        prediction = predictions[experiment_id]
        predicted = int(prediction.predicted[position])
        confidence = float(prediction.probabilities[position, predicted])
        color = MODEL_COLORS[experiment_id]
        rows.append(
            f'<div class="confidence-row"><span class="model-chip" style="color:{color}">{experiment_id}</span>'
            f'<span class="stage-chip">{stage_name(predicted)}</span><div class="bar-track">'
            f'<div class="bar-fill" style="width:{confidence * 100:.1f}%;background:{color}"></div></div>'
            f'<span class="bar-value">{confidence:.0%}</span></div>'
        )
    return (
        '<div class="confidence-panel">'
        '<div class="confidence-title">Confidence của lớp dự đoán</div>'
        '<div class="confidence-caption">Xác suất softmax của lớp mà mô hình đã chọn; đây không phải xác suất đúng và chưa được hiệu chuẩn lâm sàng.</div>'
        + "".join(rows)
        + "</div>"
    )


def event_explanation(
    record: DemoRecord, predictions: dict[str, DemoPrediction], position: int
) -> str:
    assert record.labels is not None
    truth = int(record.labels[position])
    e0 = int(predictions["E0"].predicted[position])
    e3 = int(predictions["E3"].predicted[position])
    e6 = int(predictions["E6"].predicted[position])
    near_transition = bool(stage_transition_mask(record.labels)[position])
    if truth == 3 and e3 == 2:
        agreement = "E6 cũng dự đoán N2" if e6 == 2 else f"E6 dự đoán {stage_name(e6)}"
        baseline = "E0 giữ đúng N3" if e0 == 3 else f"E0 dự đoán {stage_name(e0)}"
        location = "nằm gần một chuyển pha thật" if near_transition else "không nằm gần chuyển pha thật"
        return (
            f"Tại epoch này, nhãn chuyên gia là N3 nhưng E3 chuyển sang N2. {baseline}; {agreement}. "
            f"Điểm lỗi {location}, vì vậy đây là ví dụ trực tiếp về độ bất định tại ranh giới N2–N3."
        )
    correct = [key for key in DEMO_EXPERIMENTS if int(predictions[key].predicted[position]) == truth]
    if len(correct) == 3:
        return f"Cả ba mô hình đều khớp nhãn chuyên gia {stage_name(truth)} tại epoch này."
    if not correct:
        return f"Không mô hình nào khớp nhãn chuyên gia {stage_name(truth)} tại epoch này; đây là lỗi chung thay vì lỗi riêng của E3."
    return f"Nhãn chuyên gia là {stage_name(truth)}. Các mô hình dự đoán đúng tại epoch này: {', '.join(correct)}."


def model_summary(record: DemoRecord, predictions: dict[str, DemoPrediction]) -> pd.DataFrame:
    assert record.labels is not None
    transition = stage_transition_mask(record.labels)
    rows = []
    for experiment_id in DEMO_EXPERIMENTS:
        prediction = predictions[experiment_id]
        metrics = compute_metrics(record.labels, prediction.predicted)
        errors = n3_to_n2_mask(record.labels, prediction.predicted)
        rows.append(
            {
                "Mô hình": experiment_id,
                "Macro-F1": metrics["macro_f1"],
                "Accuracy": metrics["accuracy"],
                "N3 → N2": int(errors.sum()),
                "Gần chuyển pha": int((errors & transition).sum()),
            }
        )
    return pd.DataFrame(rows)


def format_duration(epoch_count: int) -> str:
    minutes = epoch_count * 0.5
    if minutes >= 60:
        return f"{minutes / 60:.1f} giờ · {minutes:.0f} phút"
    return f"{minutes:.0f} phút"


def record_insight_html(
    record: DemoRecord,
    summary: pd.DataFrame,
    filter_name: str,
    event_count: int,
) -> str:
    indexed = summary.set_index("Mô hình")
    e3_errors = int(indexed.loc["E3", "N3 → N2"])
    e3_near = int(indexed.loc["E3", "Gần chuyển pha"])
    e0_errors = int(indexed.loc["E0", "N3 → N2"])
    e6_errors = int(indexed.loc["E6", "N3 → N2"])
    if filter_name == "N3 → N2 của E3":
        question = FILTER_QUESTIONS[filter_name]
        observation = (
            f"E3 có {e3_errors} lỗi N3→N2; {e3_near}/{e3_errors} "
            f"({e3_near / e3_errors:.0%}) nằm trong vùng ±2 epoch quanh chuyển pha."
            if e3_errors
            else "Không tìm thấy lỗi N3→N2 của E3 trong record này."
        )
        meaning = (
            f"Đây là bằng chứng về vị trí lỗi trong record {record.record_key}, không phải bằng chứng E3 luôn tốt hơn. "
            f"Trong cùng record, E0 có {e0_errors} và E6 có {e6_errors} lỗi N3→N2."
        )
    elif filter_name == "Ba mô hình bất đồng":
        question = FILTER_QUESTIONS[filter_name]
        observation = f"Có {event_count} epoch mà ít nhất một trong E0, E3 và E6 đưa ra nhãn khác nhau."
        meaning = "Bất đồng là vùng cần xem lại hoặc ưu tiên gắn nhãn; tự nó không cho biết mô hình nào đúng."
    elif filter_name == "E3 đúng · E0 sai":
        question = FILTER_QUESTIONS[filter_name]
        observation = f"Có {event_count} epoch E3 khớp nhãn chuyên gia trong khi E0 không khớp."
        meaning = "Đây là ví dụ E3 có lợi thế cục bộ so với E0, không thay thế so sánh bắt cặp trên toàn bộ test fold."
    elif filter_name == "E3 sai · E0 đúng":
        question = FILTER_QUESTIONS[filter_name]
        observation = f"Có {event_count} epoch E3 sai trong khi E0 đúng."
        meaning = "Record này cho thấy E3 vẫn có giới hạn và giúp tránh diễn giải kết quả theo hướng E3 luôn vượt trội."
    else:
        question = FILTER_QUESTIONS[filter_name]
        observation = f"Có {event_count} epoch có nhãn thật N1 nhưng E3 dự đoán sang lớp khác."
        meaning = "N1 là vùng khó của bài toán; cần đọc cùng F1 theo lớp thay vì chỉ dựa vào Macro-F1."
    return (
        '<div class="insight-grid">'
        f'<div class="insight-card"><div class="insight-label">Câu hỏi</div><p>{html.escape(question)}</p></div>'
        f'<div class="insight-card"><div class="insight-label">Quan sát trong record</div><p>{html.escape(observation)}</p></div>'
        f'<div class="insight-card"><div class="insight-label">Ý nghĩa</div><p>{html.escape(meaning)}</p></div>'
        "</div>"
    )


def event_decision_html(
    record: DemoRecord, predictions: dict[str, DemoPrediction], position: int
) -> str:
    assert record.labels is not None
    truth = int(record.labels[position])
    cells = [
        (
            "Nhãn chuyên gia",
            stage_name(truth),
            "Mốc tham chiếu",
            "decision-ok",
        )
    ]
    for experiment_id in DEMO_EXPERIMENTS:
        predicted = int(predictions[experiment_id].predicted[position])
        matches = predicted == truth
        cells.append(
            (
                experiment_id,
                stage_name(predicted),
                "Khớp nhãn" if matches else "Lệch nhãn",
                "decision-ok" if matches else "decision-error",
            )
        )
    rendered = "".join(
        f'<div class="decision-cell"><div class="decision-label">{html.escape(label)}</div>'
        f'<div class="decision-stage">{html.escape(stage)}</div>'
        f'<div class="decision-status {css}">{html.escape(status)}</div></div>'
        for label, stage, status, css in cells
    )
    return f'<div class="decision-grid">{rendered}</div>'


def prediction_frame(record: DemoRecord, predictions: dict[str, DemoPrediction]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "record_key": record.record_key,
            "original_epoch_index": record.original_epoch_index,
            "time_minute": record.original_epoch_index * .5,
        }
    )
    if record.labels is not None:
        frame["true_stage"] = [stage_name(int(value)) for value in record.labels]
    for experiment_id, prediction in predictions.items():
        frame[f"{experiment_id}_stage"] = [stage_name(int(value)) for value in prediction.predicted]
        frame[f"{experiment_id}_confidence"] = prediction.probabilities.max(axis=1)
    return frame


def move_event(state_key: str, events: list[int], delta: int) -> None:
    current = st.session_state.get(state_key, events[0])
    position = events.index(current) if current in events else 0
    st.session_state[state_key] = events[(position + delta) % len(events)]


def render_test_analysis() -> None:
    record_paths = available_fold_records(
        ASSET_ROOT,
        REFERENCE_VARIANT_ROOT,
        validated_manifest=cached_asset_manifest(),
    )
    if not record_paths:
        st.error("Không tìm thấy dữ liệu test fold tại data/processed/filtered_v2.")
        return
    paths = {path.stem: path for path in record_paths}
    keys = list(paths)
    preferred = [key for key in CURATED_RECORDS if key in paths]
    ordered_keys = preferred + [key for key in keys if key not in preferred]
    selected_default = st.session_state.get("selected_record_key", "SC4412E")
    if selected_default not in ordered_keys:
        selected_default = ordered_keys[0]
    st.session_state["selected_record_key"] = selected_default

    control_a, control_b = st.columns([1.2, .8])
    with control_a:
        selected_key = st.selectbox(
            "Bản ghi kiểm thử",
            ordered_keys,
            key="selected_record_key",
            format_func=lambda value: (
                f"{value} — {CURATED_RECORDS[value]}" if value in CURATED_RECORDS else value
            ),
        )
    with control_b:
        view_mode = st.segmented_control(
            "Phạm vi hiển thị",
            ["Vùng quanh sự kiện", "Toàn bộ đêm"],
            default="Vùng quanh sự kiện",
            width="stretch",
        )
    record = cached_processed_record(str(paths[selected_key]))
    predictions = record_predictions(record)
    summary = model_summary(record, predictions)
    indexed = summary.set_index("Mô hình")
    e3_row = indexed.loc["E3"]

    metrics = st.columns(4)
    metrics[0].metric(
        "Bản ghi",
        record.record_key,
        CURATED_RECORDS.get(record.record_key, "đối chiếu test-fold"),
        delta_color="off",
    )
    metrics[1].metric(
        "Thời lượng sau trim",
        format_duration(len(record.x)),
        f"{int((record.labels >= 0).sum()):,} epoch có nhãn",
        delta_color="off",
    )
    metrics[2].metric(
        "E3 Macro-F1 · record",
        f"{float(e3_row['Macro-F1']):.3f}",
        "chỉ số của bản ghi đang chọn",
        delta_color="off",
    )
    metrics[3].metric(
        "Lỗi E3 · N3 → N2",
        f"{int(e3_row['N3 → N2'])}",
        f"{int(e3_row['Gần chuyển pha'])} gần chuyển pha",
        delta_color="off",
    )

    section(
        "01 · Tìm vùng cần xem",
        "Đặt một câu hỏi trước khi đọc biểu đồ",
        "Bộ lọc chọn đúng loại sự kiện cần kiểm tra; timeline chỉ định vị, còn bảng và ghi chú bên dưới giải thích ý nghĩa của số liệu.",
    )
    filter_name = st.selectbox(
        "Câu hỏi cần kiểm tra",
        ERROR_FILTERS,
        help=FILTER_QUESTIONS[ERROR_FILTERS[0]],
    )
    mask = event_mask(record, predictions, filter_name)
    events = [int(value) for value in np.flatnonzero(mask)]
    selected_event: int | None = None
    if events:
        event_key = f"event_{record.record_key}_{ERROR_FILTERS.index(filter_name)}"
        if st.session_state.get(event_key) not in events:
            transition = stage_transition_mask(record.labels)
            dense_scores = {
                value: int(mask[max(0, value - 20) : min(len(mask), value + 21)].sum())
                + int(transition[value])
                for value in events
            }
            st.session_state[event_key] = max(events, key=lambda value: dense_scores[value])
        previous, selector, following = st.columns([.18, .64, .18])
        previous.button(
            "← Trước",
            width="stretch",
            on_click=move_event,
            args=(event_key, events, -1),
        )
        with selector:
            selected_event = st.selectbox(
                "Sự kiện đang xem",
                events,
                key=event_key,
                format_func=lambda value: event_label(record, predictions, value),
                label_visibility="collapsed",
            )
        following.button(
            "Sau →",
            width="stretch",
            on_click=move_event,
            args=(event_key, events, 1),
        )
        near_events = int((mask & stage_transition_mask(record.labels)).sum())
        if filter_name == "N3 → N2 của E3":
            event_caption = f"{len(events)} lỗi được tìm thấy · {near_events} nằm trong vùng ±2 epoch quanh chuyển pha thật."
        else:
            event_caption = f"{len(events)} epoch phù hợp bộ lọc · {near_events} nằm gần chuyển pha thật."
        st.caption(
            f"Sự kiện {events.index(selected_event) + 1}/{len(events)} · {event_caption}"
        )
    else:
        st.info(f"{record.record_key} không có sự kiện thuộc bộ lọc “{filter_name}”.")

    st.markdown(
        record_insight_html(record, summary, filter_name, len(events)),
        unsafe_allow_html=True,
    )

    if view_mode == "Toàn bộ đêm" or selected_event is None:
        start, stop = 0, len(record.x)
    else:
        half_window = 20
        start = max(0, selected_event - half_window)
        stop = min(len(record.x), selected_event + half_window + 1)
    if view_mode == "Toàn bộ đêm":
        st.caption(f"Timeline toàn bộ bản ghi · {len(record.x):,} epoch · mỗi ô tương ứng 30 giây.")
    elif selected_event is not None:
        st.caption(
            f"Timeline tập trung quanh epoch gốc #{int(record.original_epoch_index[selected_event])} · hiển thị {stop - start} epoch."
        )
    frame = hypnogram_frame(record, predictions, start, stop)
    st.altair_chart(
        hypnogram_chart(frame, ["Chuyên gia", "E0", "E3", "E6"]),
        width="stretch",
        key=f"hypnogram_{record.record_key}_{start}_{stop}",
    )
    st.markdown(
        '<div class="note note-blue"><b>Cách đọc timeline.</b> Mỗi ô = 30 giây; màu ô = giai đoạn ngủ. Viền đỏ trên hàng E3 = lỗi N3→N2; viền cam = các mô hình bất đồng. Ô trống là epoch bị bỏ qua do không có nhãn. Rê chuột để xem giá trị từng epoch, kéo ngang để zoom.</div>',
        unsafe_allow_html=True,
    )

    if selected_event is not None:
        section(
            "02 · Kính hiển vi lỗi",
            f"Epoch tại phút {record.original_epoch_index[selected_event] * .5:g} · 30 giây tín hiệu",
            "Đối chiếu nhãn chuyên gia, dự đoán của ba mô hình và confidence trước khi đọc phần diễn giải.",
        )
        st.markdown(
            event_decision_html(record, predictions, selected_event),
            unsafe_allow_html=True,
        )
        signal_column, decision_column = st.columns([1.25, .75])
        with signal_column:
            st.altair_chart(
                eeg_chart(record, selected_event),
                width="stretch",
                key=f"eeg_{record.record_key}_{selected_event}",
            )
            st.caption("Tín hiệu EEG của epoch đang chọn, quy đổi gần µV để dễ đọc; kéo ngang để phóng đại hình dạng sóng.")
        with decision_column:
            st.markdown(confidence_html(record, predictions, selected_event), unsafe_allow_html=True)
        st.markdown(
            f'<div class="note note-green"><b>Diễn giải từ nhãn và dự đoán.</b> {event_explanation(record, predictions, selected_event)}</div>',
            unsafe_allow_html=True,
        )

    section(
        "03 · Đặt ví dụ vào đúng bối cảnh",
        "So sánh cả record, không chỉ một epoch",
        "Bảng này cho biết ví dụ đang xem có đại diện cho record hay chỉ là một điểm lỗi cục bộ. Kết luận khóa luận vẫn dựa trên 10-fold và so sánh bắt cặp theo đối tượng.",
    )
    st.markdown('<div class="table-heading">Chỉ số theo record đang chọn</div>', unsafe_allow_html=True)
    display_summary = summary.rename(
        columns={
            "Mô hình": "Mô hình",
            "Macro-F1": "Macro-F1",
            "Accuracy": "Accuracy",
            "N3 → N2": "Lỗi N3→N2",
            "Gần chuyển pha": "Gần chuyển pha",
        }
    )
    st.dataframe(
        display_summary.style.format({"Macro-F1": "{:.3f}", "Accuracy": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "E0 = 15CNN + BiLSTM · tín hiệu thô  |  E3 = ResNet-1D + TCN · lọc + scale  |  E6 = ResNet-1D + TCN · z-score"
    )
    st.caption(
        "Gần chuyển pha = trong phạm vi ±2 epoch quanh một thay đổi nhãn thật. Macro-F1 và Accuracy ở đây chỉ mô tả record đang chọn, không phải kết quả 10-fold."
    )
    with st.expander("Xem bảng chỉ số, provenance và tải CSV"):
        manifest = cached_asset_manifest()
        st.caption(
            f"Prediction artifact: {manifest['source_ref']} · fold {manifest['outer_fold']} · seed {manifest['seed']} · SHA-256 đã xác minh."
        )
        st.download_button(
            "Tải CSV so sánh E0/E3/E6",
            prediction_frame(record, predictions).to_csv(index=False).encode("utf-8"),
            file_name=f"{record.record_key}_model_comparison.csv",
            mime="text/csv",
            width="stretch",
        )


def qc_card(title: str, value: str, passed: bool) -> str:
    status = "✓" if passed else "✕"
    css = "qc-ok" if passed else "qc-bad"
    return f'<div class="qc-card"><div class="qc-title">{title}</div><div class="qc-value {css}">{status} {value}</div></div>'


def distribution_chart(predictions: dict[str, DemoPrediction]) -> alt.Chart:
    rows = []
    for experiment_id, prediction in predictions.items():
        counts = np.bincount(prediction.predicted, minlength=5)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "model": experiment_id,
                    "stage": STAGE_NAMES[index],
                    "fraction": float(count / counts.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("fraction:Q", title="Tỷ lệ epoch", axis=alt.Axis(format="%"), stack="normalize"),
            y=alt.Y("model:N", title=None, sort=list(predictions)),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())),
                legend=alt.Legend(title="Giai đoạn", orient="top"),
            ),
            tooltip=["model:N", "stage:N", alt.Tooltip("fraction:Q", format=".1%")],
        )
        .properties(height=max(90, 46 * len(predictions)))
        .configure_view(stroke=None)
    )


def render_upload_analysis() -> None:
    st.markdown(
        '<div class="note note-amber"><b>Luồng khám phá.</b> EDF mới không có hypnogram chuyên gia. Demo có thể hiển thị dự đoán và bất đồng mô hình, nhưng không thể gọi một dự đoán là đúng/sai hoặc tính Macro-F1.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Tải EDF PSG",
        type=["edf", "rec"],
        help="Yêu cầu EEG Fpz-Cz · 100 Hz · đơn vị µV.",
    )
    if uploaded is None:
        st.info("Chọn một file EDF để bắt đầu kiểm tra đầu vào. Dữ liệu chỉ được xử lý trên máy đang chạy demo.")
        return
    payload = uploaded.getvalue()
    inspection = cached_edf_inspection(payload, uploaded.name)
    section("Bước 1", "Kiểm tra đầu vào", "Pipeline chỉ được chạy sau khi bốn điều kiện cơ bản đều đạt.")
    sampling = inspection["sampling_rate_hz"]
    unit = inspection["physical_dimension"] or "Không xác định"
    st.markdown(
        '<div class="qc-grid">'
        + qc_card("Kênh", "EEG Fpz-Cz", inspection["has_required_channel"])
        + qc_card("Tần số", f"{sampling:g} Hz" if sampling is not None else "Không đọc được", inspection["sampling_rate_is_100hz"])
        + qc_card("Đơn vị", str(unit), inspection["unit_is_uv"])
        + qc_card("Epoch hoàn chỉnh", f"{inspection['complete_epochs']:,}", inspection["has_complete_epoch"])
        + "</div>",
        unsafe_allow_html=True,
    )
    if not inspection["ready"]:
        st.error("EDF chưa đạt hợp đồng đầu vào. Phiên bản demo hiện không tự đổi kênh hoặc resample tín hiệu.")
        return
    if inspection["trailing_samples"]:
        st.warning(f"{inspection['trailing_samples']:,} sample cuối không tạo đủ epoch 30 giây và sẽ bị bỏ.")

    section("Bước 2", "Chọn mức phân tích", "E3 là lựa chọn mặc định; chỉ nạp thêm E0/E6 khi cần so sánh.")
    mode = st.segmented_control(
        "Chế độ suy luận",
        ["E3 · Khuyến nghị", "So sánh E0 / E3 / E6"],
        default="E3 · Khuyến nghị",
        width="stretch",
    )
    selected_models = ["E3"] if mode == "E3 · Khuyến nghị" else list(DEMO_EXPERIMENTS)
    with st.expander("Thiết lập nâng cao"):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        device = st.selectbox(
            "Thiết bị suy luận",
            devices,
            format_func=lambda value: "GPU CUDA" if value == "cuda" else "CPU",
        )
        st.caption("Mặc định CPU để demo có thể chạy trên máy không có GPU.")
    digest = hashlib.sha256(payload).hexdigest()
    run_key = f"{digest}:{','.join(selected_models)}:{device}"
    if st.button("Chạy phân tích", type="primary", width="stretch"):
        progress = st.progress(0, text="Đang tiền xử lý EDF…")
        try:
            records = cached_uploaded_records(payload, uploaded.name)
            results: dict[str, DemoPrediction] = {}
            for index, experiment_id in enumerate(selected_models, start=1):
                progress.progress(
                    (index - 1) / len(selected_models),
                    text=f"Đang chạy {MODEL_LABELS[experiment_id]}…",
                )
                results[experiment_id] = predict_record(
                    records[experiment_id], cached_models(experiment_id, device)
                )
            st.session_state["edf_result"] = (run_key, records, results)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Không thể hoàn tất suy luận: {error}")
        finally:
            progress.empty()
    stored = st.session_state.get("edf_result")
    if not stored or stored[0] != run_key:
        st.caption("Nhấn **Chạy phân tích** để tiền xử lý và nạp checkpoint.")
        return
    records: dict[str, DemoRecord] = stored[1]
    predictions: dict[str, DemoPrediction] = stored[2]
    reference = records[next(iter(predictions))]

    section("Bước 3", "Đọc kết quả", "Hypnogram là dự đoán; nền vàng chỉ vùng mô hình bất đồng, không phải lỗi đã xác nhận.")
    frame = hypnogram_frame(reference, predictions)
    st.altair_chart(
        hypnogram_chart(frame, list(predictions)),
        width="stretch",
        key=f"edf_hypnogram_{run_key}",
    )
    result_cards = st.columns(len(predictions))
    for column, (experiment_id, prediction) in zip(result_cards, predictions.items(), strict=True):
        column.metric(
            experiment_id,
            f"{prediction.elapsed_seconds:.2f} giây",
            f"{prediction.epochs_per_second:.1f} epoch/s",
        )
    st.altair_chart(distribution_chart(predictions), width="stretch")
    if len(predictions) > 1:
        stack = np.stack([prediction.predicted for prediction in predictions.values()])
        disagreements = int(np.any(stack != stack[0], axis=0).sum())
        st.markdown(
            f'<div class="note note-blue"><b>{disagreements:,} epoch bất đồng.</b> Đây là các vùng nên được chuyên gia hoặc dữ liệu có nhãn xem lại; demo không tự quyết định mô hình nào đúng.</div>',
            unsafe_allow_html=True,
        )
    e3_prediction = predictions.get("E3")
    if e3_prediction is not None:
        low_confidence = int((e3_prediction.probabilities.max(axis=1) < .6).sum())
        st.caption(f"E3 có {low_confidence:,}/{len(e3_prediction.predicted):,} epoch confidence dưới 0,60; softmax chưa được hiệu chuẩn lâm sàng.")
    st.download_button(
        "Tải kết quả CSV",
        prediction_frame(reference, predictions).to_csv(index=False).encode("utf-8"),
        file_name=f"{reference.record_key}_sleep_staging.csv",
        mime="text/csv",
        width="stretch",
    )


def render_analysis() -> None:
    st.markdown(
        '<div class="section-kicker">Interactive analysis</div><div class="section-title">Từ một đêm ngủ đến một quyết định cụ thể</div><div class="section-copy">Chọn bản ghi đã có nhãn để phân tích lỗi, hoặc tải EDF mới để chạy suy luận khám phá.</div>',
        unsafe_allow_html=True,
    )
    source = st.segmented_control(
        "Nguồn dữ liệu",
        ["Bản ghi kiểm thử", "Tải EDF mới"],
        key="analysis_source",
        width="stretch",
    )
    if source == "Bản ghi kiểm thử":
        render_test_analysis()
    else:
        render_upload_analysis()
    st.markdown(
        '<div class="note note-amber" style="margin-top:2rem"><b>Giới hạn sử dụng.</b> Đây là công cụ nghiên cứu và bảo vệ khóa luận, không phải thiết bị y tế hoặc hệ thống chẩn đoán.</div>',
        unsafe_allow_html=True,
    )


def performance_chart(performance: pd.DataFrame) -> alt.Chart:
    frame = performance[performance["experiment"].isin(DEMO_EXPERIMENTS)].copy()
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=5, size=32).encode(
        y=alt.Y("experiment:N", sort=list(DEMO_EXPERIMENTS), title=None),
        x=alt.X("macro_f1:Q", title="Macro-F1 (0–1)", scale=alt.Scale(domain=[0, .82])),
        color=alt.Color(
            "experiment:N",
            scale=alt.Scale(domain=list(DEMO_EXPERIMENTS), range=[MODEL_COLORS[key] for key in DEMO_EXPERIMENTS]),
            legend=None,
        ),
        tooltip=["experiment:N", alt.Tooltip("macro_f1:Q", format=".4f")],
    )
    text = bars.mark_text(align="left", baseline="middle", dx=5, color="#27343D").encode(
        text=alt.Text("macro_f1:Q", format=".4f")
    )
    return (bars + text).properties(height=190).configure_view(stroke=None)


def class_f1_chart(performance: pd.DataFrame) -> alt.Chart:
    row = performance.set_index("experiment").loc["E3"]
    frame = pd.DataFrame(
        {"stage": list(STAGE_NAMES), "f1": [row[f"f1_{stage}"] for stage in STAGE_NAMES]}
    )
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("stage:N", sort=list(STAGE_NAMES), title=None),
            y=alt.Y("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())),
                legend=None,
            ),
            tooltip=["stage:N", alt.Tooltip("f1:Q", format=".3f")],
        )
        .properties(height=240)
        .configure_view(stroke=None)
    )


def runtime_chart() -> alt.Chart:
    frame = pd.DataFrame(
        {
            "experiment": ["E0", "E3"],
            "hours": [33 + 35 / 60 + 36 / 3600, 3 + 16 / 60 + 40 / 3600],
            "label": ["33h 35m 36s", "3h 16m 40s"],
        }
    )
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=5, size=38).encode(
        y=alt.Y("experiment:N", sort=["E0", "E3"], title=None),
        x=alt.X("hours:Q", title="Giờ · 10 fold huấn luyện + validation"),
        color=alt.Color(
            "experiment:N",
            scale=alt.Scale(domain=["E0", "E3"], range=[MODEL_COLORS["E0"], MODEL_COLORS["E3"]]),
            legend=None,
        ),
        tooltip=["experiment:N", alt.Tooltip("label:N", title="Wall-clock")],
    )
    labels = bars.mark_text(align="left", baseline="middle", dx=5, color="#27343D").encode(
        text=alt.Text("label:N")
    )
    return (bars + labels).properties(height=150).configure_view(stroke=None)


def render_evidence() -> None:
    section(
        "Locked thesis evidence",
        "Ba kết luận có thể bảo vệ",
        "Mỗi phần chỉ trả lời một câu hỏi; bảng đầy đủ được đặt trong mục chi tiết.",
    )
    performance = load_locked_table("table_performance.csv")
    complexity = load_locked_table("table_complexity_speed.csv")
    comparisons = load_locked_table("table_statistical_comparisons.csv")

    section("Kết luận 1 · Hiệu năng", "E3 đứng đầu về mô tả, nhưng không thắng mọi đối tượng", "So sánh E0/E3/E6 trên toàn chiến dịch 10-fold.")
    chart_column, text_column = st.columns([1.05, .95])
    with chart_column:
        st.altair_chart(performance_chart(performance), width="stretch")
    with text_column:
        st.markdown(
            '<div class="note note-green"><b>E3 đạt Macro-F1 0,7904.</b> E3 cao hơn E6 0,0213 điểm với CI 95% [0,0122; 0,0307] và p Holm 0,0012. Tuy nhiên, E3−E2 không đồng đều theo đối tượng nên không nên tuyên bố mọi thay đổi riêng lẻ đều vượt trội.</div>',
            unsafe_allow_html=True,
        )

    section("Kết luận 2 · Lỗi còn lại", "N1 và ranh giới N2–N3 vẫn là điểm yếu", "F1 theo lớp của E3 cho thấy hiệu năng tổng hợp che khuất khác biệt lớn giữa các giai đoạn.")
    chart_column, text_column = st.columns([1.05, .95])
    with chart_column:
        st.altair_chart(class_f1_chart(performance), width="stretch")
    with text_column:
        st.markdown(
            '<div class="note note-blue"><b>N1 là lớp khó nhất.</b> Error Explorer bổ sung điều mà bảng tổng hợp không thể hiện: vị trí cụ thể nơi N3 bị kéo về N2 và mức độ lỗi tập trung quanh chuyển pha trên từng record.</div>',
            unsafe_allow_html=True,
        )

    section("Kết luận 3 · Vận hành", "Pipeline mới rút ngắn vòng phản hồi thực nghiệm", "Thời gian wall-clock được đo trên cùng GPU V100 trong chiến dịch seed 123.")
    chart_column, text_column = st.columns([1.05, .95])
    with chart_column:
        st.altair_chart(runtime_chart(), width="stretch")
    with text_column:
        values = complexity.set_index("experiment")
        st.metric("Thời gian huấn luyện E3", "3h 16m 40s", "−90,2% so với E0", delta_color="off")
        st.metric("Tốc độ suy luận E3", f"{values.loc['E3', 'latency_ms_median']:.3f} ms", "3,76× nhanh hơn E0", delta_color="off")
        st.caption("Đổi lại, E3 có 4,37× số tham số và peak VRAM cao hơn 28,4%.")
    st.markdown(
        '<div class="note note-amber"><b>Ranh giới kết luận.</b> Đây là lợi ích vận hành đo được trong một giao thức và phần cứng cụ thể; không phải hằng số tốc độ phổ quát và không chứng minh ưu thế lâm sàng.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Xem bảng số liệu và kiểm định đầy đủ"):
        tab_a, tab_b, tab_c = st.tabs(["Hiệu năng", "Thống kê", "Vận hành"])
        with tab_a:
            st.dataframe(performance, width="stretch", hide_index=True)
        with tab_b:
            st.dataframe(comparisons, width="stretch", hide_index=True)
        with tab_c:
            st.dataframe(complexity, width="stretch", hide_index=True)


brand()
render_sleep_record_explorer()
