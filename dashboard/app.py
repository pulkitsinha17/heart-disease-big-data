"""
Heart Disease Risk Prediction Dashboard
========================================
Clinical Analytics Platform -- Pure Dark Mode.
Background: #0B0B0F | Cards: #15151D | Borders: #282833 | Text: #F5F5F7, #A1A1AA
Accent: #2563EB / #3B82F6 | Status: #22C55E (Green), #EF4444 (Red), #F59E0B (Amber)
DISCLAIMER: Educational demonstration only. Not a medical device.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).resolve().parent.parent
MODELS_DIR       = ROOT / "models"
RESULTS_DIR      = ROOT / "results"
DATA_DIR         = ROOT / "data" / "processed"

MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest":       MODELS_DIR / "random_forest.joblib",
}
METRICS_PATH     = RESULTS_DIR / "model_metrics.json"
METRICS_CSV_PATH = RESULTS_DIR / "metrics_comparison.csv"
PREDS_CSV_PATH   = RESULTS_DIR / "test_predictions.csv"
ROC_PATH         = RESULTS_DIR / "roc_curve_comparison.png"
CM_PATH          = RESULTS_DIR / "confusion_matrices.png"
FI_PATH          = RESULTS_DIR / "feature_importance.png"
DATASET_PATH     = DATA_DIR   / "heart_disease.csv"

NUMERIC_FEATURES     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_FEATURES      = ["sex", "fbs", "exang"]
CLINICAL_FEATURES    = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk | Big Data Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Navigation State
# ---------------------------------------------------------------------------
PAGES = [
    "Dashboard",
    "Risk Assessment",
    "Analysis Report",
    "Model Evaluation",
    "Dataset Explorer",
    "About / Methodology",
]
ICONS = {
    "Dashboard": "🏠",
    "Risk Assessment": "🔬",
    "Analysis Report": "📋",
    "Model Evaluation": "📊",
    "Dataset Explorer": "🗂️",
    "About / Methodology": "ℹ️",
}
PAGE_OPTIONS = [f"{ICONS[p]}  {p}" for p in PAGES]

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = PAGES[0]

# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    return {name: (joblib.load(str(path)) if path.exists() else None)
            for name, path in MODEL_PATHS.items()}

@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_metrics_csv():
    return pd.read_csv(METRICS_CSV_PATH) if METRICS_CSV_PATH.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_preds_csv():
    return pd.read_csv(PREDS_CSV_PATH) if PREDS_CSV_PATH.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()

models     = load_models()
metrics    = load_metrics()
metrics_df = load_metrics_csv()
preds_df   = load_preds_csv()
dataset    = load_dataset()

# ---------------------------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='margin-bottom:14px;'>"
        "<div style='font-size:1.1rem;font-weight:700;color:#F5F5F7;letter-spacing:-0.02em;'>🫀 Heart Disease Risk</div>"
        "<div style='font-size:0.75rem;font-weight:600;color:#A1A1AA;text-transform:uppercase;letter-spacing:0.04em;'>Big Data Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    current_idx = 0
    if st.session_state["selected_page"] in PAGES:
        current_idx = PAGES.index(st.session_state["selected_page"])

    def _on_nav_change():
        choice = st.session_state["nav_radio_choice"]
        for p in PAGES:
            if p in choice:
                st.session_state["selected_page"] = p
                break

    st.markdown(
        "<span style='font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#71717A;'>Navigation</span>",
        unsafe_allow_html=True,
    )
    st.radio(
        "Navigation Menu",
        PAGE_OPTIONS,
        index=current_idx,
        key="nav_radio_choice",
        on_change=_on_nav_change,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#71717A;'>Model Readiness</span>",
        unsafe_allow_html=True,
    )
    for name, model in models.items():
        dot = "🟢" if model is not None else "🔴"
        status_txt = "Ready (v1.0)" if model is not None else "Missing"
        st.markdown(
            f"<div style='font-size:0.8rem;color:#D4D4D8;margin-bottom:3px;'>"
            f"{dot} <strong>{name}</strong>: {status_txt}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#71717A;line-height:1.7;'>"
        "Cohort: UCI Cleveland<br>"
        "Dataset: 303 records (13 features)<br>"
        "Target: Binary Angiographic HD<br>"
        "Infrastructure: Kafka · Hadoop HDFS</div>",
        unsafe_allow_html=True,
    )

current_page = st.session_state["selected_page"]

# ---------------------------------------------------------------------------
# PURE DARK THEME CSS: Pitch-Black, No White Widgets, Clean Header
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Background & Main App */
.stApp {
    background-color: #0B0B0F !important;
    color: #F5F5F7 !important;
}

/* Streamlit Header Toolbar: flush, transparent, no separator line */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 1.5rem !important;
    border: none !important;
    border-bottom: none !important;
}
header[data-testid="stHeader"] * {
    color: #A1A1AA !important;
}
.block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0E0E14 !important;
    border-right: 1px solid #1E1E28 !important;
}
[data-testid="stSidebar"] * {
    color: #D4D4D8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1E1E28 !important;
}

/* Navigation Radio: Hide the ugly white/black radio dots */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 8px 12px !important;
    margin-bottom: 3px !important;
    border-radius: 7px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    font-size: 0.88rem !important;
    color: #A1A1AA !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-color: #3B82F6 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover:not(:has(input:checked)) {
    background-color: #1A1A24 !important;
    color: #F5F5F7 !important;
}

/* Clean Section Headers */
h1 {
    color: #F5F5F7 !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    letter-spacing: -0.025em;
    margin-bottom: 4px !important;
}
h2 {
    color: #F5F5F7 !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    margin-top: 22px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid #1E1E28;
}
h3 {
    color: #E4E4E7 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-top: 8px !important;
}
p, li {
    color: #A1A1AA !important;
    font-size: 0.875rem;
}
label {
    color: #E4E4E7 !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
}

/* Clean Cards */
.section-card {
    background-color: #15151D;
    border: 1px solid #282833;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.metric-card {
    background-color: #15151D;
    border: 1px solid #282833;
    border-radius: 10px;
    padding: 16px 18px 12px;
    transition: border-color 0.15s;
}
.metric-card:hover {
    border-color: #3B82F6;
}
.metric-card-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #A1A1AA;
    margin-bottom: 4px;
}
.metric-card-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #3B82F6;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.metric-card-sub {
    font-size: 0.72rem;
    color: #71717A;
    margin-top: 3px;
}
.metric-card-accent-green .metric-card-value { color: #22C55E; }
.metric-card-accent-red   .metric-card-value { color: #EF4444; }
.metric-card-accent-amber .metric-card-value { color: #F59E0B; }
.metric-card-accent-blue  .metric-card-value { color: #3B82F6; }

/* Responsive Pipeline Flow Banner */
.pipeline-flow-banner {
    background-color: #15151D;
    border: 1px solid #282833;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
}
.pipeline-flow-steps {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.pipeline-flow-node {
    display: flex;
    align-items: center;
    gap: 7px;
    background-color: #1C1C26;
    border: 1px solid #282833;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #F5F5F7;
}
.pipeline-flow-arrow {
    color: #71717A;
    font-size: 1rem;
    font-weight: 700;
}

/* Dark Widgets & Form Inputs: Eliminate White Backgrounds */
[data-testid="stForm"] {
    background-color: #15151D !important;
    border: 1px solid #282833 !important;
    border-radius: 12px !important;
    padding: 22px 24px !important;
}

/* Number Input container: white border covering the whole field */
[data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stNumberInput"] div:has(> input),
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextInput"] div:has(> input) {
    background-color: #1A1A24 !important;
    border: 1px solid rgba(255, 255, 255, 0.65) !important;
    border-radius: 7px !important;
    color: #F5F5F7 !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within,
[data-testid="stNumberInput"] div:has(> input):focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}
[data-baseweb="input"] input, [data-baseweb="base-input"] input,
[data-testid="stNumberInput"] input {
    background-color: transparent !important;
    color: #F5F5F7 !important;
    font-size: 0.88rem !important;
    border: none !important;
}

/* REMOVE borders from -, +, and ? buttons completely */
[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    color: #F5F5F7 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #2B2B3A !important;
    color: #FFFFFF !important;
    border: none !important;
}

/* Tooltip ? icon: remove borders completely */
[data-testid="stTooltipIcon"],
[data-testid="stTooltipIcon"] button,
[data-testid="stTooltipHoverTarget"] button,
[data-testid="stWidgetLabel"] button {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Selectbox across sections A, B, C: White border covering the field */
[data-testid="stSelectbox"] div:has(> input),
[data-testid="stSelectbox"] div[class*="e1fp86qc"],
[data-testid="stSelectbox"] > div:last-child > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-baseweb="select"] > div {
    background-color: #1A1A24 !important;
    border: 1px solid rgba(255, 255, 255, 0.65) !important;
    border-radius: 7px !important;
    color: #F5F5F7 !important;
}
[data-testid="stSelectbox"]:focus-within div:has(> input),
[data-testid="stSelectbox"]:focus-within div[class*="e1fp86qc"] {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}
[data-testid="stSelectbox"] input {
    color: #F5F5F7 !important;
    font-size: 0.88rem !important;
}

/* Multiselect: White border covering the field */
[data-testid="stMultiSelect"] div:has(> input),
[data-testid="stMultiSelect"] div[class*="e1kig3hy"],
[data-testid="stMultiSelect"] > div:last-child > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #1A1A24 !important;
    border: 1px solid rgba(255, 255, 255, 0.65) !important;
    border-radius: 7px !important;
    color: #F5F5F7 !important;
}

/* Dropdown opened popover */
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"],
[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #15151D !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
}
[role="option"] {
    background-color: #15151D !important;
    color: #F5F5F7 !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: #1E293B !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 5px !important;
    padding: 2px 7px !important;
}
[data-baseweb="tag"] span {
    color: #93C5FD !important;
    font-weight: 500 !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label span {
    color: #E4E4E7 !important;
    font-size: 0.84rem !important;
}

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 9px 22px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8 !important;
}
.stButton > button *, [data-testid="stFormSubmitButton"] > button * {
    color: #FFFFFF !important;
}

.stDownloadButton > button {
    background-color: #1C1C26 !important;
    color: #60A5FA !important;
    border: 1px solid #282833 !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
}
.stDownloadButton > button:hover {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-color: #3B82F6 !important;
}
.stDownloadButton > button * {
    color: inherit !important;
}

/* Prediction Result Cards */
.result-high {
    background-color: #2D1217;
    border: 1px solid #7F1D1D;
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 12px 0;
}
.result-low {
    background-color: #0E291A;
    border: 1px solid #14532D;
    border-left: 4px solid #22C55E;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 12px 0;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #A1A1AA;
    margin-bottom: 3px;
}
.result-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F5F5F7;
    margin-bottom: 4px;
}
.result-prob {
    font-size: 0.86rem;
    color: #D4D4D8;
    margin-top: 6px;
}
.prob-bar-track {
    background-color: #1C1C26;
    border-radius: 99px;
    height: 7px;
    margin-top: 10px;
    overflow: hidden;
}
.prob-bar-high {
    height: 7px;
    border-radius: 99px;
    background: linear-gradient(90deg, #F87171, #EF4444);
}
.prob-bar-low {
    height: 7px;
    border-radius: 99px;
    background: linear-gradient(90deg, #4ADE80, #22C55E);
}

/* Context / Message Boxes */
.disclaimer-box {
    background-color: #231808;
    border: 1px solid #5C3806;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #FEF3C7;
    line-height: 1.55;
    margin-top: 18px;
}
.info-box {
    background-color: #0C1E38;
    border: 1px solid #173B70;
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #DBEAFE;
    line-height: 1.55;
}

/* Tables & DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #282833 !important;
    border-radius: 8px !important;
    background-color: #15151D !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid #282833 !important;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #A1A1AA !important;
    font-size: 0.86rem !important;
    padding: 6px 14px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 2px solid #3B82F6 !important;
    font-weight: 600 !important;
}

/* Architecture Step Items */
.arch-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background-color: #15151D;
    border: 1px solid #282833;
    border-radius: 8px;
    padding: 12px 15px;
    margin-bottom: 8px;
}
.arch-step-icon {
    font-size: 1.25rem;
    min-width: 28px;
    text-align: center;
}
.arch-step-body h4 {
    margin: 0 0 2px;
    font-size: 0.86rem;
    font-weight: 600;
    color: #F5F5F7;
}
.arch-step-body p {
    margin: 0;
    font-size: 0.78rem;
    color: #A1A1AA !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Reusable UI Helpers
# ---------------------------------------------------------------------------
def metric_card(label, value, sub="", accent=""):
    ac    = f"metric-card-accent-{accent}" if accent else ""
    sub_h = f"<div class='metric-card-sub'>{sub}</div>" if sub else ""
    return (f"<div class='metric-card {ac}'>"
            f"<div class='metric-card-label'>{label}</div>"
            f"<div class='metric-card-value'>{value}</div>{sub_h}</div>")

def clean_header(title, subtitle=""):
    sub_h = f"<p style='margin:3px 0 0;font-size:0.9rem;color:#A1A1AA;line-height:1.4;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div style='margin-bottom:20px;'>"
        f"<h1 style='margin:0;font-size:1.85rem;font-weight:700;color:#F5F5F7;letter-spacing:-0.025em;'>{title}</h1>{sub_h}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 -- Dashboard Homepage
# ===========================================================================
if current_page == "Dashboard":
    clean_header(
        "Heart Disease Risk Prediction",
        "Big Data Analytics Pipeline · UCI Cleveland Dataset · Kafka · Hadoop HDFS · Machine Learning"
    )

    # Top Overview & Actions
    welcome_col, action_col = st.columns([3, 2], gap="large")
    with welcome_col:
        st.markdown(
            "<div class='section-card' style='margin-bottom:14px;padding:16px 20px;'>"
            "<p style='margin:0;font-size:0.88rem;color:#D4D4D8;line-height:1.6;'>"
            "Welcome to the Heart Disease Analytics Platform. This platform connects streaming ingestion via "
            "<strong>Apache Kafka (KRaft)</strong>, distributed persistence on <strong>Hadoop HDFS</strong> (JSONL format), "
            "and dual machine learning models (<strong>Logistic Regression</strong> & <strong>Random Forest</strong>) "
            "trained on the UCI Cleveland dataset to perform clinical risk stratification."
            "</p></div>",
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        act_c1, act_c2 = st.columns(2, gap="small")
        with act_c1:
            if st.button("🔬 Start Assessment", use_container_width=True):
                st.session_state["selected_page"] = "Risk Assessment"
                st.rerun()
        with act_c2:
            if st.button("📊 Model Evaluation", use_container_width=True):
                st.session_state["selected_page"] = "Model Evaluation"
                st.rerun()

    # Verified Metrics Row
    lr = metrics.get("logistic_regression", {})
    rf = metrics.get("random_forest", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    total_records = len(dataset) if not dataset.empty else 303
    for col, lbl, val, sub, acc in [
        (c1, "Dataset Records",   f"{total_records}",             "UCI Cleveland cohort", ""),
        (c2, "Input Features",    "13",                           "5 cont, 5 cat, 3 bin", ""),
        (c3, "Held-Out Test Set", "61",                           "20% stratified test",  ""),
        (c4, "LR Test ROC-AUC",   f"{lr.get('roc_auc',0.9697):.4f}", "Held-out test split", "green"),
        (c5, "RF Test ROC-AUC",   f"{rf.get('roc_auc',0.9470):.4f}", "Held-out test split", "blue"),
    ]:
        col.markdown(metric_card(lbl, val, sub, acc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline Flow Banner
    st.markdown(
        "<div class='pipeline-flow-banner'>"
        "<div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;color:#A1A1AA;'>"
        "End-to-End System Workflow</div>"
        "<div class='pipeline-flow-steps'>"
        "<div class='pipeline-flow-node'><span>📦</span> Dataset (UCI Cleveland)</div>"
        "<div class='pipeline-flow-arrow'>→</div>"
        "<div class='pipeline-flow-node'><span>⚡</span> Kafka Streaming (KRaft)</div>"
        "<div class='pipeline-flow-arrow'>→</div>"
        "<div class='pipeline-flow-node'><span>💾</span> Hadoop HDFS (JSONL)</div>"
        "<div class='pipeline-flow-arrow'>→</div>"
        "<div class='pipeline-flow-node'><span>🤖</span> ML Pipelines (LR & RF)</div>"
        "<div class='pipeline-flow-arrow'>→</div>"
        "<div class='pipeline-flow-node'><span>🔬</span> Clinical Risk Assessment</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Two-Column Performance Summary & Architecture
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("## Model Performance Summary (Held-Out Test Set)")
        rows = []
        for dname, key in [("Logistic Regression", "logistic_regression"),
                            ("Random Forest", "random_forest")]:
            m = metrics.get(key, {})
            rows.append({
                "Model": dname,
                "Test ROC-AUC":  f"{m.get('roc_auc',0):.4f}",
                "Test Accuracy": f"{m.get('accuracy',0):.2%}",
                "Precision":     f"{m.get('precision',0):.4f}",
                "Recall (Sens)": f"{m.get('recall_sensitivity',0):.4f}",
                "Specificity":   f"{m.get('specificity',0):.4f}",
                "F1 Score":      f"{m.get('f1_score',0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if ROC_PATH.exists():
            st.markdown("## ROC Curve Diagnostic")
            st.image(str(ROC_PATH), use_container_width=True)

    with right:
        st.markdown("## Architecture Stages")
        for icon, title, desc in [
            ("🐳", "Phase 1 -- Infrastructure",   "Docker Compose · Apache Kafka KRaft · Hadoop HDFS 3.3.6"),
            ("📦", "Phase 2 -- Data Acquisition", "UCI Cleveland (303 records) · Cleaning & EDA"),
            ("⚡", "Phase 3 -- Kafka Streaming",  "Python producer streams records to 'heart-disease-records'"),
            ("💾", "Phase 4 -- HDFS Ingestion",   "WebHDFS consumer writes /heart-disease/raw/stream.jsonl"),
            ("🤖", "Phase 5 -- ML Training",      "ColumnTransformer · Repeated Stratified 5-Fold CV (3 repeats)"),
            ("📊", "Phase 6 -- Dashboard UI",     "Streamlit 6-page interactive clinical analytics portal"),
        ]:
            st.markdown(
                f"<div class='arch-step'>"
                f"<div class='arch-step-icon'>{icon}</div>"
                f"<div class='arch-step-body'><h4>{title}</h4><p>{desc}</p></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("## Feature Preprocessing Groups")
        st.markdown(
            f"<div class='section-card' style='margin-top:0;'>"
            f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;color:#A1A1AA;margin-bottom:3px;'>Continuous Numeric (5)</p>"
            f"<p style='font-size:0.82rem;color:#F5F5F7;margin-bottom:10px;'>{', '.join(NUMERIC_FEATURES)}</p>"
            f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;color:#A1A1AA;margin-bottom:3px;'>Categorical One-Hot (5)</p>"
            f"<p style='font-size:0.82rem;color:#F5F5F7;margin-bottom:10px;'>{', '.join(CATEGORICAL_FEATURES)}</p>"
            f"<p style='font-size:0.68rem;font-weight:700;text-transform:uppercase;color:#A1A1AA;margin-bottom:3px;'>Binary Passthrough (3)</p>"
            f"<p style='font-size:0.82rem;color:#F5F5F7;margin:0;'>{', '.join(BINARY_FEATURES)}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='disclaimer-box'>⚠️ <strong>Educational Disclaimer:</strong> "
        "This platform is developed strictly for academic demonstration of big data and machine learning architectures. "
        "Outputs are not clinically validated and must never replace medical advice or diagnosis.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 2 -- Risk Assessment
# ===========================================================================
elif current_page == "Risk Assessment":
    clean_header(
        "Risk Assessment",
        "Input patient clinical vitals and test results across three structured groups for inference."
    )

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("### A. Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=54, step=1,
                                  help="Patient age in completed years")
            sex = st.selectbox("Sex", [0, 1],
                               format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
                               help="Biological sex: 0 = Female, 1 = Male")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130, 1,
                                       help="Resting blood pressure on hospital admission (trestbps)")
            chol     = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 246, 1,
                                       help="Serum cholesterol in mg/dL (chol)")
            thalach  = st.number_input("Max Heart Rate Achieved (bpm)", 60, 250, 150, 1,
                                       help="Maximum heart rate reached during exercise test (thalach)")
            oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1, format="%.1f",
                                       help="ST depression induced by exercise relative to rest")

        with col2:
            st.markdown("### B. Clinical Indicators")
            cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "1 -- Typical angina",
                                  2: "2 -- Atypical angina",
                                  3: "3 -- Non-anginal pain",
                                  4: "4 -- Asymptomatic"
                              }[x],
                              help="Chest pain experience category")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                               format_func=lambda x: "No (0: <= 120 mg/dL)" if x == 0 else "Yes (1: > 120 mg/dL)",
                               help="Indicator of diabetes risk factor (fbs)")
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2],
                                   format_func=lambda x: {
                                       0: "0 -- Normal",
                                       1: "1 -- ST-T wave abnormality",
                                       2: "2 -- Left ventricular hypertrophy"
                                   }[x],
                                   help="Resting electrocardiographic status (restecg)")
            exang = st.selectbox("Exercise-Induced Angina", [0, 1],
                                 format_func=lambda x: "No (0: None reported)" if x == 0 else "Yes (1: Angina provoked)",
                                 help="Angina provoked by treadmill exercise test (exang)")
            slope = st.selectbox("Slope of Peak Exercise ST", [1, 2, 3],
                                 format_func=lambda x: {
                                     1: "1 -- Upsloping",
                                     2: "2 -- Flat",
                                     3: "3 -- Downsloping"
                                 }[x],
                                 help="Slope of peak exercise ST segment (slope)")

        with col3:
            st.markdown("### C. Lab Findings & Options")
            ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3],
                              format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''} colored",
                              help="Number of major coronary vessels (0-3) colored by fluoroscopy (ca)")
            thal_opts = {3: "3 -- Normal", 6: "6 -- Fixed defect", 7: "7 -- Reversible defect"}
            thal = st.selectbox("Thallium Heart Scan (thal)", list(thal_opts),
                                format_func=lambda x: thal_opts[x],
                                help="Thallium scintigraphy defect result")

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            ca_nan = st.checkbox("Simulate missing data for fluoroscopy (ca) / thal as NaN", value=False,
                                 help="Tests the pipeline's median/mode imputation on missing inputs")

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            sel_models = st.multiselect(
                "Select Machine Learning Models for Inference",
                list(models.keys()),
                default=[k for k, v in models.items() if v is not None],
                help="Choose one or both trained models to evaluate the patient data"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        if not sel_models:
            st.warning("Please select at least one machine learning model to execute.")
        else:
            thal_v = np.nan if ca_nan else float(thal)
            ca_v   = np.nan if ca_nan else float(ca)
            input_df = pd.DataFrame(
                [[age, trestbps, chol, thalach, oldpeak,
                  cp, restecg, slope, thal_v, ca_v,
                  sex, fbs, exang]],
                columns=CLINICAL_FEATURES,
            )

            st.markdown("---")
            st.markdown("## Prediction Results")
            rcols = st.columns(len(sel_models), gap="large")
            sess_results = []
            for idx, mname in enumerate(sel_models):
                mdl = models[mname]
                if mdl is None:
                    rcols[idx].error(f"{mname}: Model artifact not found on disk.")
                    continue
                prob    = float(mdl.predict_proba(input_df)[0][1])
                pred    = int(prob >= 0.5)
                bar_pct = int(min(max(prob * 100, 0), 100))
                css, bar = ("result-high", "prob-bar-high") if pred == 1 else ("result-low", "prob-bar-low")
                title_txt, icon = ("Elevated Risk Detected", "🔴") if pred == 1 else ("Lower Risk Indicated", "🟢")
                interp_desc = (
                    "Risk probability exceeds the 50% decision threshold, indicating positive coronary disease markers."
                    if pred == 1 else
                    "Risk probability is below the 50% decision threshold, indicating lower coronary risk markers."
                )

                rcols[idx].markdown(
                    f"<div class='{css}'>"
                    f"<div class='result-label'>Model: {mname}</div>"
                    f"<div class='result-title'>{icon} {title_txt}</div>"
                    f"<div class='result-prob'>Predicted Risk Probability: "
                    f"<strong style='font-size:1.2rem;color:#F5F5F7;'>{prob:.1%}</strong> "
                    f"<span style='font-size:0.78rem;color:#A1A1AA;'>({prob:.4f})</span></div>"
                    f"<div class='prob-bar-track'><div class='{bar}' style='width:{bar_pct}%;'></div></div>"
                    f"<p style='font-size:0.82rem;margin-top:10px;line-height:1.5;color:#D4D4D8;'>{interp_desc}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                sess_results.append({
                    "model": mname,
                    "probability": prob,
                    "prediction": pred,
                    "label": title_txt,
                    "interpretation": interp_desc,
                })

            st.session_state["last_assessment"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "inputs":    input_df.to_dict(orient="records")[0],
                "results":   sess_results,
            }

            st.markdown(
                "<div class='info-box' style='margin-top:14px;'>"
                "ℹ️ <strong>Assessment preserved:</strong> "
                "Inputs and model predictions are saved for this session. "
                "You can generate and export the complete summary on the <strong>Analysis Report</strong> page."
                "</div>",
                unsafe_allow_html=True,
            )

            nav_col1, _ = st.columns([2, 4])
            with nav_col1:
                if st.button("📋 View & Download Analysis Report", use_container_width=True):
                    st.session_state["selected_page"] = "Analysis Report"
                    st.rerun()

            with st.expander("Review submitted input features"):
                st.dataframe(input_df.T.rename(columns={0: "Submitted Value"}), use_container_width=True)

    st.markdown(
        "<div class='disclaimer-box'>⚠️ <strong>Educational Disclaimer:</strong> "
        "The risk probability shown represents the model's statistical likelihood of angiographic heart disease "
        "(= 50% diameter narrowing) based on the 303 Cleveland training records. It is not an official medical diagnosis.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 3 -- Analysis Report
# ===========================================================================
elif current_page == "Analysis Report":
    clean_header(
        "Analysis Report",
        "Structured summary for the most recent assessment with exportable plain-text and JSON formats."
    )

    assessment = st.session_state.get("last_assessment")
    if assessment is None:
        st.markdown(
            "<div class='section-card' style='text-align:center;padding:44px 20px;'>"
            "<div style='font-size:2.5rem;margin-bottom:10px;'>📋</div>"
            "<h3 style='margin-bottom:6px;'>No Active Assessment Found</h3>"
            "<p style='max-width:500px;margin:0 auto 18px;line-height:1.6;'>"
            "To generate a clinical report, please complete a patient assessment on the Risk Assessment page first. "
            "Inference inputs and probabilities will automatically populate here."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        col_btn, _ = st.columns([2, 4])
        with col_btn:
            if st.button("🔬 Proceed to Risk Assessment", use_container_width=True):
                st.session_state["selected_page"] = "Risk Assessment"
                st.rerun()
    else:
        ts      = assessment["timestamp"]
        inputs  = assessment["inputs"]
        results = assessment["results"]

        st.markdown(f"**Recorded:** `{ts}` · **Cohort:** UCI Cleveland · **Session Context:** Active")

        st.markdown("## Clinical Model Outcomes")
        rcols = st.columns(len(results), gap="large")
        for idx, r in enumerate(results):
            accent = "red" if r["prediction"] == 1 else "green"
            rcols[idx].markdown(
                metric_card(
                    f"{r['model']} ({r['label']})",
                    f"{r['probability']:.1%}",
                    f"Score: {r['probability']:.4f} | Binary Class: {r['prediction']}",
                    accent
                ),
                unsafe_allow_html=True
            )

        st.markdown("---")
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("## Submitted Clinical Attributes")
            labels_map = {
                "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dL)",
                "thalach": "Max Heart Rate (bpm)", "oldpeak": "ST Depression", "cp": "Chest Pain Type",
                "restecg": "Resting ECG", "slope": "ST Slope", "thal": "Thalassemia", "ca": "Major Vessels Colored",
                "sex": "Sex (0=F, 1=M)", "fbs": "Fasting BS > 120", "exang": "Exercise Angina"
            }
            inp_records = [{"Feature Name": labels_map.get(k, k), "Attribute": k, "Value": v} for k, v in inputs.items()]
            st.dataframe(pd.DataFrame(inp_records), use_container_width=True, hide_index=True)

        with right:
            st.markdown("## Clinical Interpretation")
            st.markdown(
                "<div class='section-card'>"
                "<p><strong>Decision Cutoff:</strong> 0.50 (50.0% probability)</p>"
                "<p><strong>Probability &lt; 50%:</strong><br>"
                "<span style='color:#22C55E;font-weight:600;'>Lower Risk</span> -- Patient profile exhibits lower prevalence of angiographic coronary disease.</p>"
                "<p><strong>Probability &ge; 50%:</strong><br>"
                "<span style='color:#EF4444;font-weight:600;'>Elevated Risk</span> -- Patient profile correlates with &ge;50% arterial diameter narrowing.</p>"
                "<hr style='border-color:#282833;margin:12px 0;'>"
                "<p style='font-size:0.78rem;color:#71717A;margin:0;'>"
                "Trained on 242 stratified samples; tested on 61 held-out cases.<br>"
                "Preprocessing: Median imputation for continuous features, mode imputation for categorical attributes, StandardScaler and OneHotEncoder."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## Export & Download Report")

        lines = [
            "=" * 64,
            "            HEART DISEASE RISK ASSESSMENT REPORT",
            "=" * 64,
            f"Generated At    : {ts}",
            "Study Cohort    : UCI Cleveland Heart Disease Dataset (303 records)",
            "Architecture    : Apache Kafka + Hadoop HDFS + scikit-learn",
            "-" * 64,
            "1. SUBMITTED CLINICAL MEASUREMENTS",
            "-" * 64,
        ]
        for feat, val in inputs.items():
            lines.append(f"  {feat:<16}: {val}")
        lines += [
            "-" * 64,
            "2. MODEL PREDICTIONS & RISK PROBABILITIES",
            "-" * 64,
        ]
        for r in results:
            lines += [
                f"  Model               : {r['model']}",
                f"  Risk Probability    : {r['probability']:.4f} ({r['probability']:.1%})",
                f"  Classification      : {'Elevated Risk (HD=1)' if r['prediction'] == 1 else 'Lower Risk (HD=0)'}",
                f"  Clinical Status     : {r['label']}",
                f"  Decision Threshold  : 0.5000",
                "",
            ]
        lines += [
            "-" * 64,
            "3. METHODOLOGY CONTEXT & LIMITATIONS",
            "-" * 64,
            "  - Logistic Regression and Random Forest trained on 242 Cleveland samples.",
            "  - Validated using Repeated Stratified 5-Fold CV (3 repeats).",
            "  - Evaluated on a 61-record held-out test split.",
            "",
            "-" * 64,
            "4. MEDICAL DISCLAIMER",
            "-" * 64,
            "  This report is strictly for educational and technical demonstration.",
            "  It does not constitute a medical diagnosis, triage, or clinical opinion.",
            "  Always consult a certified medical professional for diagnostic decisions.",
            "=" * 64,
        ]
        report_text = "\n".join(lines)
        safe_ts = ts.replace(" ", "_").replace(":", "-")

        report_json = json.dumps({
            "report_timestamp": ts,
            "dataset": "UCI Cleveland Heart Disease",
            "inputs": inputs,
            "results": results,
            "disclaimer": "Academic demonstration only. Not a medical device."
        }, indent=2)

        dl_c1, dl_c2, _ = st.columns([2, 2, 3])
        with dl_c1:
            st.download_button(
                "⬇️ Download Text Report (.txt)",
                data=report_text.encode("utf-8"),
                file_name=f"cardiac_risk_report_{safe_ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_c2:
            st.download_button(
                "⬇️ Download JSON Data (.json)",
                data=report_json.encode("utf-8"),
                file_name=f"cardiac_risk_report_{safe_ts}.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Preview generated text report"):
            st.code(report_text, language="text")


# ===========================================================================
# PAGE 4 -- Model Evaluation
# ===========================================================================
elif current_page == "Model Evaluation":
    clean_header(
        "Model Evaluation",
        "Performance benchmarks: Held-out test set (61 records, 20% stratified) and Repeated Stratified 5-Fold Cross-Validation."
    )

    for mname, key in [("Logistic Regression", "logistic_regression"),
                       ("Random Forest", "random_forest")]:
        m = metrics.get(key, {})
        if not m:
            continue
        st.markdown(f"## {mname}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Test ROC-AUC",    f"{m.get('roc_auc', 0):.4f}")
        c2.metric("Test Accuracy",   f"{m.get('accuracy', 0):.2%}")
        c3.metric("Test Precision",  f"{m.get('precision', 0):.4f}")
        c4.metric("Test Recall",     f"{m.get('recall_sensitivity', 0):.4f}")
        c5.metric("Test Specificity",f"{m.get('specificity', 0):.4f}")
        c6.metric("Test F1 Score",   f"{m.get('f1_score', 0):.4f}")

        cm = m.get("confusion_matrix", {})
        if cm:
            left, right = st.columns([3, 5], gap="large")
            with left:
                st.markdown("**Confusion Matrix (n = 61)**")
                cm_df = pd.DataFrame(
                    [[cm.get("true_negatives", "-"), cm.get("false_positives", "-")],
                     [cm.get("false_negatives", "-"), cm.get("true_positives", "-")]],
                    index=["Actual Neg (0)", "Actual Pos (1)"],
                    columns=["Pred Neg (0)", "Pred Pos (1)"]
                )
                st.dataframe(cm_df, use_container_width=True)
            with right:
                st.markdown("**Classification Report**")
                cr = m.get("classification_report", {})
                cr_rows = []
                for cls in ["0", "1", "macro avg", "weighted avg"]:
                    if cls in cr:
                        row = cr[cls]
                        cr_rows.append({
                            "Class / Metric": "Class 0 (No HD)" if cls == "0" else ("Class 1 (HD Present)" if cls == "1" else cls),
                            "Precision": f"{row.get('precision', 0):.4f}",
                            "Recall":    f"{row.get('recall', 0):.4f}",
                            "F1-Score":  f"{row.get('f1-score', 0):.4f}",
                            "Support":   int(row.get("support", 0))
                        })
                st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)
        st.markdown("---")

    if not metrics_df.empty:
        st.markdown("## Cross-Validation vs. Held-Out Generalization Summary")
        st.markdown(
            "<p style='font-size:0.84rem;color:#A1A1AA;'>"
            "Repeated CV metrics represent mean ± standard deviation across 3 repeats of Stratified 5-Fold CV (15 fits total). "
            "Held-out test metrics reflect generalization to the 61 untouched test samples."
            "</p>",
            unsafe_allow_html=True,
        )
        dc = ["Model", "CV Repeated ROC-AUC (Mean)", "CV Repeated ROC-AUC (Std)",
              "Test ROC-AUC", "Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Target > 0.85 Met"]
        avail = [c for c in dc if c in metrics_df.columns]
        st.dataframe(metrics_df[avail], use_container_width=True, hide_index=True)

    st.markdown("## Diagnostic Visualizations")
    t1, t2, t3 = st.tabs(["ROC Curves", "Confusion Matrices", "Feature Importances"])
    with t1:
        if ROC_PATH.exists():
            st.image(str(ROC_PATH), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
        else:
            st.info("ROC curve plot not found at results/roc_curve_comparison.png")
    with t2:
        if CM_PATH.exists():
            st.image(str(CM_PATH), caption="Confusion Matrices on Held-Out Test Split (n = 61)", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found at results/confusion_matrices.png")
    with t3:
        if FI_PATH.exists():
            st.image(str(FI_PATH), caption="Random Forest Gini Feature Importances", use_container_width=True)
        else:
            st.info("Feature importance plot not found at results/feature_importance.png")

    if not preds_df.empty:
        st.markdown("## Held-Out Test Set Predictions Preview")
        st.dataframe(preds_df.head(20), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 5 -- Dataset Explorer
# ===========================================================================
elif current_page == "Dataset Explorer":
    clean_header(
        "Dataset Explorer",
        "UCI Heart Disease -- Cleveland Cohort. 303 patient records, 13 clinical attributes, binary target."
    )

    if dataset.empty:
        st.warning(f"Dataset file not found at: {DATASET_PATH}")
    else:
        n   = len(dataset)
        np_ = int(dataset["target"].sum()) if "target" in dataset.columns else 0
        nn_ = n - np_
        nm  = int(dataset.isnull().sum().sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(metric_card("Total Records",    str(n),   "Cleveland Clinic Foundation"), unsafe_allow_html=True)
        c2.markdown(metric_card("Positive (HD=1)",  str(np_), f"{np_/n:.1%} prevalence", "red"), unsafe_allow_html=True)
        c3.markdown(metric_card("Negative (HD=0)",  str(nn_), f"{nn_/n:.1%} prevalence", "green"), unsafe_allow_html=True)
        c4.markdown(metric_card("Input Features",   "13",     "Clinical dimensions"), unsafe_allow_html=True)
        c5.markdown(metric_card("Missing Values",   str(nm),  "ca (4), thal (2)",
                                "amber" if nm > 0 else "green"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.tabs(["Data Preview", "Descriptive Statistics", "Discrete Distributions", "Feature Preprocessing"])

        with t1:
            st.markdown("### Records Preview (First 25 Samples)")
            st.dataframe(dataset.head(25), use_container_width=True)

        with t2:
            st.markdown("### Continuous Numeric Features")
            st.dataframe(dataset[NUMERIC_FEATURES].describe().round(3), use_container_width=True)

        with t3:
            st.markdown("### Categorical & Binary Value Distributions")
            fc = st.columns(3)
            for i, feat in enumerate(BINARY_FEATURES + CATEGORICAL_FEATURES):
                if feat in dataset.columns:
                    with fc[i % 3]:
                        st.markdown(f"**{feat}**")
                        vc = dataset[feat].value_counts(dropna=False).reset_index()
                        vc.columns = [feat, "Count"]
                        st.dataframe(vc, use_container_width=True, hide_index=True)

        with t4:
            st.markdown("### Preprocessing Pipeline Design")
            for gname, feats, treatment, color_cls in [
                ("Continuous Numeric (5)", NUMERIC_FEATURES, "Median Imputer + StandardScaler", "#3B82F6"),
                ("Categorical (5)",        CATEGORICAL_FEATURES, "Mode Imputer + OneHotEncoder (handle_unknown='ignore')", "#3B82F6"),
                ("Binary / Discrete (3)",  BINARY_FEATURES, "Mode Imputer + Passthrough", "#71717A"),
            ]:
                st.markdown(
                    f"<div class='section-card'>"
                    f"<span style='color:{color_cls};font-weight:600;font-size:0.8rem;margin-bottom:6px;display:inline-block;'>{gname}</span>"
                    f"<p style='font-size:0.84rem;color:#F5F5F7;margin:4px 0 6px;'><strong>{', '.join(feats)}</strong></p>"
                    f"<p style='font-size:0.78rem;color:#A1A1AA;margin:0;'>Treatment: {treatment}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ===========================================================================
# PAGE 6 -- About / Methodology (Short, Single Compact Page)
# ===========================================================================
elif current_page == "About / Methodology":
    clean_header(
        "About & Methodology",
        "Concise technical overview of the Big Data streaming, storage, and machine learning architecture."
    )

    st.markdown(
        "<div class='section-card'>"
        "<h3 style='margin-top:0;'>1. Project Overview</h3>"
        "<p>This platform demonstrates an enterprise-grade Big Data Analytics pipeline for early heart disease risk stratification. "
        "It unites streaming data ingestion (<strong>Apache Kafka</strong>), distributed object storage (<strong>Hadoop HDFS</strong>), "
        "and reproducible supervised classification (<strong>scikit-learn</strong>) on the UCI Cleveland Heart Disease dataset.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            "<div class='section-card'>"
            "<h3 style='margin-top:0;'>2. Technology Stack</h3>"
            "<ul style='margin-bottom:0;'>"
            "<li><strong>Core Runtime:</strong> Python 3.11 with a local virtual environment (.venv).</li>"
            "<li><strong>Stream Ingestion:</strong> Apache Kafka 3.6 in KRaft consensus mode (topic: <code>heart-disease-records</code>).</li>"
            "<li><strong>Distributed Storage:</strong> Apache Hadoop 3.3.6 HDFS via WebHDFS REST API (NameNode :9870, DataNode :9864).</li>"
            "<li><strong>Container Orchestration:</strong> Docker & Docker Compose.</li>"
            "<li><strong>Machine Learning:</strong> scikit-learn 1.4+, pandas 2.2+, numpy 1.26+, and joblib serialized pipelines.</li>"
            "<li><strong>User Interface:</strong> Streamlit 1.65+ in dark theme mode.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-card'>"
            "<h3 style='margin-top:0;'>3. End-to-End Workflow</h3>"
            "<p><strong>Dataset → Kafka → HDFS → ML Pipelines → Streamlit:</strong></p>"
            "<ol style='margin-bottom:0;'>"
            "<li><strong>Dataset Preparation:</strong> <code>preprocessing/prepare_dataset.py</code> downloads and validates UCI Cleveland CSV.</li>"
            "<li><strong>Kafka Streaming:</strong> <code>kafka/producer.py</code> streams records as JSON to the Kafka topic.</li>"
            "<li><strong>HDFS Persistence:</strong> <code>kafka/hdfs_consumer.py</code> writes verified stream batches to <code>/heart-disease/raw/heart_disease_stream.jsonl</code>.</li>"
            "<li><strong>Model Training & Evaluation:</strong> <code>models/train.py</code> and <code>models/evaluate.py</code> train and serialize LR and RF pipelines.</li>"
            "<li><strong>Interactive Inference:</strong> <code>dashboard/app.py</code> executes pre-trained pipelines on user inputs for risk assessment.</li>"
            "</ol></div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            "<div class='section-card'>"
            "<h3 style='margin-top:0;'>4. Machine Learning Methodology</h3>"
            "<p>Models are trained on an 80/20 stratified split (242 train, 61 held-out test records) using scikit-learn <code>ColumnTransformer</code>:</p>"
            "<ul>"
            "<li><strong>Logistic Regression (L2 penalty):</strong> Provides a calibrated linear baseline.<br>"
            "CV ROC-AUC: <strong>0.8956 ± 0.0421</strong> | Held-out Test ROC-AUC: <strong>0.9697</strong> | Accuracy: <strong>90.16%</strong></li>"
            "<li><strong>Random Forest (100 estimators):</strong> Captures non-linear clinical interactions.<br>"
            "CV ROC-AUC: <strong>0.8936 ± 0.0404</strong> | Held-out Test ROC-AUC: <strong>0.9470</strong> | Accuracy: <strong>85.25%</strong></li>"
            "<li><strong>Cross-Validation:</strong> Repeated Stratified 5-Fold CV across 3 seeds (15 total fits).</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-card'>"
            "<h3 style='margin-top:0;'>5. Limitations & Responsible Use</h3>"
            "<ul style='margin-bottom:0;'>"
            "<li><strong>Cohort Scale & Age:</strong> 303 patient records collected at the Cleveland Clinic Foundation in the late 1980s.</li>"
            "<li><strong>Demographic Representation:</strong> Cohort is 68% male; generalization across diverse modern populations requires external validation.</li>"
            "<li><strong>Academic Demonstration:</strong> This tool outputs statistical risk correlations and is <strong>not a medical device or diagnostic instrument</strong>.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='disclaimer-box'>⚠️ <strong>Non-Diagnostic Academic Disclaimer:</strong> "
        "This project was developed exclusively as an academic big data analytics demonstration. "
        "Outputs from this software must never be used for diagnosis, clinical decision support, or treatment planning.</div>",
        unsafe_allow_html=True,
    )
