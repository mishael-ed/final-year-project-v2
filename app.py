from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

from sap.config import APP_SUBTITLE, APP_TITLE, DEFAULT_THRESHOLDS, INTERVENTION_MESSAGES
from sap.db import (
    fetch_imported_records,
    get_db_filter_options,
    save_imported_records,
    save_prediction_results,
    test_connection,
)
from sap.explainer import (
    SHAP_AVAILABLE,
    explain_student,
    explain_student_ensemble,
    explain_student_lstm,
    explain_student_xgboost,
    plot_global_importance,
    plot_student_waterfall,
)
from sap.features import build_term_features, clean_data
from sap.io import load_table, prepare
from sap.lms_model import infer_lms, prepare_lms
from sap.model import infer
from sap.reporting import export_csv, export_pdf

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "db_url": "",
        "auto_save_mysql": False,
        "preds": None,
        "pred_source": "",
        "term_df": None,
        "lms_preds": None,
        "lms_pred_source": "",
        # Auth
        "access_token": None,
        "refresh_token": None,
        "current_user": None,   # {"username": ..., "email": ..., "role": ...}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["db_url"]:
        try:
            st.session_state["db_url"] = st.secrets.get("MYSQL_URL", "")
        except Exception:
            st.session_state["db_url"] = os.getenv("MYSQL_URL", "")

_init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────────────────────────────────────

def _current_role() -> str:
    user = st.session_state.get("current_user")
    return user["role"] if user else "viewer"


def _auth_headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _try_refresh() -> bool:
    """Attempt a silent token refresh. Returns True if successful."""
    rt = st.session_state.get("refresh_token")
    if not rt:
        return False
    try:
        r = requests.post(f"{BACKEND_URL}/auth/refresh", json={"refresh_token": rt}, timeout=30)
        if r.status_code == 200:
            st.session_state["access_token"] = r.json()["access_token"]
            return True
    except Exception:
        pass
    return False


def _logout() -> None:
    st.session_state["access_token"] = None
    st.session_state["refresh_token"] = None
    st.session_state["current_user"]  = None


_LOGIN_CSS = """
<style>
header[data-testid="stHeader"],
[data-testid="stSidebar"],
footer,
[data-testid="stStatusWidget"],
button[data-testid="stBaseButton-header"],
.stDeployButton { display: none !important; }

/* Prevent Streamlit rerun grey-out */
[data-stale], [data-stale="true"] { opacity: 1 !important; }

/* Prevent iOS zoom on input focus (font-size must be >= 16px) */
input, textarea, select { font-size: 16px !important; }
</style>
"""

# ── Palette tokens ────────────────────────────────────────────────────────────
# bg0  = page background   #0d1117
# bg1  = surface / card    #161b22
# bg2  = raised element    #21262d
# bdr  = border            #30363d
# tx0  = primary text      #e6edf3
# tx1  = secondary text    #8b949e
# acc  = accent blue       #2563eb / #3b82f6
# ─────────────────────────────────────────────────────────────────────────────

_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600;700&display=swap');

* { font-family: 'Volte', 'Volte Rounded', 'Jost', 'Segoe UI', sans-serif !important; }
[data-testid="stExpandSidebarButton"] span,
[data-testid="stExpandSidebarButton"] > *,
[data-testid="stMainMenuButton"] span,
[data-testid="stMainMenuButton"] > *,
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

.stApp { background: #e8ecf1; color: #111827; }
footer { visibility: hidden; }

p, li, span, label, div { color: #111827; }

small, .stCaption p,
[data-testid="stCaptionContainer"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #374151 !important; font-size: .82rem; }

button[data-testid="stBaseButton-header"] { display: none !important; }
.stDeployButton, [data-testid="stDeployButton"], button[kind="deployButton"] { display: none !important; }
h2 a, h3 a, h4 a { display: none !important; }

/* Rolling spinner */
@keyframes sap-roll { to { transform: rotate(360deg); } }
[data-testid="stSpinner"] > div {
    border: 2px solid #d1d5db !important;
    border-top-color: #1a56db !important;
    border-radius: 50% !important;
    width: 22px !important; height: 22px !important;
    animation: sap-roll 0.65s linear infinite !important;
}
[data-testid="stSpinner"] svg { display: none !important; }

.hero {
    background: linear-gradient(120deg, #1e3a8a 0%, #1a56db 60%, #3b82f6 100%);
    border-radius: 0px; padding: 1.2rem 1.6rem; color: white; margin-bottom: 1rem;
    display: flex; align-items: center;
}
.hero h2 { margin: 0; font-size: clamp(1rem, 3.5vw, 1.4rem); font-weight: 700; letter-spacing: 0.01em; white-space: normal; word-break: break-word; color: white !important; }
.hero p  { margin: .3rem 0 0; opacity: .9; font-size: .9rem; color: white !important; }
@media (max-width: 640px) {
    .hero { flex-direction: column !important; align-items: flex-start !important; gap: .4rem; }
    .hero > div:last-child { text-align: left !important; }
}

div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #9ca3af; border-radius: 0px;
    padding: .5rem .75rem;
}

.badge-high   { background:#dc2626; color:white; border-radius:0px; padding:2px 8px; font-weight:bold; }
.badge-medium { background:#d97706; color:white; border-radius:0px; padding:2px 8px; font-weight:bold; }
.badge-low    { background:#16a34a; color:white; border-radius:0px; padding:2px 8px; font-weight:bold; }

h3, h4 { color: #1e3a8a !important; }
.streamlit-expanderHeader { color: #1a56db !important; font-weight: 600; }

.stButton > button { border-radius: 0px !important; }

.stTextInput > div > div > input { border-radius: 0px !important; border-color: #9ca3af !important; font-size: 16px !important; }
[data-baseweb="select"] { border-radius: 0px !important; }
[data-baseweb="select"] > div { border-color: #9ca3af !important; }
[data-baseweb="input"]  { border-radius: 0px !important; border-color: #9ca3af !important; }
[data-baseweb="input"] input { font-size: 16px !important; }

/* Prevent Streamlit rerun grey-out */
[data-stale], [data-stale="true"] { opacity: 1 !important; }

[data-testid="stFileUploader"] { border-radius: 0px !important; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px solid #9ca3af !important; border-radius: 0px !important; background: #ffffff !important; }

.stTabs [data-baseweb="tab"]     { border-radius: 0px !important; color: #374151 !important; }
.stTabs [data-baseweb="tab-list"] { border-radius: 0px !important; border-bottom: 2px solid #9ca3af !important; background: #ffffff !important; padding-left: 1.6rem !important; }

.stDataFrame { border-radius: 0px !important; }
.streamlit-expanderContent { border-radius: 0px !important; }

[data-testid="stAlert"] {
    border-radius: 0px !important;
    border-left: 4px solid #1a56db !important;
    background: #dbeafe !important;
}
[data-testid="stAlert"] p { color: #1e3a8a !important; font-weight: 500 !important; }

.block-container { max-width: 1200px; }
</style>
"""


def _login_gate() -> None:
    """Block the app until the user authenticates against the backend."""
    if st.session_state.get("current_user"):
        return

    # Try silent token refresh on page reload
    if _try_refresh():
        try:
            r = requests.get(f"{BACKEND_URL}/auth/me", headers=_auth_headers(), timeout=30)
            if r.status_code == 200:
                st.session_state["current_user"] = r.json()
                return
        except Exception:
            pass

    # ── Render the login page ────────────────────────────────────────────────
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.title("Student Academic Performance Prediction System")
        st.caption("Sign in to continue")
        st.divider()

        username = st.text_input("Username", key="_login_username")
        password = st.text_input("Password", type="password", key="_login_password")

        error_slot = st.empty()

        if st.button("Sign in", use_container_width=True, type="primary"):
            if not username or not password:
                error_slot.error("Please enter both username and password.")
            else:
                with st.spinner("Signing in…"):
                    try:
                        r = requests.post(
                            f"{BACKEND_URL}/auth/login",
                            json={"username": username, "password": password},
                            timeout=30,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.session_state["access_token"]  = data["access_token"]
                            st.session_state["refresh_token"] = data["refresh_token"]
                            st.session_state["current_user"]  = {
                                "username": data["username"],
                                "role":     data["role"],
                            }
                            st.rerun()
                        elif r.status_code == 401:
                            error_slot.error("Incorrect username or password.")
                        elif r.status_code == 403:
                            error_slot.error("This account has been disabled. Contact your administrator.")
                        else:
                            error_slot.error(f"Login failed (status {r.status_code}). Try again.")
                    except requests.exceptions.ConnectionError:
                        error_slot.error(
                            f"Cannot reach the backend at **{BACKEND_URL}**. "
                            "Ensure `python run_backend.py` is running."
                        )

        st.caption("No account? Contact your system administrator.")
    st.stop()

_login_gate()

# ── Main app styles + hero (only rendered when authenticated) ─────────────────
st.markdown(_APP_CSS, unsafe_allow_html=True)

_user = st.session_state.get("current_user", {})
_username = _user.get("username", "")
_welcome = f"Welcome, {_username.replace('_', ' ').title()}" if _username else ""

st.markdown(f"""
<div class="hero">
  <div style="flex:1;">
    <h2>{APP_TITLE}</h2>
    <p>{APP_SUBTITLE}</p>
  </div>
  <div style="text-align:right; white-space:nowrap; opacity:0.9;">
    <p style="margin:0; font-size:0.88rem;">{_welcome}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Sign out", use_container_width=True):
        _logout()
        st.rerun()

    st.markdown("---")
    st.markdown("### Database Integration")
    db_url = st.text_input(
        "MySQL SQLAlchemy URL",
        value=st.session_state["db_url"],
        type="password",
        placeholder="mysql+pymysql://user:pass@host:3306/db",
    )
    st.session_state["db_url"] = db_url.strip()
    st.session_state["auto_save_mysql"] = st.checkbox(
        "Auto-save records & prediction runs",
        value=st.session_state["auto_save_mysql"],
    )
    if st.button("Test Connection"):
        if not st.session_state["db_url"]:
            st.warning("Enter a MySQL URL first.")
        else:
            try:
                test_connection(st.session_state["db_url"])
                st.success("Connected")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "This system uses an ensemble of **Random Forest**, **XGBoost**, and "
        "**LSTM** models to predict student academic outcomes and flag students "
        "at risk of poor performance. Predictions are explained via **SHAP** values."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_issue_reason(df: pd.DataFrame) -> pd.DataFrame:
    def reason(r: pd.Series) -> str:
        issues: list[tuple[str, float]] = []
        ca_pct    = float(r.get("avg_ca_pct",    100))
        att_pct   = float(r.get("avg_attendance", 100))
        overall   = float(r.get("avg_overall_pct", 100))
        fail_cnt  = int(r.get("subject_failure_count", 0))

        if ca_pct   < 55: issues.append(("Low CA",             55 - ca_pct))
        if att_pct  < 75: issues.append(("Low attendance",     75 - att_pct))
        if overall  < 50: issues.append(("Low performance",    50 - overall))
        if fail_cnt >= 3: issues.append(("Many failed subjects", (fail_cnt - 2) * 5))

        if not issues:
            return "No major risk"
        issues.sort(key=lambda x: x[1], reverse=True)
        return ", ".join(lbl for lbl, _ in issues[:2])

    out = df.copy()
    out["main_issue"] = out.apply(reason, axis=1)
    return out


def _add_intervention(df: pd.DataFrame) -> pd.DataFrame:
    def get_intervention(issue: str) -> str:
        for key, msg in INTERVENTION_MESSAGES.items():
            if key in issue:
                return msg
        return INTERVENTION_MESSAGES["No major risk"]

    out = df.copy()
    out["intervention"] = out["main_issue"].apply(get_intervention)
    return out


def _present(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Student Name" not in out.columns:
        out["Student Name"] = out["Student ID"].astype(str)
    if "avg_overall_pct" not in out.columns:
        out["avg_overall_pct"] = out.get("avg_ca_pct", 0.0)

    out["Likelihood of Passing (%)"] = (out["pass_probability"] * 100).round(1)
    out["CA Progress (%)"]           = out["avg_ca_pct"].round(1)
    out["Overall Score (%)"]         = out["avg_overall_pct"].round(1)
    out["Attendance (%)"]            = out["avg_attendance"].round(1)
    out["Risk Level"]                = out["risk_tier"]
    out["Predicted Outcome"]         = out["predicted_outcome"]
    out["Subjects at Risk"]          = out["subject_failure_count"].astype(int)
    out["Main Issue"]                = out["main_issue"]
    out["Recommended Action"]        = out.get("intervention", "")

    keep = [
        "Student ID", "Student Name", "Term", "Class",
        "Predicted Outcome", "Likelihood of Passing (%)", "Risk Level",
        "CA Progress (%)", "Overall Score (%)", "Attendance (%)",
        "Subjects at Risk", "Main Issue", "Recommended Action",
    ]
    return out[[c for c in keep if c in out.columns]]


def _maybe_save(raw_df: pd.DataFrame | None, preds: pd.DataFrame,
                source: str, save_raw: bool) -> None:
    db_url    = st.session_state.get("db_url", "").strip()
    auto_save = st.session_state.get("auto_save_mysql", False)
    if not db_url or not auto_save:
        return
    try:
        if save_raw and raw_df is not None and len(raw_df):
            n = save_imported_records(raw_df, db_url, source_label=source)
            st.caption(f"Saved {n:,} records to MySQL.")
        run_id = save_prediction_results(preds, db_url, source_label=source)
        st.caption(f"Saved prediction run (run_id={run_id}) to MySQL.")
    except Exception as e:
        st.warning(f"MySQL save skipped: {e}")


def _run_pipeline(raw_df: pd.DataFrame, source: str, save_raw: bool) -> None:
    thresholds = DEFAULT_THRESHOLDS
    prepared   = prepare(raw_df, mode="predict")
    clean      = clean_data(prepared)
    term_df    = build_term_features(clean)
    preds      = infer(term_df, thresholds=thresholds)
    preds      = _add_issue_reason(preds)
    preds      = _add_intervention(preds)

    st.session_state["preds"]    = preds
    st.session_state["term_df"]  = term_df
    st.session_state["pred_source"] = source

    models_used = preds["prediction_models"].iloc[0] if "prediction_models" in preds.columns else "Ensemble"
    st.success(f"Predictions generated for {len(preds):,} student-term records  ·  {models_used}")
    _maybe_save(raw_df, preds, source, save_raw)


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_predict, tab_lms, tab_train, tab_explain, tab_db = st.tabs([
    "Academic Predict",
    "LMS Predict",
    "Train Models",
    "Explainability (SHAP)",
    "Database Records",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.info(
        "Models are pre-trained by the project team. "
        "Admins only need to upload intra-semester records to get predictions."
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### Step 1 · Upload or enter student records")
    col_up, col_manual = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("#### Upload File (CSV / Excel)")
        pred_file = st.file_uploader(
            "Drop file here", type=["csv", "xlsx", "xls"], key="pred_file"
        )
        if pred_file and st.button("Run Predictions from File", use_container_width=True):
            suffix = "." + pred_file.name.rsplit(".", 1)[-1].lower()
            tmp    = Path(f"_tmp_pred_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}")
            try:
                tmp.write_bytes(pred_file.getbuffer())
                _run_pipeline(load_table(tmp), source="upload", save_raw=True)
            except Exception as e:
                st.error("Prediction failed — check file format and column names.")
                st.caption(f"Detail: {e}")
            finally:
                tmp.unlink(missing_ok=True)

    with col_manual:
        st.markdown("#### Manual Entry")
        seed = pd.DataFrame([{
            "Student ID": "STU_001", "Student Name": "Mishael Edegwa",
            "Gender": "Male",        "Term": "2026_T1",
            "Class": "SS2",          "Subject": "Mathematics",
            "CA Score": 35,          "CA Total Score": 40,
            "Classes Held": 30,      "Classes Attended": 27,
            "Behavioral Rating": "Good",
        }])
        manual_df = st.data_editor(
            seed, num_rows="dynamic", use_container_width=True,
            column_config={
                "Behavioral Rating": st.column_config.SelectboxColumn(
                    "Behavioral Rating",
                    options=["Poor", "Average", "Good", "Excellent"],
                    required=True,
                )
            },
            key="manual_editor",
        )
        if st.button("Run Predictions from Manual Entry", use_container_width=True):
            entry = manual_df.dropna(how="all").copy()
            entry["Student ID"] = entry["Student ID"].astype(str).str.strip()
            entry = entry[entry["Student ID"] != ""]
            if entry.empty:
                st.warning("Please add at least one row with a Student ID.")
            else:
                try:
                    _run_pipeline(entry, source="manual", save_raw=True)
                except Exception as e:
                    st.error("Manual prediction failed — check required fields.")
                    st.caption(f"Detail: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state["preds"] is not None:
        preds = st.session_state["preds"]

        st.markdown("---")
        st.markdown("### Step 2 · Review predictions & early-warning list")
        st.caption(f"Source: {st.session_state['pred_source']}")

        # KPI strip
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Students",  len(preds))
        m2.metric("High Risk",         int((preds["risk_tier"] == "High").sum()))
        m3.metric("Medium Risk",       int((preds["risk_tier"] == "Medium").sum()))
        m4.metric("Low Risk",          int((preds["risk_tier"] == "Low").sum()))
        m5.metric("Avg Pass Likelihood",
                  f"{preds['pass_probability'].mean() * 100:.1f}%")

        # Filters
        st.markdown("#### Filters")
        fc1, fc2, fc3, fc4 = st.columns(4)
        quick        = fc1.selectbox("Quick Filter", ["All", "High Risk Only", "Predicted Fail"])
        class_filter = fc2.multiselect("Class",  sorted(preds["Class"].dropna().unique()))
        term_filter  = fc3.multiselect("Term",   sorted(preds["Term"].dropna().unique()))
        sid_filter   = fc4.text_input("Search Student ID")

        filt = preds.copy()
        if quick == "High Risk Only":
            filt = filt[filt["risk_tier"] == "High"]
        elif quick == "Predicted Fail":
            filt = filt[filt["predicted_outcome"] == "Fail"]
        if class_filter: filt = filt[filt["Class"].isin(class_filter)]
        if term_filter:  filt = filt[filt["Term"].isin(term_filter)]
        if sid_filter:   filt = filt[filt["Student ID"].astype(str).str.contains(sid_filter, case=False, na=False)]

        view = _present(filt)
        st.dataframe(
            view.style.apply(
                lambda col: [
                    "background-color:#fee2e2; color:#991b1b" if v == "High"
                    else "background-color:#fef3c7; color:#92400e" if v == "Medium"
                    else "background-color:#d1fae5; color:#065f46" if v == "Low"
                    else ""
                    for v in col
                ] if col.name == "Risk Level" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
        )

        # Downloads
        st.markdown("#### Download Report")
        report_name = f"sap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_path    = export_csv(view, report_name)
        pdf_path    = export_pdf(view, report_name)

        dl1, dl2 = st.columns(2)
        with dl1:
            with open(csv_path, "rb") as f:
                st.download_button(
                    "Download CSV", data=f.read(),
                    file_name=csv_path.name, mime="text/csv",
                    use_container_width=True,
                )
        with dl2:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF", data=f.read(),
                    file_name=pdf_path.name, mime="application/pdf",
                    use_container_width=True,
                )

        # Visualisations
        st.markdown("---")
        st.markdown("### Step 3 · Analytics")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            vc1, vc2 = st.columns(2)

            # Risk distribution pie
            with vc1:
                st.markdown("#### Risk Distribution")
                counts  = preds["risk_tier"].value_counts()
                pie_labels = counts.index.tolist()
                pie_vals   = counts.values.tolist()
                pie_colors = ["#e74c3c" if l == "High" else "#e67e22" if l == "Medium" else "#27ae60"
                              for l in pie_labels]
                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                ax_pie.pie(pie_vals, labels=pie_labels, colors=pie_colors,
                           autopct="%1.0f%%", startangle=90,
                           textprops={"color": "#111827", "fontsize": 10})
                fig_pie.patch.set_facecolor("#f8f9fb")
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            # Pass probability histogram
            with vc2:
                st.markdown("#### Pass Probability Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(4, 4))
                ax_hist.hist(
                    preds["pass_probability"] * 100, bins=20,
                    color="#1a56db", edgecolor="#f8f9fb", alpha=0.85,
                )
                ax_hist.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5, label="50% threshold")
                ax_hist.axvline(75, color="#16a34a", linestyle="--", linewidth=1.5, label="75% threshold")
                ax_hist.set_xlabel("Likelihood of Passing (%)", color="#111827")
                ax_hist.set_ylabel("No. of Students", color="#111827")
                ax_hist.tick_params(colors="#111827")
                ax_hist.spines[:].set_color("#e5e7eb")
                ax_hist.set_facecolor("#ffffff")
                fig_hist.patch.set_facecolor("#f8f9fb")
                ax_hist.legend(facecolor="#ffffff", labelcolor="#111827")
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            # Attendance vs CA scatter (risk-coloured)
            if "avg_attendance" in preds.columns and "avg_ca_pct" in preds.columns:
                st.markdown("#### Attendance vs. CA Score (coloured by risk)")
                risk_cmap = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
                fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
                for tier, grp in preds.groupby("risk_tier"):
                    ax_sc.scatter(
                        grp["avg_attendance"], grp["avg_ca_pct"],
                        c=risk_cmap.get(tier, "#7c3aed"),
                        label=tier, alpha=0.75, edgecolors="none", s=50,
                    )
                ax_sc.set_xlabel("Average Attendance (%)", color="#111827")
                ax_sc.set_ylabel("Average CA Score (%)",   color="#111827")
                ax_sc.tick_params(colors="#111827")
                ax_sc.spines[:].set_color("#e5e7eb")
                ax_sc.set_facecolor("#ffffff")
                fig_sc.patch.set_facecolor("#f8f9fb")
                ax_sc.legend(title="Risk", facecolor="#ffffff", labelcolor="#111827",
                             title_fontsize=9)
                st.pyplot(fig_sc)
                plt.close(fig_sc)

        except Exception as e:
            st.caption(f"Charts unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – LMS PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_lms:
    st.info(
        "Upload LMS engagement data to predict student pass/fail outcomes based on "
        "digital learning behaviour (logins, submissions, video completion, etc.)."
    )

    def _run_lms_pipeline(raw_df: pd.DataFrame, source: str) -> None:
        try:
            preds = infer_lms(raw_df, thresholds=DEFAULT_THRESHOLDS)
        except RuntimeError as e:
            st.error(str(e))
            return
        st.session_state["lms_preds"]       = preds
        st.session_state["lms_pred_source"] = source
        models_used = preds["prediction_models"].iloc[0] if len(preds) else "—"
        st.success(f"LMS predictions generated for {len(preds):,} students  ·  {models_used}")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### Step 1 · Upload or enter LMS data")
    lms_col_up, lms_col_manual = st.columns([1, 1], gap="large")

    with lms_col_up:
        st.markdown("#### Upload File (CSV / Excel)")
        lms_file = st.file_uploader(
            "Drop file here", type=["csv", "xlsx", "xls"], key="lms_pred_file"
        )
        if lms_file and st.button("Run LMS Predictions from File", use_container_width=True):
            suffix = "." + lms_file.name.rsplit(".", 1)[-1].lower()
            tmp    = Path(f"_tmp_lms_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}")
            try:
                tmp.write_bytes(lms_file.getbuffer())
                _run_lms_pipeline(load_table(tmp), source="lms_upload")
            except Exception as e:
                st.error("LMS prediction failed — check file format.")
                st.caption(f"Detail: {e}")
            finally:
                tmp.unlink(missing_ok=True)

        with st.expander("Required columns"):
            st.markdown(
                "| Column | Example |\n|---|---|\n"
                "| `student_id` | S001 *(optional)* |\n"
                "| `lms_logins_per_semester` | 57 |\n"
                "| `avg_session_duration_minutes` | 49 |\n"
                "| `assignment_submission_rate` | 0.76 *(or 76)* |\n"
                "| `forum_participation_count` | 10 |\n"
                "| `video_completion_rate` | 0.75 *(or 75)* |"
            )

    with lms_col_manual:
        st.markdown("#### Manual Entry")
        lms_seed = pd.DataFrame([{
            "student_id": "STU_001",
            "lms_logins_per_semester": 58,
            "avg_session_duration_minutes": 49,
            "assignment_submission_rate": 0.75,
            "forum_participation_count": 10,
            "video_completion_rate": 0.75,
        }])
        lms_manual_df = st.data_editor(
            lms_seed, num_rows="dynamic", use_container_width=True,
            column_config={
                "lms_logins_per_semester": st.column_config.NumberColumn("LMS Logins", min_value=0, step=1),
                "avg_session_duration_minutes": st.column_config.NumberColumn("Avg Session (min)", min_value=0, step=1),
                "assignment_submission_rate": st.column_config.NumberColumn("Submission Rate", min_value=0.0, max_value=1.0, step=0.01),
                "forum_participation_count": st.column_config.NumberColumn("Forum Posts", min_value=0, step=1),
                "video_completion_rate": st.column_config.NumberColumn("Video Completion", min_value=0.0, max_value=1.0, step=0.01),
            },
            key="lms_manual_editor",
        )
        if st.button("Run LMS Predictions from Manual Entry", use_container_width=True):
            entry = lms_manual_df.dropna(how="all").copy()
            if entry.empty:
                st.warning("Please add at least one row.")
            else:
                try:
                    _run_lms_pipeline(entry, source="lms_manual")
                except Exception as e:
                    st.error("LMS prediction failed.")
                    st.caption(f"Detail: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("lms_preds") is not None:
        lms_preds = st.session_state["lms_preds"]

        st.markdown("---")
        st.markdown("### Step 2 · LMS Prediction Results")
        st.caption(f"Source: {st.session_state.get('lms_pred_source', '')}")

        # KPI strip
        lm1, lm2, lm3, lm4, lm5 = st.columns(5)
        lm1.metric("Total Students",  len(lms_preds))
        lm2.metric("High Risk",        int((lms_preds["risk_tier"] == "High").sum()))
        lm3.metric("Medium Risk",      int((lms_preds["risk_tier"] == "Medium").sum()))
        lm4.metric("Low Risk",         int((lms_preds["risk_tier"] == "Low").sum()))
        lm5.metric("Avg Pass Likelihood",
                   f"{lms_preds['pass_probability'].mean() * 100:.1f}%")

        # Filter
        lf1, lf2, lf3 = st.columns(3)
        lms_quick   = lf1.selectbox("Quick Filter", ["All", "High Risk Only", "Predicted Fail"],
                                    key="lms_quick")
        lms_outcome = lf2.selectbox("Predicted Outcome", ["All", "Pass", "Fail"],
                                    key="lms_outcome")
        lms_sid     = lf3.text_input("Search Student ID", key="lms_sid")

        lms_filt = lms_preds.copy()
        if lms_quick == "High Risk Only":
            lms_filt = lms_filt[lms_filt["risk_tier"] == "High"]
        elif lms_quick == "Predicted Fail":
            lms_filt = lms_filt[lms_filt["predicted_outcome"] == "Fail"]
        if lms_outcome != "All":
            lms_filt = lms_filt[lms_filt["predicted_outcome"] == lms_outcome]
        if lms_sid:
            lms_filt = lms_filt[lms_filt["Student ID"].astype(str).str.contains(lms_sid, case=False, na=False)]

        # Display table — rename for readability
        lms_view = lms_filt.copy()
        lms_view = lms_view.rename(columns={
            "predicted_outcome": "Prediction",
            "pass_probability":  "Pass Probability",
            "risk_tier":         "Risk Level",
        })
        lms_view["Pass Probability"] = (lms_view["Pass Probability"] * 100).round(1).astype(str) + "%"
        display_cols = [c for c in [
            "Student ID", "LMS Logins", "Avg Session (min)",
            "Submission Rate", "Forum Posts", "Video Completion",
            "Prediction", "Pass Probability", "Risk Level",
        ] if c in lms_view.columns]

        st.dataframe(
            lms_view[display_cols].style.apply(
                lambda col: [
                    "background-color:#fee2e2; color:#991b1b" if v == "High"
                    else "background-color:#fef3c7; color:#92400e" if v == "Medium"
                    else "background-color:#d1fae5; color:#065f46" if v == "Low"
                    else ""
                    for v in col
                ] if col.name == "Risk Level" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
        )

        # Downloads
        st.markdown("#### Download")
        lms_report = f"lms_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        lms_csv    = export_csv(lms_view[display_cols], lms_report)
        lms_pdf    = export_pdf(lms_view[display_cols], lms_report)
        dl1, dl2 = st.columns(2)
        with dl1:
            with open(lms_csv, "rb") as f:
                st.download_button("Download CSV", data=f.read(),
                                   file_name=lms_csv.name, mime="text/csv",
                                   use_container_width=True)
        with dl2:
            with open(lms_pdf, "rb") as f:
                st.download_button("Download PDF", data=f.read(),
                                   file_name=lms_pdf.name, mime="application/pdf",
                                   use_container_width=True)

        # Charts
        st.markdown("---")
        st.markdown("### Step 3 · Analytics")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            lvc1, lvc2 = st.columns(2)
            with lvc1:
                st.markdown("#### Risk Distribution")
                counts = lms_preds["risk_tier"].value_counts()
                pie_labels = counts.index.tolist()
                pie_vals   = counts.values.tolist()
                pie_colors = ["#e74c3c" if l == "High" else "#e67e22" if l == "Medium" else "#27ae60"
                              for l in pie_labels]
                fig_lp, ax_lp = plt.subplots(figsize=(4, 4))
                ax_lp.pie(pie_vals, labels=pie_labels, colors=pie_colors,
                          autopct="%1.0f%%", startangle=90,
                          textprops={"color": "#111827", "fontsize": 10})
                fig_lp.patch.set_facecolor("#e8ecf1")
                st.pyplot(fig_lp)
                plt.close(fig_lp)

            with lvc2:
                st.markdown("#### Pass Probability Distribution")
                fig_lh, ax_lh = plt.subplots(figsize=(4, 4))
                ax_lh.hist(lms_preds["pass_probability"] * 100, bins=20,
                           color="#1a56db", edgecolor="#e8ecf1", alpha=0.85)
                ax_lh.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5, label="50% threshold")
                ax_lh.axvline(75, color="#16a34a", linestyle="--", linewidth=1.5, label="75% threshold")
                ax_lh.set_xlabel("Likelihood of Passing (%)", color="#111827")
                ax_lh.set_ylabel("No. of Students",           color="#111827")
                ax_lh.tick_params(colors="#111827")
                ax_lh.spines[:].set_color("#9ca3af")
                ax_lh.set_facecolor("#ffffff")
                fig_lh.patch.set_facecolor("#e8ecf1")
                ax_lh.legend(facecolor="#ffffff", labelcolor="#111827")
                st.pyplot(fig_lh)
                plt.close(fig_lh)


        except Exception as e:
            st.caption(f"Charts unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if _current_role() != "admin":
        st.warning("Training controls are restricted to administrators. You have read-only access to this page.")
    st.markdown("### Train / Retrain Ensemble Models")
    st.info(
        "Upload a labelled training file (must include **Final Outcome** column). "
        "Models are saved to `./models/` and used for all future predictions."
    )

    train_file = st.file_uploader("Upload Training File (CSV / Excel)", type=["csv", "xlsx", "xls"], key="train_file")
    skip_lstm  = st.checkbox("Skip LSTM training (faster — for quick retrain)", value=False)

    if train_file and _current_role() == "admin" and st.button("Start Training", use_container_width=True):
        suffix = "." + train_file.name.rsplit(".", 1)[-1].lower()
        tmp    = Path(f"_tmp_train_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}")
        try:
            tmp.write_bytes(train_file.getbuffer())
            df      = load_table(tmp)
            prepared = prepare(df, mode="train")
            clean   = clean_data(prepared)
            term_df = build_term_features(clean)

            st.write(f"Loaded **{len(df):,}** raw records → **{len(term_df):,}** student-term rows.")

            from sap.model import train_lstm, train_rf, train_xgboost

            # RF
            with st.spinner("Training Random Forest…"):
                rf_res = train_rf(term_df)
            c1, c2, c3, c4 = st.columns(4)
            m = rf_res.metrics
            c1.metric("RF Accuracy",  f"{m.get('accuracy', 0):.1%}")
            c2.metric("RF Precision", f"{m.get('precision', 0):.1%}")
            c3.metric("RF Recall",    f"{m.get('recall', 0):.1%}")
            c4.metric("RF F1",        f"{m.get('f1', 0):.1%}")

            if m.get("auc"):
                st.caption(f"RF AUC-ROC: {m.get('auc', 0):.4f}  ·  Method: {m.get('evaluation_method', '')}")

            st.markdown("**Feature Importance (RF)**")
            fi = rf_res.feature_importance.copy()
            fi["importance %"] = (fi["importance"] * 100).round(2)
            st.dataframe(fi[["feature", "importance %"]], use_container_width=False)

            # XGBoost
            with st.spinner("Training XGBoost…"):
                xgb_m = train_xgboost(term_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("XGB Accuracy",  f"{xgb_m.get('accuracy', 0):.1%}")
            c2.metric("XGB Precision", f"{xgb_m.get('precision', 0):.1%}")
            c3.metric("XGB Recall",    f"{xgb_m.get('recall', 0):.1%}")
            c4.metric("XGB F1",        f"{xgb_m.get('f1', 0):.1%}")

            # LSTM
            if not skip_lstm:
                with st.spinner("Training LSTM (this may take a minute)…"):
                    lstm_m = train_lstm(term_df)
                if "error" in lstm_m:
                    st.warning(f"LSTM: {lstm_m['error']}")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("LSTM Accuracy",  f"{lstm_m.get('accuracy', 0):.1%}")
                    c2.metric("LSTM Precision", f"{lstm_m.get('precision', 0):.1%}")
                    c3.metric("LSTM Recall",    f"{lstm_m.get('recall', 0):.1%}")
                    c4.metric("LSTM F1",        f"{lstm_m.get('f1', 0):.1%}")

            st.success("All models trained and saved to ./models/")

        except Exception as e:
            st.error(f"Training failed: {e}")
        finally:
            tmp.unlink(missing_ok=True)

    # ── LMS Retrain ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Train / Retrain LMS Models")
    st.info(
        "Upload a labelled LMS dataset (must include **pass_fail** column and the 5 engagement features). "
        "Models are saved to `./models/` and used for all future LMS predictions."
    )

    lms_train_file = st.file_uploader("Upload LMS Training File (CSV)", type=["csv"], key="lms_train_file")

    if lms_train_file and _current_role() == "admin" and st.button("Start LMS Training", use_container_width=True):
        tmp_lms = Path(f"_tmp_lms_train_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        try:
            tmp_lms.write_bytes(lms_train_file.getbuffer())
            lms_df = pd.read_csv(tmp_lms)
            st.write(f"Loaded **{len(lms_df):,}** rows.")

            from sap.lms_model import prepare_lms, train_lms_rf, train_lms_xgboost

            lms_prepared = prepare_lms(lms_df)

            if "target" not in lms_prepared.columns:
                st.error("Dataset must contain a **pass_fail** column (0 = Fail, 1 = Pass).")
            else:
                vc = lms_prepared["target"].value_counts().to_dict()
                st.caption(f"Target distribution — Pass: {vc.get(1, 0)}, Fail: {vc.get(0, 0)}")

                # RF
                with st.spinner("Training LMS Random Forest…"):
                    lms_rf = train_lms_rf(lms_prepared)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RF Accuracy",  f"{lms_rf.get('accuracy', 0):.1%}")
                c2.metric("RF Precision", f"{lms_rf.get('precision', 0):.1%}")
                c3.metric("RF Recall",    f"{lms_rf.get('recall', 0):.1%}")
                c4.metric("RF F1",        f"{lms_rf.get('f1', 0):.1%}")
                if lms_rf.get("auc"):
                    st.caption(f"RF AUC-ROC: {lms_rf['auc']:.4f}  ·  Method: {lms_rf.get('evaluation_method', '')}")

                st.markdown("**Feature Importance (LMS RF)**")
                fi_lms = lms_rf["feature_importance"].copy()
                fi_lms["importance %"] = (fi_lms["importance"] * 100).round(2)
                st.dataframe(fi_lms[["feature", "importance %"]], use_container_width=False)

                # XGBoost
                with st.spinner("Training LMS XGBoost…"):
                    lms_xgb = train_lms_xgboost(lms_prepared)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("XGB Accuracy",  f"{lms_xgb.get('accuracy', 0):.1%}")
                c2.metric("XGB Precision", f"{lms_xgb.get('precision', 0):.1%}")
                c3.metric("XGB Recall",    f"{lms_xgb.get('recall', 0):.1%}")
                c4.metric("XGB F1",        f"{lms_xgb.get('f1', 0):.1%}")

                st.success("LMS models trained and saved to ./models/")

        except Exception as e:
            st.error(f"LMS training failed: {e}")
        finally:
            tmp_lms.unlink(missing_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("### Explainable AI · SHAP Analysis")

    if not SHAP_AVAILABLE:
        st.warning(
            "The `shap` package is not installed. "
            "Run `pip install shap` to enable explainability features."
        )
    elif st.session_state["term_df"] is None:
        st.info("Run a prediction first (in the **Academic Predict** tab) to load student data.")
    else:
        term_df = st.session_state["term_df"]
        preds   = st.session_state["preds"]

        _EXPLAIN_FNS = {
            "Random Forest":  explain_student,
            "XGBoost":        explain_student_xgboost,
            "LSTM":           explain_student_lstm,
            "Ensemble (avg)": explain_student_ensemble,
        }

        selected_model = st.radio(
            "Model",
            list(_EXPLAIN_FNS.keys()),
            horizontal=True,
            help="LSTM and Ensemble use KernelExplainer and may take longer to compute.",
        )

        # Global importance
        st.markdown("#### Global Feature Importance")
        lstm_note = " *(capped at 100 rows for speed)*" if selected_model in ("LSTM", "Ensemble (avg)") else ""
        with st.spinner(f"Computing SHAP values for {selected_model}…{lstm_note}"):
            fig_global = plot_global_importance(
                term_df,
                model=selected_model.replace(" (avg)", ""),
            )

        if fig_global:
            st.pyplot(fig_global)
            import matplotlib.pyplot as plt
            plt.close(fig_global)
        else:
            st.warning(f"Could not compute SHAP values for {selected_model}. Ensure the model is trained.")

        st.markdown("---")

        # Per-student explanation
        st.markdown("#### Student-Level Explanation")
        if preds is not None and len(preds):
            student_options = (
                preds[["Student ID", "Student Name"]].drop_duplicates()
                .apply(lambda r: f"{r['Student ID']} — {r['Student Name']}", axis=1)
                .tolist()
            )
            selected = st.selectbox("Select a student", student_options)
            sel_id   = selected.split("—")[0].strip()
            row_idx  = term_df.index[term_df["Student ID"].astype(str) == sel_id].tolist()

            if row_idx:
                explain_fn = _EXPLAIN_FNS[selected_model]
                with st.spinner(f"Computing {selected_model} SHAP for this student…"):
                    sv = explain_fn(term_df, row_idx[0])
                if sv:
                    prow = preds[preds["Student ID"].astype(str) == sel_id].iloc[0]
                    st.markdown(
                        f"**Predicted outcome:** {prow['predicted_outcome']} &nbsp;·&nbsp;"
                        f"**Risk:** {prow['risk_tier']} &nbsp;·&nbsp;"
                        f"**Pass probability:** {prow['pass_probability']*100:.1f}%"
                    )
                    fig_wf = plot_student_waterfall(sv)
                    if fig_wf:
                        st.pyplot(fig_wf)
                        import matplotlib.pyplot as plt
                        plt.close(fig_wf)
                    else:
                        st.dataframe(
                            pd.DataFrame(sv.items(), columns=["Feature", "SHAP Value"])
                            .sort_values("SHAP Value", key=abs, ascending=False),
                            use_container_width=False,
                        )
                else:
                    st.warning(f"Could not compute {selected_model} SHAP for this student. Ensure the model is trained.")
            else:
                st.warning(f"No term data found for student {sel_id}.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATABASE RECORDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_db:
    st.markdown("### MySQL Database Records")

    if _current_role() == "viewer":
        st.warning("Database access is not available for your role.")
    elif not st.session_state.get("db_url", "").strip():
        st.info("Add a MySQL URL in the sidebar to use database features.")
    else:
        try:
            terms, classes = get_db_filter_options(st.session_state["db_url"])
            dc1, dc2, dc3 = st.columns(3)
            chosen_term   = dc1.selectbox("Term",  ["All"] + terms,   key="db_term")
            chosen_class  = dc2.selectbox("Class", ["All"] + classes, key="db_class")
            db_limit      = int(dc3.number_input("Max rows", 100, 20_000, 5_000, 100))

            if st.button("Run Predictions from Database", use_container_width=True):
                db_df = fetch_imported_records(
                    st.session_state["db_url"],
                    limit=db_limit,
                    term=None if chosen_term == "All" else chosen_term,
                    class_name=None if chosen_class == "All" else chosen_class,
                )
                if db_df.empty:
                    st.warning("No records found for the selected filters.")
                else:
                    _run_pipeline(db_df, source="mysql", save_raw=False)

            with st.expander("Preview imported records"):
                preview = fetch_imported_records(st.session_state["db_url"], limit=100)
                st.dataframe(preview, use_container_width=True)

        except Exception as e:
            st.warning(f"Database section unavailable: {e}")
