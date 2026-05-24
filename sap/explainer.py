from __future__ import annotations

"""SHAP-based explainability for the full ensemble (RF, XGBoost, LSTM).

SHAP is only computed when the `shap` package is available. All public
functions gracefully return None / empty structures when SHAP is absent so
the rest of the app continues to work without it.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .features import FEATURE_LABELS, RF_FEATURES
from .model import load_lstm, load_rf, load_xgboost


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_labeled_df(sv_array: np.ndarray, index) -> pd.DataFrame:
    df = pd.DataFrame(sv_array, columns=RF_FEATURES, index=index)
    return df.rename(columns=FEATURE_LABELS)


# ──────────────────────────────────────────────────────────────────────────────
# Random Forest  (TreeExplainer)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """SHAP values for the RF model. Rows = students, cols = features."""
    if not SHAP_AVAILABLE:
        return None
    try:
        rf = load_rf()
    except Exception:
        return None

    x = term_df[RF_FEATURES].copy()
    explainer = shap.TreeExplainer(rf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(x)

    # Binary RF: sv is list[class0, class1] or ndarray (n, f, 2)
    if isinstance(sv, list):
        sv_fail = sv[1]
    elif sv.ndim == 3:
        sv_fail = sv[:, :, 1]
    else:
        sv_fail = sv

    return _to_labeled_df(sv_fail, term_df.index)


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost / GradientBoostingClassifier  (TreeExplainer)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values_xgboost(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """SHAP values for the XGBoost (GBM) model."""
    if not SHAP_AVAILABLE:
        return None
    try:
        xgb = load_xgboost()
        if xgb is None:
            return None
    except Exception:
        return None

    x = term_df[RF_FEATURES].copy()
    explainer = shap.TreeExplainer(xgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(x)

    # sklearn GBM binary returns a single 2-D array
    if isinstance(sv, list):
        sv_fail = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv_fail = sv[:, :, 1]
    else:
        sv_fail = sv

    return _to_labeled_df(sv_fail, term_df.index)


# ──────────────────────────────────────────────────────────────────────────────
# LSTM  (KernelExplainer — model-agnostic, slower)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values_lstm(
    term_df: pd.DataFrame,
    n_background: int = 50,
    nsamples: int = 100,
    max_explain: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """SHAP values for the LSTM via KernelExplainer.

    max_explain caps the number of rows explained (use for global importance
    to keep runtime reasonable).
    """
    if not SHAP_AVAILABLE:
        return None
    try:
        lstm, scaler = load_lstm()
        if lstm is None or scaler is None:
            return None
    except Exception:
        return None

    x_raw = term_df[RF_FEATURES].values.astype(float)
    x_scaled = scaler.transform(x_raw)

    def _predict(x_2d: np.ndarray) -> np.ndarray:
        x_3d = x_2d.reshape(x_2d.shape[0], 1, x_2d.shape[1])
        return 1.0 - lstm.predict(x_3d, verbose=0).flatten()  # FAIL probability

    bg_size = min(n_background, len(x_scaled))
    background = x_scaled[:bg_size]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(_predict, background)

        explain_x = x_scaled if max_explain is None else x_scaled[:max_explain]
        sv = explainer.shap_values(explain_x, nsamples=nsamples)

    idx = term_df.index if max_explain is None else term_df.index[:max_explain]
    return _to_labeled_df(sv, idx)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble average
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values_ensemble(
    term_df: pd.DataFrame,
    lstm_max_explain: int = 100,
) -> Optional[pd.DataFrame]:
    """Mean SHAP values across whichever models are available."""
    frames = []
    rf_shap = compute_shap_values(term_df)
    if rf_shap is not None:
        frames.append(rf_shap)

    xgb_shap = compute_shap_values_xgboost(term_df)
    if xgb_shap is not None:
        frames.append(xgb_shap)

    lstm_shap = compute_shap_values_lstm(term_df, max_explain=lstm_max_explain)
    if lstm_shap is not None:
        # Align index to common rows
        common_idx = frames[0].index if frames else lstm_shap.index
        frames.append(lstm_shap.reindex(common_idx))

    if not frames:
        return None

    # Average element-wise across available models
    return pd.concat(frames).groupby(level=0).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Global importance  (mean |SHAP|)
# ──────────────────────────────────────────────────────────────────────────────

def _importance_from_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    imp.columns = ["Feature", "Mean |SHAP|"]
    return imp


def get_feature_importance_shap(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    shap_df = compute_shap_values(term_df)
    return _importance_from_shap(shap_df) if shap_df is not None else None


def get_feature_importance_shap_xgboost(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    shap_df = compute_shap_values_xgboost(term_df)
    return _importance_from_shap(shap_df) if shap_df is not None else None


def get_feature_importance_shap_lstm(
    term_df: pd.DataFrame, max_explain: int = 100
) -> Optional[pd.DataFrame]:
    shap_df = compute_shap_values_lstm(term_df, max_explain=max_explain)
    return _importance_from_shap(shap_df) if shap_df is not None else None


def get_feature_importance_shap_ensemble(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    shap_df = compute_shap_values_ensemble(term_df)
    return _importance_from_shap(shap_df) if shap_df is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# Per-student explanation
# ──────────────────────────────────────────────────────────────────────────────

def explain_student(term_df: pd.DataFrame, student_idx: int) -> Optional[dict]:
    shap_df = compute_shap_values(term_df)
    if shap_df is None or student_idx >= len(shap_df):
        return None
    return shap_df.iloc[student_idx].to_dict()


def explain_student_xgboost(term_df: pd.DataFrame, student_idx: int) -> Optional[dict]:
    shap_df = compute_shap_values_xgboost(term_df)
    if shap_df is None or student_idx >= len(shap_df):
        return None
    return shap_df.iloc[student_idx].to_dict()


def explain_student_lstm(term_df: pd.DataFrame, student_idx: int) -> Optional[dict]:
    row_df = term_df.iloc[[student_idx]]
    shap_df = compute_shap_values_lstm(row_df, n_background=min(50, len(term_df)), nsamples=100)
    if shap_df is None:
        return None
    return shap_df.iloc[0].to_dict()


def explain_student_ensemble(term_df: pd.DataFrame, student_idx: int) -> Optional[dict]:
    results = []
    for fn in (explain_student, explain_student_xgboost, explain_student_lstm):
        sv = fn(term_df, student_idx)
        if sv:
            results.append(sv)
    if not results:
        return None
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results if k in r])) for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib figures
# ──────────────────────────────────────────────────────────────────────────────

def _bar_figure(imp: pd.DataFrame, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if imp is None or imp.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(imp))]
    ax.barh(imp["Feature"][::-1], imp["Mean |SHAP|"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value| (impact on FAIL probability)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_global_importance(term_df: pd.DataFrame, model: str = "RF"):
    """Bar chart of mean |SHAP|. model: 'RF', 'XGBoost', 'LSTM', 'Ensemble'."""
    if model == "XGBoost":
        imp = get_feature_importance_shap_xgboost(term_df)
        title = "Global Feature Importance — XGBoost (SHAP)"
    elif model == "LSTM":
        imp = get_feature_importance_shap_lstm(term_df)
        title = "Global Feature Importance — LSTM (SHAP)"
    elif model == "Ensemble":
        imp = get_feature_importance_shap_ensemble(term_df)
        title = "Global Feature Importance — Ensemble Average (SHAP)"
    else:
        imp = get_feature_importance_shap(term_df)
        title = "Global Feature Importance — Random Forest (SHAP)"
    return _bar_figure(imp, title)


def plot_student_waterfall(shap_values: dict, base_value: float = 0.5):
    """Horizontal waterfall chart for a single student."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not shap_values:
        return None

    features = list(shap_values.keys())
    values   = list(shap_values.values())

    order    = np.argsort(np.abs(values))[::-1][:8]
    f_sorted = [features[i] for i in order]
    v_sorted = [values[i]   for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in v_sorted]
    ax.barh(f_sorted[::-1], v_sorted[::-1], color=bar_colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (positive = increases FAIL probability)")
    ax.set_title("Student-level Explanation (SHAP)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
