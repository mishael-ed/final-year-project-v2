from __future__ import annotations

"""SHAP-based explainability for the RF model.

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
from .model import load_rf


# ──────────────────────────────────────────────────────────────────────────────
# SHAP values
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return a DataFrame of SHAP values (rows = students, cols = features).

    Returns None if SHAP is unavailable or the model is not trained.
    """
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

    # sv has shape (n_samples, n_features, 2) for binary RF; take class-1 slice
    if isinstance(sv, list):
        sv_fail = sv[1]          # probability of FAIL
    elif sv.ndim == 3:
        sv_fail = sv[:, :, 1]
    else:
        sv_fail = sv

    shap_df = pd.DataFrame(sv_fail, columns=RF_FEATURES, index=term_df.index)
    # Rename to human-readable labels
    shap_df = shap_df.rename(columns=FEATURE_LABELS)
    return shap_df


def get_feature_importance_shap(term_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Mean absolute SHAP value per feature (global importance)."""
    shap_df = compute_shap_values(term_df)
    if shap_df is None:
        return None

    importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    importance.columns = ["Feature", "Mean |SHAP|"]
    return importance


def explain_student(term_df: pd.DataFrame, student_idx: int) -> Optional[dict]:
    """SHAP contributions for a single student row.

    Returns a dict  {feature_label: shap_value}  or None.
    """
    shap_df = compute_shap_values(term_df)
    if shap_df is None or student_idx >= len(shap_df):
        return None

    row = shap_df.iloc[student_idx]
    return row.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib figures (returned as Figure objects for Streamlit)
# ──────────────────────────────────────────────────────────────────────────────

def plot_global_importance(term_df: pd.DataFrame):
    """Bar chart of mean |SHAP| across all students. Returns a matplotlib Figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    imp = get_feature_importance_shap(term_df)
    if imp is None or imp.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(imp))]
    ax.barh(imp["Feature"][::-1], imp["Mean |SHAP|"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value| (impact on FAIL probability)")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_student_waterfall(shap_values: dict, base_value: float = 0.5):
    """Simple horizontal waterfall for a single student. Returns Figure."""
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

    # Sort by absolute contribution
    order   = np.argsort(np.abs(values))[::-1][:8]   # top-8
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
