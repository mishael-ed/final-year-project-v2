from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

from .config import DEFAULT_THRESHOLDS, MODEL_DIR

# ------------------------------------------------------------------------------
# Feature definitions  (ml_student_data.csv schema)
# ------------------------------------------------------------------------------

LMS_FEATURES = [
    "lms_logins_per_semester",
    "avg_session_duration_minutes",
    "assignment_submission_rate",
    "forum_participation_count",
    "video_completion_rate",
]

LMS_FEATURE_LABELS = {
    "lms_logins_per_semester":      "LMS Logins (Semester)",
    "avg_session_duration_minutes": "Avg Session Duration (min)",
    "assignment_submission_rate":   "Assignment Submission Rate",
    "forum_participation_count":    "Forum Participation",
    "video_completion_rate":        "Video Completion Rate",
}

LMS_PREDICT_REQUIRED = [
    "lms_logins_per_semester",
    "avg_session_duration_minutes",
    "assignment_submission_rate",
    "forum_participation_count",
    "video_completion_rate",
]

# ------------------------------------------------------------------------------
# Column normalisation aliases
# ------------------------------------------------------------------------------

_ALIAS_MAP: dict[str, str] = {
    "student_id":                    "student_id",
    "studentid":                     "student_id",
    "id":                            "student_id",
    "student_name":                  "student_name",
    "studentname":                   "student_name",
    "name":                          "student_name",
    "gpa":                           "GPA",
    "lms_logins_per_semester":       "lms_logins_per_semester",
    "lms_logins":                    "lms_logins_per_semester",
    "logins_per_semester":           "lms_logins_per_semester",
    "logins":                        "lms_logins_per_semester",
    "avg_session_duration_minutes":  "avg_session_duration_minutes",
    "session_duration":              "avg_session_duration_minutes",
    "avg_session_duration":          "avg_session_duration_minutes",
    "assignment_submission_rate":    "assignment_submission_rate",
    "submission_rate":               "assignment_submission_rate",
    "forum_participation_count":     "forum_participation_count",
    "forum_participation":           "forum_participation_count",
    "forum":                         "forum_participation_count",
    "video_completion_rate":         "video_completion_rate",
    "video_completion":              "video_completion_rate",
    "pass_fail":                     "pass_fail",
}


def _normalise_lms_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapper: dict[str, str] = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        mapper[c] = _ALIAS_MAP.get(key, c.strip())
    return df.rename(columns=mapper)


# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------

def prepare_lms(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and coerce numerics. Target is pass_fail (0=Fail, 1=Pass)."""
    data = _normalise_lms_columns(df).copy()

    for col in LMS_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    if "pass_fail" in data.columns:
        data["target"] = pd.to_numeric(data["pass_fail"], errors="coerce").fillna(0).astype(int)

    return data


# ------------------------------------------------------------------------------
# Build X, y
# ------------------------------------------------------------------------------

def build_lms_xy(df: pd.DataFrame):
    x = df[LMS_FEATURES].copy()
    y = df["target"].fillna(0).astype(int)
    return x, y


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_lms_rf(df: pd.DataFrame) -> dict:
    """Train Random Forest on prepared LMS data. Returns metrics dict + feature_importance."""
    x, y = build_lms_xy(df)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred      = cross_val_predict(model, x, y, cv=cv, method="predict")
    y_prob_pass = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]

    metrics = {
        "accuracy":          float(accuracy_score(y, y_pred)),
        "precision":         float(precision_score(y, y_pred, zero_division=0)),
        "recall":            float(recall_score(y, y_pred, zero_division=0)),
        "f1":                float(f1_score(y, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y, y_prob_pass)) if y.nunique() > 1 else None,
        "evaluation_method": "stratified_5fold_cv",
        "n_samples":         int(len(y)),
    }

    model.fit(x, y)
    fi = pd.DataFrame({
        "feature":    LMS_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    joblib.dump(model, MODEL_DIR / "lms_rf_model.joblib")
    return {**metrics, "feature_importance": fi}


def train_lms_xgboost(df: pd.DataFrame) -> dict:
    """Train Gradient Boosting on prepared LMS data."""
    from sklearn.model_selection import train_test_split

    x, y = build_lms_xy(df)

    n_fail = int((y == 0).sum())
    n_pass = int((y == 1).sum())
    weight_map = {0: 1.0, 1: n_fail / max(n_pass, 1)}
    sample_weights = y.map(weight_map).values

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=4,
        random_state=42,
    )

    x_tr, x_te, y_tr, y_te, sw_tr, _ = train_test_split(
        x, y, sample_weights, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    model.fit(x_tr, y_tr, sample_weight=sw_tr)
    y_pred      = model.predict(x_te)
    y_prob_pass = model.predict_proba(x_te)[:, 1]

    metrics = {
        "accuracy":          float(accuracy_score(y_te, y_pred)),
        "precision":         float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y_te, y_prob_pass)) if y_te.nunique() > 1 else None,
        "evaluation_method": "stratified_holdout",
        "n_samples":         int(len(y)),
    }

    model.fit(x, y, sample_weight=sample_weights)
    joblib.dump(model, MODEL_DIR / "lms_xgb_model.joblib")
    return metrics


# ------------------------------------------------------------------------------
# Loading saved models
# ------------------------------------------------------------------------------

def load_lms_rf() -> RandomForestClassifier:
    return joblib.load(MODEL_DIR / "lms_rf_model.joblib")


def load_lms_xgboost() -> GradientBoostingClassifier | None:
    try:
        return joblib.load(MODEL_DIR / "lms_xgb_model.joblib")
    except Exception:
        return None


# ------------------------------------------------------------------------------
# Risk classification
# ------------------------------------------------------------------------------

def classify_risk_lms(pass_prob: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if pass_prob >= t["low"]:
        return "Low"
    if pass_prob >= t["medium"]:
        return "Medium"
    return "High"


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

def infer_lms(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """
    Run LMS-based Pass/Fail predictions.
    Input df can be raw (column names are normalised internally).
    Returns a results DataFrame with predicted_outcome, pass_probability, risk_tier.
    """
    prepared = prepare_lms(df)
    x = prepared[LMS_FEATURES].copy()

    prob_stacks: list[np.ndarray] = []
    models_used: list[str] = []

    try:
        rf = load_lms_rf()
        prob_stacks.append(rf.predict_proba(x)[:, 1])
        models_used.append("LMS-RF")
    except Exception:
        pass

    try:
        xgb = load_lms_xgboost()
        if xgb is not None:
            prob_stacks.append(xgb.predict_proba(x)[:, 1])
            models_used.append("LMS-XGB")
    except Exception:
        pass

    if not prob_stacks:
        raise RuntimeError(
            "No LMS models found in ./models/. Run train.py first."
        )

    pass_prob = np.mean(prob_stacks, axis=0)

    out = pd.DataFrame()
    out["Student ID"] = (
        prepared["student_id"].values
        if "student_id" in prepared.columns
        else [f"STU_{i+1:04d}" for i in range(len(prepared))]
    )

    if "GPA" in prepared.columns:
        out["GPA"] = prepared["GPA"].round(2).values

    if "lms_logins_per_semester" in prepared.columns:
        out["LMS Logins"] = prepared["lms_logins_per_semester"].astype(int).values

    if "avg_session_duration_minutes" in prepared.columns:
        out["Avg Session (min)"] = prepared["avg_session_duration_minutes"].astype(int).values

    if "assignment_submission_rate" in prepared.columns:
        sub = pd.to_numeric(prepared["assignment_submission_rate"], errors="coerce").fillna(0)
        out["Submission Rate"] = (sub.where(sub <= 1.0, sub / 100.0) * 100).round(1).astype(str) + "%"

    if "forum_participation_count" in prepared.columns:
        out["Forum Posts"] = prepared["forum_participation_count"].astype(int).values

    if "video_completion_rate" in prepared.columns:
        vid = pd.to_numeric(prepared["video_completion_rate"], errors="coerce").fillna(0)
        out["Video Completion"] = (vid.where(vid <= 1.0, vid / 100.0) * 100).round(1).astype(str) + "%"

    out["pass_probability"] = np.round(pass_prob, 4)
    out["predicted_outcome"] = np.where(pass_prob >= 0.5, "Pass", "Fail")
    out["risk_tier"]         = [classify_risk_lms(p, thresholds) for p in pass_prob]
    out["prediction_models"] = " + ".join(models_used)

    return out.reset_index(drop=True)
