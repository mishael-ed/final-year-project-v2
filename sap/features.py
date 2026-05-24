from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import CA_MAX_SCORE, EXAM_MAX_SCORE
from .schema import (
    BEHAVIOR_ORDER,
    CATEGORICAL_BASE,
    GENDER_ORDER,
    GRADE_ORDER,
    NUMERIC_BASE,
    OUTCOME_ORDER,
)

# ──────────────────────────────────────────────────────────────────────────────
# Feature names used by tree-based models
# ──────────────────────────────────────────────────────────────────────────────
RF_FEATURES = [
    "Gender Enc",
    "avg_ca_pct",
    "avg_overall_pct",
    "avg_attendance",
    "min_attendance",
    "avg_behavior",
    "subject_count",
    "subject_failure_count",
    "attendance_trend",
    "ca_trend",
    "Term Num",
    # Engineered features
    "failure_rate",
    "low_attendance_flag",
    "low_ca_flag",
    "prev_ca_pct",
    "prev_attendance",
    "ca_x_attendance",
]

FEATURE_LABELS = {
    "Gender Enc":             "Gender",
    "avg_ca_pct":             "Avg CA (%)",
    "avg_overall_pct":        "Avg Overall (%)",
    "avg_attendance":         "Avg Attendance (%)",
    "min_attendance":         "Min Attendance (%)",
    "avg_behavior":           "Avg Behaviour Score",
    "subject_count":          "No. of Subjects",
    "subject_failure_count":  "Subjects at Risk (CA<50%)",
    "attendance_trend":       "Attendance Trend",
    "ca_trend":               "CA Score Trend",
    "Term Num":               "Term Number",
    "failure_rate":           "Subject Failure Rate",
    "low_attendance_flag":    "Low Attendance (<60%)",
    "low_ca_flag":            "Low CA Score (<60%)",
    "prev_ca_pct":            "Prev Term CA (%)",
    "prev_attendance":        "Prev Term Attendance (%)",
    "ca_x_attendance":        "CA × Attendance (Interaction)",
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – clean raw record-level data
# ──────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    numeric_cols = [c for c in NUMERIC_BASE if c in data.columns]
    cat_cols = [c for c in CATEGORICAL_BASE if c in data.columns]

    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        if data[c].isna().all():
            data[c] = 0.0

    if numeric_cols:
        data[numeric_cols] = SimpleImputer(strategy="median").fit_transform(data[numeric_cols])

    if cat_cols:
        for c in cat_cols:
            if data[c].isna().all():
                data[c] = "Unknown"
        data[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data[cat_cols])

    # Ordinal encodings
    data["Gender Enc"] = data["Gender"].map(GENDER_ORDER).fillna(0) if "Gender" in data.columns else 0
    data["Behavioral Enc"] = (
        data["Behavioral Rating"].map(BEHAVIOR_ORDER).fillna(2) if "Behavioral Rating" in data.columns else 2
    )
    data["Grade Enc"] = data["Grade"].map(GRADE_ORDER).fillna(3) if "Grade" in data.columns else 3
    data["Outcome Enc"] = (
        data["Final Outcome"].map(OUTCOME_ORDER).fillna(1) if "Final Outcome" in data.columns else 1
    )

    # Numeric term index
    term_num = (
        pd.to_numeric(
            data["Term"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
        )
        if "Term" in data.columns
        else pd.Series([0] * len(data))
    )
    data["Term Num"] = term_num.fillna(0)

    # Attendance %
    if "Attendance %" not in data.columns and {"Classes Held", "Classes Attended"}.issubset(data.columns):
        held = pd.to_numeric(data["Classes Held"], errors="coerce").replace(0, np.nan)
        att = pd.to_numeric(data["Classes Attended"], errors="coerce")
        data["Attendance %"] = ((att / held) * 100.0).fillna(0.0)

    # CA %
    if "CA Score" in data.columns:
        if "CA Total Score" in data.columns:
            ca_total = pd.to_numeric(data["CA Total Score"], errors="coerce")
            ca_total = ca_total.where(ca_total > 0, CA_MAX_SCORE).fillna(CA_MAX_SCORE)
            data["CA %"] = (data["CA Score"] / ca_total) * 100.0
        else:
            data["CA %"] = (data["CA Score"] / CA_MAX_SCORE) * 100.0
    else:
        data["CA %"] = 0.0

    # Exam %
    if "Exam Score (%)" in data.columns:
        data["Exam %"] = pd.to_numeric(data["Exam Score (%)"], errors="coerce").fillna(0.0)
    elif "Exam Score" in data.columns:
        data["Exam %"] = (
            pd.to_numeric(data["Exam Score"], errors="coerce").fillna(0.0) / EXAM_MAX_SCORE
        ) * 100.0
    else:
        data["Exam %"] = 0.0

    # Overall %: blend CA + exam where exam data is present, otherwise use CA only
    has_exam = data["Exam %"] > 0
    data["Overall %"] = np.where(
        has_exam,
        (data["CA %"] * 0.6) + (data["Exam %"] * 0.4),
        data["CA %"],
    )

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – aggregate to one row per (student, term)
# ──────────────────────────────────────────────────────────────────────────────

def build_term_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (student_id, term), g in df.groupby(["Student ID", "Term"], dropna=False):
        row: dict = {
            "Student ID":            student_id,
            "Student Name":          g["Student Name"].iloc[0] if "Student Name" in g.columns else str(student_id),
            "Term":                  term,
            "Class":                 g["Class"].iloc[0],
            "Gender Enc":            float(g["Gender Enc"].iloc[0]),
            "Term Num":              float(g["Term Num"].iloc[0]),
            # Performance aggregates
            "avg_ca":                float(g["CA Score"].mean()),
            "avg_ca_pct":            float(g["CA %"].mean()),
            "avg_exam_pct":          float(g["Exam %"].mean()),
            "avg_overall_pct":       float(g["Overall %"].mean()),
            # Attendance
            "avg_attendance":        float(g["Attendance %"].mean()),
            "min_attendance":        float(g["Attendance %"].min()),
            # Behaviour
            "avg_behavior":          float(g["Behavioral Enc"].mean()),
            # Subject stats
            "subject_count":         int(g["Subject"].nunique()),
            "subject_failure_count": int((g["CA %"] < 50).sum()),
            # Risk flag (from actual outcome, used as label during training)
            "at_risk": int((g["Outcome Enc"] == 0).any()) if "Outcome Enc" in g.columns else 0,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["Student ID", "Term Num", "Term"])

    # Trend features: difference from student's first recorded term
    out["attendance_trend"] = out.groupby("Student ID")["avg_attendance"].transform(
        lambda s: s - s.iloc[0]
    )
    out["ca_trend"] = out.groupby("Student ID")["avg_ca_pct"].transform(
        lambda s: s - s.iloc[0]
    )

    # Engineered features
    out["failure_rate"] = out["subject_failure_count"] / out["subject_count"].clip(lower=1)
    out["low_attendance_flag"] = (out["avg_attendance"] < 60).astype(float)
    out["low_ca_flag"] = (out["avg_ca_pct"] < 60).astype(float)
    # Lag features: previous term's CA and attendance (fill first term with own value = no change)
    out["prev_ca_pct"] = (
        out.groupby("Student ID")["avg_ca_pct"].shift(1)
        .fillna(out["avg_ca_pct"])
    )
    out["prev_attendance"] = (
        out.groupby("Student ID")["avg_attendance"].shift(1)
        .fillna(out["avg_attendance"])
    )
    # Interaction: high CA + high attendance = strongly passing; either low = risk
    out["ca_x_attendance"] = (out["avg_ca_pct"] * out["avg_attendance"]) / 10_000.0

    # Forward-shifted label: current-term features → next-term risk
    out["target_next_term_risk"] = out.groupby("Student ID")["at_risk"].shift(-1)

    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Build X, y for tree models
# ──────────────────────────────────────────────────────────────────────────────

def build_rf_xy(term_df: pd.DataFrame):
    x = term_df[RF_FEATURES].copy()
    if "target_next_term_risk" in term_df.columns:
        y = term_df["target_next_term_risk"]
    else:
        y = term_df["at_risk"]
    y = y.fillna(0).astype(int)
    return x, y
