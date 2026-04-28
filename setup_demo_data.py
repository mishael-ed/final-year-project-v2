from __future__ import annotations

"""Generate realistic synthetic student data for demo/testing.

Usage:
    python setup_demo_data.py

Produces:
    data/demo_train.csv    — labelled training data (4 terms, 120 students)
    data/demo_predict.csv  — prediction data without Final Outcome (current term)
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CLASSES   = ["JSS1", "JSS2", "JSS3", "SS1", "SS2", "SS3"]
SUBJECTS  = ["Mathematics", "English Language", "Physics", "Chemistry",
             "Biology", "Geography", "Economics", "Literature",
             "Further Mathematics", "Computer Science"]
GENDERS   = ["Male", "Female"]
BEHAVIORS = ["Poor", "Average", "Good", "Excellent"]
TERMS     = ["2024_T1", "2024_T2", "2025_T1", "2025_T2", "2026_T1"]
TRAIN_TERMS   = TERMS[:4]
PREDICT_TERMS = TERMS[4:]

N_STUDENTS = 120
N_SUBJECTS = 6   # subjects per student per term


def _outcome(ca_pct: float, att_pct: float, behaviour: str) -> str:
    """Simple deterministic outcome rule for synthetic data."""
    score = ca_pct * 0.6 + att_pct * 0.3
    if behaviour == "Poor":
        score -= 10
    elif behaviour == "Excellent":
        score += 5

    noise = np.random.normal(0, 5)
    return "Pass" if (score + noise) >= 45 else "Fail"


def _build_rows(
    student_ids: list[str],
    genders: dict[str, str],
    classes: dict[str, str],
    terms: list[str],
    include_outcome: bool,
) -> pd.DataFrame:
    rows = []
    for sid in student_ids:
        gender    = genders[sid]
        cls       = classes[sid]
        # Give at-risk students lower base CA / attendance
        at_risk   = random.random() < 0.25
        ca_base   = random.uniform(18, 32) if at_risk else random.uniform(28, 38)
        att_base  = random.uniform(40, 70) if at_risk else random.uniform(65, 95)
        beh_base  = random.choice(["Poor", "Average"]) if at_risk else random.choice(["Average", "Good", "Excellent"])

        name = f"Student {sid.split('_')[1]}"

        for term in terms:
            # Add slight term-to-term variation
            ca_drift  = random.uniform(-3, 3)
            att_drift = random.uniform(-8, 8)
            behaviour = beh_base if random.random() > 0.15 else random.choice(BEHAVIORS)

            subjects = random.sample(SUBJECTS, min(N_SUBJECTS, len(SUBJECTS)))
            for subj in subjects:
                ca_score  = max(0, min(40, ca_base + ca_drift + np.random.normal(0, 2)))
                att_held  = random.randint(25, 35)
                att_att   = max(0, min(att_held, int((att_base + att_drift) / 100 * att_held)))
                att_pct   = (att_att / att_held) * 100 if att_held else 0

                row: dict = {
                    "Student ID":       sid,
                    "Student Name":     name,
                    "Gender":           gender,
                    "Term":             term,
                    "Class":            cls,
                    "Subject":          subj,
                    "CA Score":         round(ca_score, 1),
                    "CA Total Score":   40,
                    "Classes Held":     att_held,
                    "Classes Attended": att_att,
                    "Behavioral Rating": behaviour,
                }
                if include_outcome:
                    ca_pct = (ca_score / 40) * 100
                    row["Final Outcome"] = _outcome(ca_pct, att_pct, behaviour)

                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    student_ids = [f"STU_{i:04d}" for i in range(1, N_STUDENTS + 1)]
    genders     = {sid: random.choice(GENDERS) for sid in student_ids}
    classes     = {sid: random.choice(CLASSES) for sid in student_ids}

    train_df = _build_rows(student_ids, genders, classes, TRAIN_TERMS, include_outcome=True)
    pred_df  = _build_rows(student_ids, genders, classes, PREDICT_TERMS, include_outcome=False)

    train_path = DATA_DIR / "demo_train.csv"
    pred_path  = DATA_DIR / "demo_predict.csv"

    train_df.to_csv(train_path, index=False)
    pred_df.to_csv(pred_path,  index=False)

    print(f"Generated training data:   {train_path}  ({len(train_df):,} rows)")
    print(f"Generated prediction data: {pred_path}   ({len(pred_df):,} rows)")
    print()
    print("Next steps:")
    print(f"  python train.py   --input {train_path}")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
