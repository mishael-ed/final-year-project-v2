from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Column names that are required for training vs. prediction
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_REQUIRED_COLUMNS = [
    "Student ID",
    "Term",
    "Class",
    "Subject",
    "CA Score",
    "Classes Held",
    "Classes Attended",
    "Behavioral Rating",
    "Final Outcome",
]

PREDICT_REQUIRED_COLUMNS = [
    "Student ID",
    "Term",
    "Class",
    "Subject",
    "CA Score",
    "Classes Held",
    "Classes Attended",
    "Behavioral Rating",
]

OPTIONAL_COLUMNS = [
    "Student Name",
    "Gender",
    "CA Total Score",
    "Exam Score",
    "Exam Score (%)",
    "Grade",
    "Attendance %",
]

# ──────────────────────────────────────────────────────────────────────────────
# Flexible header aliases (lower-case, underscored) → canonical column name
# ──────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, str] = {
    "student_id":         "Student ID",
    "student_name":       "Student Name",
    "name":               "Student Name",
    "gender":             "Gender",
    "term":               "Term",
    "class":              "Class",
    "subject":            "Subject",
    "ca_score":           "CA Score",
    "ca_total_score":     "CA Total Score",
    "ca_max_score":       "CA Total Score",
    "exam_score":         "Exam Score",
    "exam_score_(%)":     "Exam Score (%)",
    "exam_score_%":       "Exam Score (%)",
    "exam_percentage":    "Exam Score (%)",
    "total_score":        "Total Score",
    "grade":              "Grade",
    "classes_held":       "Classes Held",
    "classes_attended":   "Classes Attended",
    "attendance_pct":     "Attendance %",
    "attendance_%":       "Attendance %",
    "attendance":         "Attendance %",
    "behavioral_rating":  "Behavioral Rating",
    "behaviour_rating":   "Behavioral Rating",
    "behavior_rating":    "Behavioral Rating",
    "final_outcome":      "Final Outcome",
    "outcome":            "Final Outcome",
}

# ──────────────────────────────────────────────────────────────────────────────
# Ordinal encodings for categorical columns
# ──────────────────────────────────────────────────────────────────────────────
BEHAVIOR_ORDER: dict[str, int] = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
GRADE_ORDER:    dict[str, int] = {"F": 1, "E": 2, "D": 3, "C": 4, "B": 5, "A": 6}
OUTCOME_ORDER:  dict[str, int] = {"Fail": 0, "Pass": 1}
GENDER_ORDER:   dict[str, int] = {"Male": 0, "Female": 1}

# ──────────────────────────────────────────────────────────────────────────────
# Column type hints for imputation
# ──────────────────────────────────────────────────────────────────────────────
NUMERIC_BASE = [
    "CA Score",
    "CA Total Score",
    "Exam Score",
    "Exam Score (%)",
    "Classes Held",
    "Classes Attended",
    "Attendance %",
]

CATEGORICAL_BASE = [
    "Student ID",
    "Gender",
    "Term",
    "Class",
    "Subject",
    "Grade",
    "Behavioral Rating",
    "Final Outcome",
]
