from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .io import normalize_columns


# ──────────────────────────────────────────────────────────────────────────────
# Column maps (app column name → DB column name)
# ──────────────────────────────────────────────────────────────────────────────

RAW_RENAME = {
    "Student ID":    "student_id",
    "Student Name":  "student_name",
    "Gender":        "gender",
    "Term":          "term",
    "Class":         "class_name",
    "Subject":       "subject",
    "CA Score":      "ca_score",
    "CA Total Score":"ca_total_score",
    "Exam Score":    "exam_score",
    "Exam Score (%)":"exam_score_pct",
    "Grade":         "grade",
    "Classes Held":  "classes_held",
    "Classes Attended": "classes_attended",
    "Attendance %":  "attendance_pct",
    "Behavioral Rating": "behavioral_rating",
    "Final Outcome": "final_outcome",
}

PRED_RENAME = {
    "Student ID":         "student_id",
    "Student Name":       "student_name",
    "Term":               "term",
    "Class":              "class_name",
    "avg_ca_pct":         "avg_ca_pct",
    "avg_overall_pct":    "avg_overall_pct",
    "avg_attendance":     "avg_attendance",
    "subject_failure_count": "subject_failure_count",
    "pass_probability":   "pass_probability",
    "predicted_outcome":  "predicted_outcome",
    "risk_tier":          "risk_tier",
    "main_issue":         "main_issue",
}


# ──────────────────────────────────────────────────────────────────────────────
# Engine / table setup
# ──────────────────────────────────────────────────────────────────────────────

def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)


def ensure_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS imported_records (
                id              BIGINT AUTO_INCREMENT PRIMARY KEY,
                student_id      VARCHAR(128),
                student_name    VARCHAR(255),
                gender          VARCHAR(32),
                term            VARCHAR(64),
                class_name      VARCHAR(64),
                subject         VARCHAR(128),
                ca_score        DOUBLE,
                ca_total_score  DOUBLE,
                exam_score      DOUBLE,
                exam_score_pct  DOUBLE,
                grade           VARCHAR(16),
                classes_held    DOUBLE,
                classes_attended DOUBLE,
                attendance_pct  DOUBLE,
                behavioral_rating VARCHAR(32),
                final_outcome   VARCHAR(16),
                source_label    VARCHAR(64),
                ingested_at     DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_id               VARCHAR(64),
                student_id           VARCHAR(128),
                student_name         VARCHAR(255),
                term                 VARCHAR(64),
                class_name           VARCHAR(64),
                avg_ca_pct           DOUBLE,
                avg_overall_pct      DOUBLE,
                avg_attendance       DOUBLE,
                subject_failure_count INT,
                pass_probability     DOUBLE,
                predicted_outcome    VARCHAR(16),
                risk_tier            VARCHAR(16),
                main_issue           VARCHAR(255),
                source_label         VARCHAR(64),
                created_at           DATETIME
            )
        """))


def test_connection(db_url: str) -> None:
    engine = get_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    ensure_tables(engine)


# ──────────────────────────────────────────────────────────────────────────────
# Save / fetch
# ──────────────────────────────────────────────────────────────────────────────

def _prep_raw(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_columns(df.copy())
    cols = [c for c in RAW_RENAME if c in work.columns]
    out  = work[cols].rename(columns={c: RAW_RENAME[c] for c in cols})
    out["ingested_at"] = datetime.utcnow()
    return out


def save_imported_records(df: pd.DataFrame, db_url: str, source_label: str = "") -> int:
    engine = get_engine(db_url)
    ensure_tables(engine)
    out = _prep_raw(df)
    out["source_label"] = source_label
    out.to_sql("imported_records", engine, if_exists="append", index=False)
    return len(out)


def save_prediction_results(preds: pd.DataFrame, db_url: str, source_label: str = "") -> str:
    engine = get_engine(db_url)
    ensure_tables(engine)
    run_id = str(uuid4())
    cols   = [c for c in PRED_RENAME if c in preds.columns]
    out    = preds[cols].rename(columns={c: PRED_RENAME[c] for c in cols}).copy()
    out["run_id"]       = run_id
    out["source_label"] = source_label
    out["created_at"]   = datetime.utcnow()
    out.to_sql("prediction_runs", engine, if_exists="append", index=False)
    return run_id


def fetch_imported_records(
    db_url: str,
    limit: int = 5000,
    term: str | None = None,
    class_name: str | None = None,
) -> pd.DataFrame:
    engine = get_engine(db_url)
    ensure_tables(engine)
    where_parts = []
    params: dict = {"lim": limit}
    if term:
        where_parts.append("term = :term")
        params["term"] = term
    if class_name:
        where_parts.append("class_name = :class_name")
        params["class_name"] = class_name
    where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = text(f"SELECT * FROM imported_records {where} ORDER BY ingested_at DESC LIMIT :lim")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)

    # Map DB columns back to app column names
    inv = {v: k for k, v in RAW_RENAME.items()}
    return df.rename(columns=inv)


def get_db_filter_options(db_url: str) -> tuple[list[str], list[str]]:
    engine = get_engine(db_url)
    ensure_tables(engine)
    with engine.connect() as conn:
        terms   = [r[0] for r in conn.execute(text("SELECT DISTINCT term FROM imported_records ORDER BY term"))]
        classes = [r[0] for r in conn.execute(text("SELECT DISTINCT class_name FROM imported_records ORDER BY class_name"))]
    return terms, classes
