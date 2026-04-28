#!/usr/bin/env python
"""
Standalone training script — trains both Academic and LMS models.

Usage:
    python train.py

Outputs (saved to ./models/):
    Academic:  rf_model.joblib, xgboost_model.joblib
    LMS:       lms_rf_model.joblib, lms_xgb_model.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURE PATHS HERE
# ──────────────────────────────────────────────────────────────────────────────
ACADEMIC_TRAIN_CSV = Path(__file__).parent / "data" / "demo_train.csv"
LMS_TRAIN_CSV      = Path(__file__).parent / "data" / "ml_student_data.csv"
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from sap.features import build_term_features, clean_data
from sap.io import load_table, prepare
from sap.lms_model import prepare_lms, train_lms_rf, train_lms_xgboost
from sap.model import train_rf, train_xgboost


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def train_academic() -> None:
    _banner("ACADEMIC MODEL TRAINING")

    if not ACADEMIC_TRAIN_CSV.exists():
        print(f"[ERROR] Training file not found: {ACADEMIC_TRAIN_CSV}")
        return

    print(f"Loading: {ACADEMIC_TRAIN_CSV}")
    df       = load_table(ACADEMIC_TRAIN_CSV)
    prepared = prepare(df, mode="train")
    clean    = clean_data(prepared)
    term_df  = build_term_features(clean)
    print(f"Records: {len(df):,} raw rows → {len(term_df):,} student-term rows")

    print("\nTraining Random Forest …")
    rf_res = train_rf(term_df)
    m = rf_res.metrics
    print(f"  Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.1%}  AUC: {m.get('auc', 0):.4f}")
    print("  Saved → models/rf_model.joblib")
    if rf_res.warnings:
        for w in rf_res.warnings:
            print(f"  [WARN] {w}")

    print("\nTraining XGBoost …")
    xgb_m = train_xgboost(term_df)
    print(f"  Accuracy: {xgb_m['accuracy']:.1%}  F1: {xgb_m['f1']:.1%}  AUC: {xgb_m.get('auc', 0):.4f}")
    print("  Saved → models/xgboost_model.joblib")

    print("\n  Top 5 features (RF):")
    for _, row in rf_res.feature_importance.head(5).iterrows():
        print(f"    {row['feature']:<35} {row['importance']*100:.2f}%")


def train_lms() -> None:
    _banner("LMS MODEL TRAINING")

    if not LMS_TRAIN_CSV.exists():
        print(f"[ERROR] LMS dataset not found: {LMS_TRAIN_CSV}")
        print("        Update LMS_TRAIN_CSV at the top of train.py.")
        return

    print(f"Loading: {LMS_TRAIN_CSV}")
    df       = load_table(LMS_TRAIN_CSV)
    prepared = prepare_lms(df)
    print(f"Records: {len(prepared):,} students")

    if "target" in prepared.columns:
        vc = prepared["target"].value_counts()
        print(f"Label distribution — Pass (Low risk): {vc.get(1, 0)}  |  Fail (Med/High): {vc.get(0, 0)}")

    print("\nTraining LMS Random Forest …")
    rf_m = train_lms_rf(prepared)
    print(f"  Accuracy: {rf_m['accuracy']:.1%}  F1: {rf_m['f1']:.1%}  AUC: {rf_m.get('auc', 0):.4f}")
    print("  Saved → models/lms_rf_model.joblib")

    if "feature_importance" in rf_m:
        print("\n  Top 5 features (LMS-RF):")
        for _, row in rf_m["feature_importance"].head(5).iterrows():
            print(f"    {row['feature']:<35} {row['importance']*100:.2f}%")

    print("\nTraining LMS XGBoost …")
    xgb_m = train_lms_xgboost(prepared)
    print(f"  Accuracy: {xgb_m['accuracy']:.1%}  F1: {xgb_m['f1']:.1%}  AUC: {xgb_m.get('auc', 0):.4f}")
    print("  Saved → models/lms_xgb_model.joblib")


if __name__ == "__main__":
    train_academic()
    train_lms()
    print("\n" + "=" * 60)
    print("  All models trained and saved to ./models/")
    print("=" * 60 + "\n")
