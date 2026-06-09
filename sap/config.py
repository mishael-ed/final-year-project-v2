from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

for _d in (MODEL_DIR, REPORT_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Scoring
CA_MAX_SCORE = 40.0
EXAM_MAX_SCORE = 60.0
TOTAL_MAX_SCORE = 100.0

# Risk thresholds (pass probability)
# Calibrated to the outline's rule: fail_prob > 0.70 triggers a High alert.
#   High   : pass_prob < 0.30  (fail_prob > 0.70)  → immediate intervention
#   Medium : pass_prob < 0.60  (fail_prob 0.40-0.70) → monitor & support
#   Low    : pass_prob >= 0.60 (fail_prob < 0.40)   → on track
DEFAULT_THRESHOLDS = {
    "low": 0.60,    # pass_prob >= 0.60 => Low risk
    "medium": 0.30, # pass_prob >= 0.30 => Medium risk
                    # pass_prob <  0.30 => High risk  (fail_prob > 0.70 → alert)
}

# SHAP-feature-driven intervention messages (keyed by display feature label)
INTERVENTION_MESSAGES = {
    "CA × Attendance":    "Schedule targeted CA revision sessions and address attendance barriers immediately.",
    "Avg CA (%)":         "Provide additional practice materials and CA support sessions.",
    "Failure Rate":       "Enrol student in multi-subject support programme and notify parents/guardians.",
    "Avg Attendance (%)": "Contact student and guardian regarding attendance patterns. Consider welfare counselling.",
    "Attendance Trend":   "Investigate cause of declining attendance; arrange a pastoral welfare check.",
    "CA Trend":           "Monitor for academic disengagement; consider peer-mentoring referral.",
    "Avg Overall (%)":    "Enrol in remedial tuition and assign a dedicated subject mentor.",
    "Failed Subjects":    "Trigger school-wide intervention protocol; convene teacher review meeting.",
    "Prev CA (%)":        "Review prior-term weaknesses and set targeted learning goals with the student.",
    "Min Attendance":     "Flag worst-attended subject to the relevant teacher for immediate follow-up.",
    "Low CA flag":        "Provide targeted CA support; CA average is critically below the passing threshold.",
    "Low Attendance flag":"Issue formal attendance warning and schedule a parental meeting.",
    "No major risk":      "Continue monitoring. No immediate intervention required.",
}

APP_TITLE = "AI-Powered Student Academic Performance Predictor"
APP_SUBTITLE = "Early-Warning System · Ensemble ML (RF + XGBoost + LSTM) · Explainable AI (SHAP)"
