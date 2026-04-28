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
DEFAULT_THRESHOLDS = {
    "low": 0.75,    # pass_prob >= 0.75 => Low risk
    "medium": 0.50, # pass_prob >= 0.50 => Medium risk
                    # pass_prob <  0.50 => High risk
}

# Early-warning intervention messages
INTERVENTION_MESSAGES = {
    "Low CA": "Provide targeted CA support and additional practice materials.",
    "Low attendance": "Contact student/guardian regarding attendance. Consider counselling.",
    "Low performance": "Enrol student in remedial / extra-tuition sessions.",
    "Many failed subjects": "Trigger multi-subject support programme and parental notification.",
    "No major risk": "Continue monitoring. No immediate intervention required.",
}

APP_TITLE = "AI-Powered Student Academic Performance Predictor"
APP_SUBTITLE = "Early-Warning System · Ensemble ML (RF + XGBoost + LSTM) · Explainable AI (SHAP)"
