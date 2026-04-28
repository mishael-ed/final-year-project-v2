# AI-Powered Student Academic Performance Predictor

An early-warning system that accurately forecasts student academic performance and identifies students at risk of poor outcomes using an ensemble of **Random Forest**, **XGBoost**, and **LSTM** models with **SHAP**-based explainability.

---

## Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
# Optional: TensorFlow for LSTM
pip install tensorflow
```

### 2 · Generate demo data
```bash
python setup_demo_data.py
```

### 3 · Train the models
```bash
python train.py --input data/demo_train.csv
# Add --skip-lstm for faster training without deep learning
python train.py --input data/demo_train.csv --skip-lstm
```

### 4 · Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
FYP 2/
├── app.py                  # Streamlit dashboard (4 tabs)
├── train.py                # CLI training script
├── predict.py              # CLI prediction script
├── setup_demo_data.py      # Synthetic demo data generator
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark-mode theme
├── sap/                    # Core Python package
│   ├── config.py           # Constants & thresholds
│   ├── schema.py           # Column names & encodings
│   ├── io.py               # File loading & validation
│   ├── features.py         # Feature engineering
│   ├── model.py            # RF + XGBoost + LSTM training & inference
│   ├── explainer.py        # SHAP explainability
│   ├── db.py               # MySQL integration
│   └── reporting.py        # CSV & PDF export
├── models/                 # Saved model artefacts (auto-created)
├── reports/                # Generated reports (auto-created)
└── data/                   # Database & demo data (auto-created)
```

---

## Data Format

### Training file (CSV / Excel)

| Column | Required | Notes |
|---|---|---|
| Student ID | ✅ | Unique identifier |
| Term | ✅ | e.g. `2025_T1` |
| Class | ✅ | e.g. `SS2` |
| Subject | ✅ | Subject name |
| CA Score | ✅ | Continuous assessment marks |
| Classes Held | ✅ | Total classes |
| Classes Attended | ✅ | Classes the student attended |
| Behavioral Rating | ✅ | Poor / Average / Good / Excellent |
| Final Outcome | ✅ (train only) | Pass / Fail |
| Student Name | Optional | |
| Gender | Optional | Male / Female |
| CA Total Score | Optional | Defaults to 40 |
| Exam Score (%) | Optional | |

### Prediction file
Same as training but **without** `Final Outcome`.

---

## AI Architecture

```
Raw Records
    │
    ▼
Feature Engineering ──► RF Features (11 features per student-term)
    │                     [CA %, Attendance %, Behaviour, Trends, …]
    │
    ├──► Random Forest  ──────────────────────────────┐
    │                                                 │
    ├──► XGBoost (Gradient Boosting)  ────────────────┤ Ensemble Average
    │                                                 │
    └──► LSTM (Sequential Time-Series) ───────────────┘
                                                      │
                                                      ▼
                                         Pass Probability + Risk Tier
                                                      │
                                                      ▼
                                    SHAP Explainability (feature contributions)
```

### Risk Tiers
| Tier | Pass Probability |
|---|---|
| 🟢 Low | ≥ 75% |
| 🟡 Medium | 50–74% |
| 🔴 High | < 50% |

---

## Features

- **Ensemble predictions** — RF + XGBoost + LSTM averaged for maximum accuracy
- **SHAP explainability** — per-student feature contribution waterfall charts
- **Early-warning alerts** — auto-generated intervention recommendations
- **Upload or manual entry** — CSV/Excel upload or inline data editor
- **MySQL integration** — optional database persistence
- **Colour-coded reports** — downloadable CSV and PDF with risk highlighting
- **Interactive analytics** — risk distribution pie, histogram, attendance vs. CA scatter

---

## CLI Usage

```bash
# Train
python train.py --input data/my_students.csv

# Predict
python predict.py --input data/current_term.csv --output results.csv
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `MYSQL_URL` | SQLAlchemy connection string, e.g. `mysql+pymysql://user:pass@host/db` |

---

*Built for Final Year Project — Mishael Edegwa*
