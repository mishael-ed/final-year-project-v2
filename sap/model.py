from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .config import DEFAULT_THRESHOLDS, MODEL_DIR
from .features import RF_FEATURES, build_rf_xy


# ──────────────────────────────────────────────────────────────────────────────
# Return type for RF training
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    metrics: dict
    feature_importance: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# LSTM helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_sequences(term_df: pd.DataFrame, seq_len: int = 2):
    """Sliding-window sequences per student for LSTM training."""
    sequences, targets = [], []
    df = term_df.sort_values(["Student ID", "Term Num"]).copy()

    for _, group in df.groupby("Student ID"):
        if len(group) < seq_len + 1:
            continue
        g = group.reset_index(drop=True)
        for i in range(len(g) - seq_len):
            seq = g.iloc[i : i + seq_len][RF_FEATURES].values
            tgt = g.iloc[i + seq_len]["target_next_term_risk"]
            if pd.notna(tgt):
                sequences.append(seq)
                targets.append(int(tgt))

    if not sequences:
        return np.array([]), np.array([])
    return np.array(sequences), np.array(targets)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_rf(term_df: pd.DataFrame) -> TrainResult:
    train_df = term_df[term_df["target_next_term_risk"].notna()].copy() \
        if "target_next_term_risk" in term_df.columns else term_df.copy()

    x, y = build_rf_xy(train_df)
    groups = train_df["Student ID"] if "Student ID" in train_df.columns else None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    warns: list[str] = []

    if groups is not None and groups.nunique() >= 4 and y.nunique() > 1:
        n_splits = min(5, int(groups.nunique()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(x, y, groups=groups))
        y_pred      = cross_val_predict(model, x, y, cv=splits, method="predict")
        y_prob_fail = cross_val_predict(model, x, y, cv=splits, method="predict_proba")[:, 1]
        y_eval = y
        eval_method = f"stratified_group_{n_splits}fold_cv"
    else:
        if groups is not None and groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, te_idx = next(gss.split(x, y, groups=groups))
            x_tr, x_te = x.iloc[tr_idx], x.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, y, test_size=0.2, random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )
        model.fit(x_tr, y_tr)
        y_pred      = model.predict(x_te)
        y_prob_fail = model.predict_proba(x_te)[:, 1]
        y_eval = y_te
        eval_method = "holdout_split"
        if y_eval.nunique() < 2:
            warns.append("Evaluation set had only one class; AUC undefined.")

    metrics = {
        "accuracy":          float(accuracy_score(y_eval, y_pred)),
        "precision":         float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_eval, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y_eval, y_prob_fail))
                             if len(np.unique(y_eval)) > 1 else None,
        "evaluation_method": eval_method,
        "n_samples":         int(len(y)),
    }

    # Fit final model on full data
    model.fit(x, y)
    fi = pd.DataFrame({"feature": RF_FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    joblib.dump(model, MODEL_DIR / "rf_model.joblib")
    return TrainResult(metrics=metrics, feature_importance=fi, warnings=warns)


def train_xgboost(term_df: pd.DataFrame) -> dict:
    train_df = term_df[term_df["target_next_term_risk"].notna()].copy() \
        if "target_next_term_risk" in term_df.columns else term_df.copy()

    x, y = build_rf_xy(train_df)
    groups = train_df["Student ID"] if "Student ID" in train_df.columns else None

    sample_weights = compute_sample_weight("balanced", y)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )

    x_arr = x.values
    y_arr = y.values

    if groups is not None and groups.nunique() >= 4 and y.nunique() > 1:
        n_splits = min(5, int(groups.nunique()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred      = np.zeros(len(y_arr), dtype=int)
        y_prob_fail = np.zeros(len(y_arr))
        for tr_idx, te_idx in cv.split(x_arr, y_arr, groups=groups):
            sw_tr = compute_sample_weight("balanced", y_arr[tr_idx])
            model.fit(x_arr[tr_idx], y_arr[tr_idx], sample_weight=sw_tr)
            y_pred[te_idx]      = model.predict(x_arr[te_idx])
            y_prob_fail[te_idx] = model.predict_proba(x_arr[te_idx])[:, 1]
        y_eval = y
    else:
        if groups is not None and groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, te_idx = next(gss.split(x_arr, y_arr, groups=groups))
        else:
            from sklearn.model_selection import train_test_split as _tts
            tr_idx, te_idx = next(
                iter(_tts(np.arange(len(y_arr)), test_size=0.2, random_state=42,
                          stratify=y_arr if y.nunique() > 1 else None))
            )
            tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
        sw_tr = compute_sample_weight("balanced", y_arr[tr_idx])
        model.fit(x_arr[tr_idx], y_arr[tr_idx], sample_weight=sw_tr)
        y_pred      = model.predict(x_arr[te_idx])
        y_prob_fail = model.predict_proba(x_arr[te_idx])[:, 1]
        y_eval      = y.iloc[te_idx]

    metrics = {
        "accuracy":          float(accuracy_score(y_eval, y_pred)),
        "precision":         float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_eval, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y_eval, y_prob_fail))
                             if len(np.unique(y_eval)) > 1 else None,
        "evaluation_method": "xgboost_cv",
        "n_samples":         int(len(y)),
    }

    model.fit(x, y, sample_weight=sample_weights)
    joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")
    return metrics


def train_lstm(term_df: pd.DataFrame) -> dict:
    if not TF_AVAILABLE:
        return {"error": "TensorFlow not installed. Run: pip install tensorflow"}

    train_df = term_df[term_df["target_next_term_risk"].notna()].copy() \
        if "target_next_term_risk" in term_df.columns else term_df.copy()

    X, y = _make_sequences(train_df, seq_len=3)

    if len(X) == 0:
        return {"error": "Insufficient data for LSTM (need students with ≥4 terms)", "n_samples": 0}

    # Scale features
    shape = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, shape[-1])).reshape(shape)

    strat_y = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_y)

    from tensorflow.keras.regularizers import l2
    n_fail = int((y == 0).sum())
    n_pass = int((y == 1).sum())
    class_weight = {0: n_pass / max(n_fail, 1), 1: 1.0}

    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(shape[1], shape[2]),
             return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        LSTM(32, activation="tanh", kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    model.fit(X_tr, y_tr, epochs=150, batch_size=16,
              validation_data=(X_te, y_te), callbacks=callbacks,
              class_weight=class_weight, verbose=0)

    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":          float(accuracy_score(y_te, y_pred)),
        "precision":         float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else None,
        "evaluation_method": "lstm_holdout",
        "n_samples":         int(len(X)),
    }

    model.save(MODEL_DIR / "lstm_model.keras")
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Loading saved models
# ──────────────────────────────────────────────────────────────────────────────

def load_rf() -> RandomForestClassifier:
    return joblib.load(MODEL_DIR / "rf_model.joblib")


def load_xgboost() -> GradientBoostingClassifier | None:
    try:
        return joblib.load(MODEL_DIR / "xgboost_model.joblib")
    except Exception:
        return None


def load_lstm():
    if not TF_AVAILABLE:
        return None, None
    try:
        mdl    = load_model(MODEL_DIR / "lstm_model.keras")
        scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
        return mdl, scaler
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Risk classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_risk(pass_prob: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if pass_prob >= t["low"]:
        return "Low"
    if pass_prob >= t["medium"]:
        return "Medium"
    return "High"


# ──────────────────────────────────────────────────────────────────────────────
# Inference  (ensemble: RF + XGBoost + LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def infer(term_df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    work = term_df.copy()
    if "Student Name" not in work.columns:
        work["Student Name"] = work["Student ID"].astype(str)
    if "avg_overall_pct" not in work.columns:
        work["avg_overall_pct"] = work.get("avg_ca_pct", 0.0)

    out = work[[
        "Student ID", "Student Name", "Term", "Class",
        "avg_ca_pct", "avg_overall_pct", "avg_attendance", "subject_failure_count",
    ]].copy()

    x = term_df[RF_FEATURES].copy()
    predictions: list[tuple[str, np.ndarray]] = []

    # Random Forest
    try:
        rf = load_rf()
        predictions.append(("RF", 1.0 - rf.predict_proba(x)[:, 1]))
    except Exception:
        pass

    # XGBoost
    try:
        xgb = load_xgboost()
        if xgb is not None:
            predictions.append(("XGBoost", 1.0 - xgb.predict_proba(x)[:, 1]))
    except Exception:
        pass

    # LSTM (single-step: reshape each row into (1, n_features))
    try:
        lstm, scaler = load_lstm()
        if lstm is not None and scaler is not None:
            x_scaled = scaler.transform(x.values)
            x_lstm   = x_scaled.reshape(x_scaled.shape[0], 1, x_scaled.shape[1])
            predictions.append(("LSTM", 1.0 - lstm.predict(x_lstm, verbose=0).flatten()))
    except Exception:
        pass

    if predictions:
        model_names  = [n for n, _ in predictions]
        pass_prob    = np.mean([p for _, p in predictions], axis=0)
        out["prediction_models"] = f"Ensemble({', '.join(model_names)})"
    else:
        # Last-resort fallback
        rf = load_rf()
        pass_prob = 1.0 - rf.predict_proba(x)[:, 1]
        out["prediction_models"] = "RF (fallback)"

    out["pass_probability"]  = pass_prob
    out["predicted_outcome"] = np.where(out["pass_probability"] >= 0.5, "Pass", "Fail")
    out["risk_tier"]         = out["pass_probability"].apply(
        lambda p: classify_risk(float(p), thresholds)
    )
    return out
