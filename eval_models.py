import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (StratifiedGroupKFold, GroupShuffleSplit,
                                     StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sap.features import build_term_features, clean_data, build_rf_xy
from sap.io import load_table, prepare
from sap.lms_model import prepare_lms, build_lms_xy

def row(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y_true, y_prob)
    except: auc = float('nan')
    print(f"  {name:<30} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.4f}")

# ---- ACADEMIC ----
print("\n=== ACADEMIC PIPELINE ===")
df = load_table('data/demo_train.csv')
prepared = prepare(df, mode='train')
clean = clean_data(prepared)
term_df = build_term_features(clean)
train_df = term_df[term_df['target_next_term_risk'].notna()].copy()
x, y = build_rf_xy(train_df)
groups = train_df['Student ID']
n_splits = min(5, int(groups.nunique()))
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
splits = list(cv.split(x, y, groups=groups))

rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
     min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
     random_state=42, n_jobs=-1)
rf_pred = cross_val_predict(rf, x, y, cv=splits, method='predict')
rf_prob = cross_val_predict(rf, x, y, cv=splits, method='predict_proba')[:,1]
row('Random Forest', y, rf_pred, rf_prob)

xgb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
      min_samples_split=5, min_samples_leaf=3, subsample=0.8, max_features='sqrt', random_state=42)
xgb_pred = np.zeros(len(y), dtype=int)
xgb_prob = np.zeros(len(y))
for tr_idx, te_idx in splits:
    sw = compute_sample_weight('balanced', y.values[tr_idx])
    xgb.fit(x.values[tr_idx], y.values[tr_idx], sample_weight=sw)
    xgb_pred[te_idx] = xgb.predict(x.values[te_idx])
    xgb_prob[te_idx] = xgb.predict_proba(x.values[te_idx])[:,1]
row('XGBoost (GBM)', y, xgb_pred, xgb_prob)

ens_prob = (rf_prob + xgb_prob) / 2
ens_pred = (ens_prob >= 0.5).astype(int)
row('Ensemble RF+XGBoost', y, ens_pred, ens_prob)

print("  -- Baselines (20% group holdout) --")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(x, y, groups=groups))
x_tr, x_te = x.iloc[tr_idx], x.iloc[te_idx]
y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
scaler = StandardScaler()
x_tr_s = scaler.fit_transform(x_tr); x_te_s = scaler.transform(x_te)
lr = LogisticRegression(max_iter=1000, random_state=42).fit(x_tr_s, y_tr)
row('Logistic Regression', y_te, lr.predict(x_te_s), lr.predict_proba(x_te_s)[:,1])
dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(x_tr, y_tr)
row('Decision Tree', y_te, dt.predict(x_te), dt.predict_proba(x_te)[:,1])
knn = KNeighborsClassifier(n_neighbors=5).fit(x_tr_s, y_tr)
row('k-NN', y_te, knn.predict(x_te_s), knn.predict_proba(x_te_s)[:,1])

# ---- LMS ----
print("\n=== LMS PIPELINE ===")
lms_df = load_table('data/ml_student_data.csv')
lms_p = prepare_lms(lms_df)
xl, yl = build_lms_xy(lms_p)

rf_lms = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
         min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample',
         random_state=42, n_jobs=-1)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
yl_pred_rf = cross_val_predict(rf_lms, xl, yl, cv=cv5, method='predict')
yl_prob_rf = cross_val_predict(rf_lms, xl, yl, cv=cv5, method='predict_proba')[:,1]
row('LMS Random Forest', yl, yl_pred_rf, yl_prob_rf)

sw = compute_sample_weight('balanced', yl)
xl_tr, xl_te, yl_tr, yl_te, sw_tr, _ = train_test_split(xl, yl, sw, test_size=0.2,
     random_state=42, stratify=yl)
xgb_lms = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
           min_samples_split=5, min_samples_leaf=3, subsample=0.8, max_features='sqrt',
           random_state=42)
xgb_lms.fit(xl_tr, yl_tr, sample_weight=sw_tr)
yl_pred_xgb = xgb_lms.predict(xl_te)
yl_prob_xgb = xgb_lms.predict_proba(xl_te)[:,1]
row('LMS XGBoost (holdout)', yl_te, yl_pred_xgb, yl_prob_xgb)

print("  -- LMS Baselines (same 20% holdout) --")
scaler2 = StandardScaler()
xl_tr_s = scaler2.fit_transform(xl_tr); xl_te_s = scaler2.transform(xl_te)
lr2 = LogisticRegression(max_iter=1000, random_state=42).fit(xl_tr_s, yl_tr)
row('LMS Logistic Regression', yl_te, lr2.predict(xl_te_s), lr2.predict_proba(xl_te_s)[:,1])
dt2 = DecisionTreeClassifier(max_depth=5, random_state=42).fit(xl_tr, yl_tr)
row('LMS Decision Tree', yl_te, dt2.predict(xl_te), dt2.predict_proba(xl_te)[:,1])
knn2 = KNeighborsClassifier(n_neighbors=5).fit(xl_tr_s, yl_tr)
row('LMS k-NN', yl_te, knn2.predict(xl_te_s), knn2.predict_proba(xl_te_s)[:,1])
