import sys
sys.path.insert(0, ".")
import pandas as pd
from sap.lms_model import prepare_lms, train_lms_rf, train_lms_xgboost

print("Loading dataset...")
df = pd.read_csv("data/ml_student_data.csv")
print(f"  {len(df)} rows, columns: {list(df.columns)}")

prepared = prepare_lms(df)
vc = prepared["target"].value_counts().to_dict()
print(f"  target distribution: {vc}")

print("Training RF...")
m = train_lms_rf(prepared)
print(f"  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUC={m['auc']:.4f}")
fi = m["feature_importance"]
for _, row in fi.iterrows():
    print(f"    {row['feature']:40s}  {row['importance']*100:.2f}%")

print("Training XGBoost...")
xm = train_lms_xgboost(prepared)
print(f"  Acc={xm['accuracy']:.3f}  F1={xm['f1']:.3f}  AUC={xm['auc']:.4f}")
print("Done.")
