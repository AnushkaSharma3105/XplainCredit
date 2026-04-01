"""
XplainCredit - Model Training Script
Run this ONCE to generate data, train model, and save it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

np.random.seed(42)


# GENERATE SYNTHETIC DATASET

print("Generating synthetic loan dataset...")

N = 5000

age             = np.random.randint(21, 65, N)
income          = np.random.randint(15000, 200000, N)
loan_amount     = np.random.randint(50000, 2000000, N)
loan_tenure     = np.random.choice([12, 24, 36, 48, 60], N)
employment_type = np.random.choice(["Salaried", "Self-Employed", "Business"], N)
cibil_score     = np.random.randint(300, 900, N)
existing_loans  = np.random.randint(0, 5, N)
education       = np.random.choice(["Graduate", "Post-Graduate", "Under-Graduate"], N)
loan_purpose    = np.random.choice(["Home", "Personal", "Education", "Business", "Vehicle"], N)
monthly_emi     = (loan_amount / loan_tenure) * (1 + np.random.uniform(0.08, 0.18, N) / 12)
emi_to_income   = monthly_emi / income

# Target: default probability driven by real-world logic
default_prob = (
    0.35 * (cibil_score < 600).astype(float) +
    0.25 * (emi_to_income > 0.5).astype(float) +
    0.15 * (existing_loans > 2).astype(float) +
    0.10 * (employment_type == "Self-Employed").astype(float) +
    0.10 * (income < 30000).astype(float) +
    0.05 * np.random.random(N)
)
default_prob = np.clip(default_prob, 0, 1)
target = (np.random.random(N) < default_prob).astype(int)

df = pd.DataFrame({
    "age":             age,
    "income":          income,
    "loan_amount":     loan_amount,
    "loan_tenure":     loan_tenure,
    "employment_type": employment_type,
    "cibil_score":     cibil_score,
    "existing_loans":  existing_loans,
    "education":       education,
    "loan_purpose":    loan_purpose,
    "monthly_emi":     monthly_emi.round(2),
    "emi_to_income":   emi_to_income.round(4),
    "default":         target
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/loan_data.csv", index=False)
print(f"Dataset saved → data/loan_data.csv  |  Shape: {df.shape}")
print(f"Default rate: {target.mean():.1%}\n")


# PREPROCESSING

df = pd.read_csv("data/loan_data.csv")

le_emp  = LabelEncoder()
le_edu  = LabelEncoder()
le_purp = LabelEncoder()

df["employment_type_enc"] = le_emp.fit_transform(df["employment_type"])
df["education_enc"]       = le_edu.fit_transform(df["education"])
df["loan_purpose_enc"]    = le_purp.fit_transform(df["loan_purpose"])

FEATURES = [
    "age", "income", "loan_amount", "loan_tenure",
    "cibil_score", "existing_loans", "monthly_emi", "emi_to_income",
    "employment_type_enc", "education_enc", "loan_purpose_enc"
]

X = df[FEATURES]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# TRAIN & COMPARE MODELS

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                         use_label_encoder=False, eval_metric="logloss", random_state=42)
}

results = {}
print("=" * 55)
print(f"{'Model':<22} {'AUC-ROC':>8} {'F1 Score':>9}")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    results[name] = {"model": model, "auc": auc, "f1": f1}
    print(f"{name:<22} {auc:>8.4f} {f1:>9.4f}")

print("=" * 55)


# SAVE BEST MODEL + ENCODERS

best_name  = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/xplaincredit_model.pkl")
joblib.dump(le_emp,     "model/le_employment.pkl")
joblib.dump(le_edu,     "model/le_education.pkl")
joblib.dump(le_purp,    "model/le_purpose.pkl")
joblib.dump(FEATURES,   "model/feature_names.pkl")

print("Model + encoders saved to /model/")
print("\nTraining complete! Now run:  streamlit run app.py")
