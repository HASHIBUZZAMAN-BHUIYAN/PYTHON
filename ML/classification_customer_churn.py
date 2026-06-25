"""
What it does: Predicts customer churn using synthetic telco-style data.
What it teaches: Binary classification, LogisticRegression, RandomForest,
                 GradientBoosting, ROC-AUC, precision/recall/F1, ROC curve plotting.
Category: CLASSIFICATION
RAM estimate: < 150 MB
Time estimate: < 10 seconds
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

# --- Generate synthetic customer churn data ---
np.random.seed(42)
n = 1000

tenure_months  = np.random.randint(1, 72, n).astype(float)
monthly_charge = np.random.uniform(20, 120, n)
num_products   = np.random.randint(1, 5, n).astype(float)
support_calls  = np.random.randint(0, 10, n).astype(float)

# Churn probability: high charge + low tenure + many support calls
logit = (-0.05 * tenure_months
         + 0.04 * monthly_charge
         + 0.3  * support_calls
         - 0.2  * num_products
         - 2.0)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (np.random.uniform(0, 1, n) < prob_churn).astype(int)

df = pd.DataFrame({
    "tenure_months":  tenure_months,
    "monthly_charge": monthly_charge,
    "num_products":   num_products,
    "support_calls":  support_calls,
    "churn":          churn,
})

print(f"Churn rate: {churn.mean():.2%}")

X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Models ---
models = {
    "LogisticRegression":       LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest":             RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting":         GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "LogisticRegression":
        model.fit(X_train_s, y_train)
        preds      = model.predict(X_test_s)
        proba      = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds      = model.predict(X_test)
        proba      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    auc  = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    results[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
                     "auc": auc, "fpr": fpr, "tpr": tpr}

# --- Print comparison table ---
print("\nCustomer Churn Classification - Model Comparison")
print("-" * 70)
hdr = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}"
print(hdr)
print("-" * 70)
for name, m in results.items():
    print(f"{name:<22} {m['acc']:>6.3f} {m['prec']:>6.3f} {m['rec']:>6.3f} "
          f"{m['f1']:>6.3f} {m['auc']:>7.4f}")
print("-" * 70)

# --- Plot: ROC curves ---
colors = ["steelblue", "darkorange", "green"]
fig, ax = plt.subplots(figsize=(7, 6))
for (name, m), color in zip(results.items(), colors):
    ax.plot(m["fpr"], m["tpr"], color=color, lw=2,
            label=f"{name} (AUC={m['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Customer Churn")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("ML/outputs/churn_roc.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/churn_roc.png")
