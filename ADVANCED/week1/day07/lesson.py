# Advanced Day 07 — Model Evaluation
# ~200 MB RAM, <30s on CPU

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             classification_report)

np.random.seed(42)

bc = load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─── 1. PIPELINE ─────────────────────────────────────────────────────────────
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, random_state=42))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# ─── 2. CLASSIFICATION METRICS ───────────────────────────────────────────────
print("=== Classification Metrics ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}  (TP / (TP+FP))")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}  (TP / (TP+FN))")
print(f"F1        : {f1_score(y_test, y_pred):.4f}  (harmonic mean of P and R)")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=bc.target_names))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ─── 3. K-FOLD CROSS-VALIDATION ──────────────────────────────────────────────
print("=== Cross-Validation ===")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
print(f"10-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Scores per fold: {cv_scores.round(3)}")

# ─── 4. LEARNING CURVE ───────────────────────────────────────────────────────
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="f1"
)

# ─── 5. ROC & PR CURVES ──────────────────────────────────────────────────────
fpr, tpr, _  = roc_curve(y_test, y_prob)
prec, rec, _ = precision_recall_curve(y_test, y_prob)

# ─── 6. COMPARE MODELS ───────────────────────────────────────────────────────
models = {
    "LogReg":  pipe,
    "RF":      Pipeline([("scaler",StandardScaler()),
                          ("clf",RandomForestClassifier(50,random_state=42))]),
    "GBM":     Pipeline([("scaler",StandardScaler()),
                          ("clf",GradientBoostingClassifier(50,random_state=42))]),
}
print("\n=== Model Comparison (10-fold CV F1) ===")
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5, scoring="f1")
    print(f"  {name:<8}: {scores.mean():.4f} ± {scores.std():.4f}")

# ─── 7. VISUALISE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
axes[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j,i,cm[i,j],ha="center",va="center",fontsize=16,
                     color="white" if cm[i,j]>cm.max()/2 else "black")
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(bc.target_names)
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(bc.target_names)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual"); axes[0].set_title("Confusion Matrix")

# ROC curve
axes[1].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC={roc_auc_score(y_test,y_prob):.3f}")
axes[1].plot([0,1],[0,1],"k--"); axes[1].set_title("ROC Curve")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].legend()

# Learning curve
train_mean = train_scores.mean(axis=1); train_std = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1);   val_std   = val_scores.std(axis=1)
axes[2].plot(train_sizes, train_mean, "o-", label="Train")
axes[2].fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1)
axes[2].plot(train_sizes, val_mean, "o-", label="Validation")
axes[2].fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.1)
axes[2].set_title("Learning Curve"); axes[2].set_xlabel("Training size")
axes[2].set_ylabel("F1"); axes[2].legend()

plt.tight_layout(); plt.savefig("model_evaluation.png", dpi=80)
print("\nSaved model_evaluation.png")
