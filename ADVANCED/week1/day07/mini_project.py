# Advanced Day 07 Mini-Project — End-to-End ML Pipeline: Churn Prediction
# ~200 MB RAM, <30s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, accuracy_score, f1_score)

np.random.seed(42)

# ─── Synthetic churn dataset ──────────────────────────────────────────────────
X, y = make_classification(
    n_samples=2000, n_features=12, n_informative=8, n_redundant=2,
    weights=[0.75, 0.25], random_state=42  # 25% churn rate
)
feature_names = [f"feature_{i}" for i in range(12)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Churn rate: {y.mean()*100:.1f}%")

# ─── Train multiple pipelines ────────────────────────────────────────────────
pipelines = {
    "LogReg":  Pipeline([("sc",StandardScaler()),
                          ("clf",LogisticRegression(class_weight="balanced",max_iter=2000))]),
    "RF":      Pipeline([("sc",StandardScaler()),
                          ("clf",RandomForestClassifier(100,class_weight="balanced",random_state=42,n_jobs=-1))]),
    "GBM":     Pipeline([("sc",StandardScaler()),
                          ("clf",GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,random_state=42))]),
}

print("\n=== Cross-Validation Results (5-fold, AUC-ROC) ===")
cv_results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<8}: {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_pipe  = pipelines[best_name]
best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

print(f"\n=== Test Results ({best_name}) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1       : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["Stay","Churn"]))

# ─── Visualise ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Churn Prediction — {best_name}", fontsize=13)

# ROC Curve (all models)
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0,1],[0,1],"k--",linewidth=0.8)
axes[0].set_title("ROC Curves"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend(fontsize=9)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
axes[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1].text(j,i,cm[i,j],ha="center",va="center",fontsize=16,
                     color="white" if cm[i,j]>cm.max()/2 else "black")
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Stay","Churn"])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["Stay","Churn"])
axes[1].set_title("Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

# Feature importances (RF)
rf_pipe = pipelines["RF"]
rf_model = rf_pipe.named_steps["clf"]
imp = sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x:-x[1])[:8]
axes[2].barh([i[0] for i in imp], [i[1] for i in imp], color="steelblue", edgecolor="white")
axes[2].set_title("Feature Importances (RF)"); axes[2].set_xlabel("Importance")

plt.tight_layout(); plt.savefig("churn_pipeline.png", dpi=80)
print("\nSaved churn_pipeline.png")
plt.show()
