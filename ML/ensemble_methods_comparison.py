"""
What it does: Compares ensemble methods (Bagging, RandomForest, GradientBoosting,
              AdaBoost) against a baseline DecisionTree using 5-fold cross-validation.
What it teaches: Bagging vs boosting concepts, variance reduction, bias-variance
                 tradeoff in ensembles, CV accuracy with error bars.
Category: ENSEMBLE METHODS
RAM estimate: < 150 MB
Time estimate: < 20 seconds
"""

# Bagging (Bootstrap Aggregating):
#   Train many independent models on bootstrap samples; average/vote.
#   Reduces VARIANCE -> good when base model overfits.
#   Examples: BaggingClassifier, RandomForest.
#
# Boosting:
#   Train models sequentially; each focuses on errors of the previous.
#   Reduces BIAS -> good when base model underfits.
#   Examples: GradientBoosting, AdaBoost.

import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

# --- Dataset ---
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
print(f"Dataset: {X.shape}, class balance: {y.mean():.2f}")

# --- Models ---
base_dt = DecisionTreeClassifier(random_state=42)

models = {
    "DecisionTree (base)":   DecisionTreeClassifier(random_state=42),
    "Bagging(DT, n=50)":     BaggingClassifier(estimator=DecisionTreeClassifier(),
                                               n_estimators=50, random_state=42, n_jobs=-1),
    "RandomForest(n=100)":   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradBoost(n=100)":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost(n=100)":       AdaBoostClassifier(n_estimators=100, random_state=42),
}

# --- 5-fold CV ---
cv_results = {}
print("\nRunning 5-fold CV (this may take 15-20s)...")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<26} {scores.mean():.4f} +/- {scores.std():.4f}")

# --- Print table ---
print("\nEnsemble Methods - 5-Fold CV Accuracy")
print("-" * 50)
print(f"{'Model':<26} {'Mean':>8} {'Std':>8}")
print("-" * 50)
for name, scores in cv_results.items():
    print(f"{name:<26} {scores.mean():>8.4f} {scores.std():>8.4f}")
print("-" * 50)

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"Best model: {best_name} (mean acc={cv_results[best_name].mean():.4f})")

# --- Bar chart with error bars ---
names   = list(cv_results.keys())
means   = [cv_results[n].mean() for n in names]
stds    = [cv_results[n].std()  for n in names]
x_pos   = np.arange(len(names))

colors_bar = ["grey", "steelblue", "darkorange", "green", "red"]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_pos, means, yerr=stds, color=colors_bar, alpha=0.75,
              capsize=5, error_kw={"elinewidth": 1.5})

ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("CV Accuracy")
ax.set_title("Ensemble Methods - 5-Fold CV Accuracy (+/- 1 std)")
ax.set_ylim(0.7, 1.0)

# Annotate bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("ML/outputs/ensemble_comparison.png", dpi=100)
plt.close()
print("Plot saved -> ML/outputs/ensemble_comparison.png")
