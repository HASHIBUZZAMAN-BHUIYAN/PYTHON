# Advanced Day 05 — Intro to Machine Learning
# ~150 MB RAM, <15s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report,
                             confusion_matrix)

np.random.seed(42)

# ─── 1. THE ML WORKFLOW ───────────────────────────────────────────────────────
print("=== 1. ML Workflow ===")
print("1. Collect & clean data")
print("2. Feature engineering")
print("3. Split into train / test")
print("4. Choose & train model")
print("5. Evaluate on test set")
print("6. Tune & iterate\n")

# ─── 2. LINEAR REGRESSION ────────────────────────────────────────────────────
print("=== 2. Linear Regression ===")
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

print(f"Coefficient : {model_lr.coef_[0]:.2f}")
print(f"Intercept   : {model_lr.intercept_:.2f}")
print(f"MSE         : {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE        : {mean_squared_error(y_test, y_pred)**0.5:.2f}")
print(f"R² score    : {r2_score(y_test, y_pred):.4f}")

# ─── 3. LOGISTIC REGRESSION ──────────────────────────────────────────────────
print("\n=== 3. Logistic Regression (Iris — 2 classes) ===")
iris = load_iris()
# Use only classes 0 and 1 and 2 features for simplicity
mask = iris.target < 2
X_cls = iris.data[mask, :2]
y_cls = iris.target[mask]

X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_tr_sc, y_tr)
y_pred_cls = clf.predict(X_te_sc)

print(f"Accuracy : {accuracy_score(y_te, y_pred_cls):.4f}")
print(classification_report(y_te, y_pred_cls, target_names=["setosa","versicolor"]))
print("Confusion matrix:\n", confusion_matrix(y_te, y_pred_cls))

# ─── 4. WHAT IS SCALING? ─────────────────────────────────────────────────────
print("\n=== 4. Feature Scaling ===")
# Many models (logistic regression, SVM, KNN) are sensitive to feature scale.
# StandardScaler transforms each feature to mean=0, std=1.
raw = np.array([[100, 0.001], [200, 0.002], [50, 0.0005]])
scaled = StandardScaler().fit_transform(raw)
print("Raw    :", raw)
print("Scaled :", scaled.round(3))

# ─── 5. VISUALISE DECISION BOUNDARY ─────────────────────────────────────────
h = 0.02
x_min, x_max = X_te_sc[:, 0].min()-1, X_te_sc[:, 0].max()+1
y_min, y_max = X_te_sc[:, 1].min()-1, X_te_sc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
for cls, color, name in [(0,"blue","setosa"),(1,"red","versicolor")]:
    mask_c = y_te == cls
    ax1.scatter(X_te_sc[mask_c, 0], X_te_sc[mask_c, 1], c=color, label=name, edgecolors="k", s=50)
ax1.set_title("Logistic Regression Decision Boundary"); ax1.legend()

ax2.scatter(X_test, y_test, alpha=0.4, label="Test data")
x_line = np.linspace(X_test.min(), X_test.max(), 100)
ax2.plot(x_line, model_lr.predict(x_line.reshape(-1,1)), "r-", label="Regression line")
ax2.set_title(f"Linear Regression (R²={r2_score(y_test, y_pred):.3f})"); ax2.legend()

plt.tight_layout(); plt.savefig("ml_intro.png", dpi=80)
print("\nSaved ml_intro.png")
