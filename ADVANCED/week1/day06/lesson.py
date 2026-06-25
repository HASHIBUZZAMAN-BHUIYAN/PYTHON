# Advanced Day 06 — Classical ML Models
# ~200 MB RAM, <30s on CPU

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ─── Dataset (Wine — 178 samples, 13 features, 3 classes) ────────────────────
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s  = scaler.transform(X_test)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# ─── 1. DECISION TREE ────────────────────────────────────────────────────────
print("\n=== Decision Tree ===")
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)  # trees don't need scaling
y_pred_dt = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Max depth used: {dt.get_depth()}")
print("Feature importances (top 5):")
imp = sorted(zip(wine.feature_names, dt.feature_importances_), key=lambda x:-x[1])[:5]
for name, score in imp:
    print(f"  {name:<25}: {score:.4f}")

# ─── 2. RANDOM FOREST ────────────────────────────────────────────────────────
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# ─── 3. KNN ──────────────────────────────────────────────────────────────────
print("\n=== K-Nearest Neighbors ===")
results_knn = {}
for k in [1, 3, 5, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr_s, y_train)
    acc = accuracy_score(y_test, knn.predict(X_te_s))
    results_knn[k] = acc
    print(f"  k={k}: accuracy={acc:.4f}")

best_k = max(results_knn, key=results_knn.get)
print(f"Best k={best_k} with accuracy={results_knn[best_k]:.4f}")

# ─── 4. SVM ──────────────────────────────────────────────────────────────────
print("\n=== Support Vector Machine ===")
for kernel in ["linear", "rbf", "poly"]:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_tr_s, y_train)
    acc = accuracy_score(y_test, svm.predict(X_te_s))
    print(f"  kernel={kernel}: accuracy={acc:.4f}")

# ─── 5. COMPARISON ───────────────────────────────────────────────────────────
print("\n=== Model Comparison ===")
models_to_compare = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5),
    "SVM (rbf)":     SVC(kernel="rbf", random_state=42),
}
comparison = {}
for name, m in models_to_compare.items():
    needs_scaling = name in ("KNN (k=5)", "SVM (rbf)")
    Xt, Xe = (X_tr_s, X_te_s) if needs_scaling else (X_train, X_test)
    m.fit(Xt, y_train)
    acc = accuracy_score(y_test, m.predict(Xe))
    comparison[name] = acc
    print(f"  {name:<20}: {acc:.4f}")

# ─── 6. GRID SEARCH ──────────────────────────────────────────────────────────
print("\n=== Grid Search on Random Forest ===")
param_grid = {"n_estimators": [20, 50], "max_depth": [3, 5, None]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Best CV accuracy: {gs.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, gs.predict(X_test)):.4f}")

# ─── 7. VISUALISE ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot decision tree (only first 2 levels for readability)
plot_tree(dt, max_depth=2, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True, rounded=True, ax=axes[0])
axes[0].set_title("Decision Tree (depth ≤ 2)")

# Model comparison bar chart
names = list(comparison.keys())
accs  = list(comparison.values())
bars  = axes[1].barh(names, accs, color="steelblue", edgecolor="white")
axes[1].set_xlim(0.8, 1.02)
axes[1].bar_label(bars, fmt="%.3f", padding=3)
axes[1].set_title("Model Accuracy Comparison"); axes[1].set_xlabel("Accuracy")

plt.tight_layout(); plt.savefig("classical_ml.png", dpi=80)
print("\nSaved classical_ml.png")
