# ML Reference — Linear Regression Pipeline Template
# Copy & modify for any regression task.
# ~20 MB RAM, <2s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── 1. DATA ─────────────────────────────────────────────────────────────────
# Replace with your own data:
# df = pd.read_csv("your_data.csv")
# X = df[["feature1","feature2"]].values
# y = df["target"].values

X, y = make_regression(n_samples=500, n_features=5, noise=15.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── 2. PIPELINES ────────────────────────────────────────────────────────────
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    LinearRegression()),
    ]),
    "Ridge(a=1)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ]),
    "Lasso(a=0.1)": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Lasso(alpha=0.1, max_iter=5000)),
    ]),
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ]),
    "Polynomial(deg=2)": Pipeline([
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ]),
}

# ─── 3. TRAIN & EVALUATE ────────────────────────────────────────────────────
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>6} {'CV-R²':>10}")
print("-" * 58)
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
    print(f"{name:<22} {rmse:>8.2f} {mae:>8.2f} {r2:>6.3f} {cv:>10.3f}")
    results[name] = {"y_pred": y_pred, "rmse": rmse, "r2": r2}

# ─── 4. DIAGNOSTIC PLOTS ─────────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["rmse"])
best_pred = results[best_name]["y_pred"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(y_test, best_pred, alpha=0.5, s=20)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5)
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"{best_name}: Actual vs Predicted")

residuals = y_test - best_pred
axes[1].scatter(best_pred, residuals, alpha=0.5, s=20)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

names = list(results.keys()); rmses = [results[n]["rmse"] for n in names]
bars = axes[2].barh(names, rmses, color="steelblue")
axes[2].set_xlabel("RMSE"); axes[2].set_title("Model Comparison (lower=better)")
for bar, v in zip(bars, rmses):
    axes[2].text(v+0.5, bar.get_y()+bar.get_height()/2, f"{v:.1f}", va="center", fontsize=8)

plt.tight_layout(); plt.savefig("ml_regression.png", dpi=80)
print(f"\nBest model: {best_name}  RMSE={results[best_name]['rmse']:.2f}  R²={results[best_name]['r2']:.3f}")
print("Saved ml_regression.png")
