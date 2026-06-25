# Advanced Day 05 Mini-Project — House Price Predictor
# ~150 MB RAM, <20s on CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# Use California housing — small enough for CPU
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")
print(df.describe().round(2))

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s  = scaler.transform(X_test)

models = {"Linear": LinearRegression(), "Ridge(α=1)": Ridge(1), "Ridge(α=10)": Ridge(10)}
results = {}
for name, m in models.items():
    m.fit(X_tr_s, y_train)
    y_pred = m.predict(X_te_s)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2   = r2_score(y_test, y_pred)
    results[name] = {"rmse": rmse, "r2": r2, "pred": y_pred}
    print(f"{name:<15}: RMSE={rmse:.4f}  R²={r2:.4f}")

# ─── Visualise best model ─────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["r2"])
best_pred = results[best_name]["pred"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"House Price Prediction — {best_name}")

axes[0].scatter(y_test, best_pred, alpha=0.3, s=10, c="steelblue")
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn,mx],[mn,mx],"r--", linewidth=2)
axes[0].set_xlabel("Actual ($100k)"); axes[0].set_ylabel("Predicted ($100k)")
axes[0].set_title(f"Actual vs Predicted (R²={results[best_name]['r2']:.3f})")

residuals = y_test.values - best_pred
axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residuals"); axes[1].set_xlabel("Error")

plt.tight_layout(); plt.savefig("house_price.png", dpi=80)
print(f"\nSaved house_price.png. Best model: {best_name}")
